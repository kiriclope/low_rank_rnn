from __future__ import annotations

import torch


@torch.no_grad()
def init_dpa_internal_readout_prepost(
    model,
    mem: int = 0,
    out: int = 1,
    memory_lambda: float = 0.990,
    decision_lambda: float = 0.900,
    target_mn_corr: float = 0.5,
    target_out_mn_corr: float | None = 0.5,
    sample_scale: float = 0.8,
    test_scale: float = 0.8,
    readout_scale: float = 1.0,
    mix_strength: float = 0.10,
    readout_private_scale: float = 0.02,
    noise_scale_mn: float = 0.05,
    noise_scale_in: float = 0.05,
    rwd_input_scale: float = 1.0,
    seed: int | None = None,
    verbose: bool = True,
):
    """
    DPA initializer for internal-readout low-rank RNNs.

    Input channel convention:
        0 = A sample     1 = B sample
        2 = C test       3 = D test
        (4+ = GNG / cue / reward — left as weak random)

    Rank roles:
        mem : sample-memory rank, target eigenvalue = memory_lambda
        out : decision/readout rank, target eigenvalue = decision_lambda

    Sets corr(m_mem, n_mem) ≈ target_mn_corr and
         corr(m_out, n_out) ≈ target_out_mn_corr (if not None).

    Returns a dict of diagnostics and intermediate vectors.
    """
    assert model.rank >= 2,      "Need rank >= 2."
    assert model.input_size >= 4,"Need input channels A,B,C,D."
    assert mem != out,           "mem and out ranks must differ."
    assert 0.0 < target_mn_corr <= 1.0
    assert target_out_mn_corr is None or 0.0 < target_out_mn_corr <= 1.0
    assert 0.0 <= mix_strength <= 1.0
    assert memory_lambda > 0.0
    assert decision_lambda >= 0.0
    assert readout_scale > 0.0

    device = model.device
    N = model.hidden_size

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen)

    def zscore(v):
        v = v - v.mean()
        return v / v.std().clamp_min(1e-6)

    def orthogonalize(v, basis):
        for b in basis:
            v = v - (v @ b) / (b @ b).clamp_min(1e-6) * b
        return v

    def corr(x, y):
        x = x.detach().flatten().cpu()
        y = y.detach().flatten().cpu()
        x = x - x.mean()
        y = y - y.mean()
        return (x @ y / (x.norm() * y.norm()).clamp_min(1e-8)).item()

    # ------------------------------------------------------------------
    # 1. Sample-memory rank
    # ------------------------------------------------------------------
    u_mem = zscore(randn(N))
    p_m   = zscore(orthogonalize(randn(N), [u_mem]))
    p_n   = zscore(orthogonalize(randn(N), [u_mem, p_m]))

    # m = a*u + s*p_m, n = a*u + s*p_n  =>  n^T m / N ≈ a^2, corr(m,n) ≈ a^2/(a^2+s^2)
    a = memory_lambda ** 0.5
    s = (memory_lambda * (1.0 / target_mn_corr - 1.0)) ** 0.5
    m_mem = a * u_mem + s * p_m
    n_mem = a * u_mem + s * p_n
    # Correct finite-N deviation so C[mem,mem] is exact.
    C_mem = (n_mem @ m_mem) / N
    n_mem = n_mem * (memory_lambda / C_mem.clamp_min(1e-8))

    # ------------------------------------------------------------------
    # 2. Test direction (orthogonal to memory)
    # ------------------------------------------------------------------
    u_test = zscore(orthogonalize(randn(N), [u_mem, p_m, p_n]))
    u_mix  = zscore(orthogonalize(u_mem * u_test, [u_mem, p_m, p_n, u_test]))
    u_priv = zscore(orthogonalize(randn(N), [u_mem, p_m, p_n, u_test, u_mix]))
    u_noise= zscore(orthogonalize(randn(N), [u_mem, p_m, p_n, u_test, u_mix, u_priv]))
    u_read = zscore(mix_strength * u_mix + (1.0 - mix_strength**2)**0.5 * u_noise)

    # ------------------------------------------------------------------
    # 3. Reset all parameters weakly
    # ------------------------------------------------------------------
    model.m.normal_(0.0, noise_scale_mn)
    model.n.normal_(0.0, noise_scale_mn)
    if model.wi is not None:
        model.wi.weight.normal_(0.0, noise_scale_in)
        model.wi.bias.zero_()
    if model.wo is not None:
        model.wo.weight.normal_(0.0, noise_scale_in)
        model.wo.bias.zero_()

    # ------------------------------------------------------------------
    # 4. Install sample-memory rank
    # ------------------------------------------------------------------
    model.m[:, mem] = m_mem
    model.n[:, mem] = n_mem

    # ------------------------------------------------------------------
    # 5. Install decision/readout rank
    # ------------------------------------------------------------------
    n_out = readout_scale * u_read
    p_out = zscore(orthogonalize(u_priv.clone(), [n_out]))
    n_var = (n_out @ n_out) / N
    q_out = (decision_lambda / n_var.clamp_min(1e-8)) * n_out

    if target_out_mn_corr is not None:
        rho = float(target_out_mn_corr)
        q_var = (q_out @ q_out) / N
        private_scale = torch.sqrt(q_var * (1.0 / rho**2 - 1.0))
    else:
        private_scale = torch.as_tensor(float(readout_private_scale), device=device, dtype=q_out.dtype)

    m_out = q_out + private_scale * p_out
    # Correct finite-N deviation so C[out,out] is exact.
    C_out = (n_out @ m_out) / N
    m_out = m_out + ((decision_lambda - C_out) / n_var.clamp_min(1e-8)) * n_out

    model.n[:, out] = n_out
    model.m[:, out] = m_out

    # ------------------------------------------------------------------
    # 6. Install DPA inputs
    #    A,C -> +*+ = +   B,D -> -*- = +   A,D -> +*- = -   B,C -> -*+ = -
    # ------------------------------------------------------------------
    model.wi.weight[:, 0] = +sample_scale * u_mem   # A
    model.wi.weight[:, 1] = -sample_scale * u_mem   # B
    model.wi.weight[:, 2] = +test_scale   * u_test  # C
    model.wi.weight[:, 3] = -test_scale   * u_test  # D
    if model.input_size > 4:
        model.wi.weight[:, 4:].normal_(0.0, noise_scale_in)

    if getattr(model, "rwd", False) and rwd_input_scale != 0.0:
        model.wi.weight[:, -1] = rwd_input_scale * u_read

    # ------------------------------------------------------------------
    # 7. Diagnostics
    # ------------------------------------------------------------------
    C      = (model.n.T @ model.m) / model.hidden_size
    eig_C  = torch.linalg.eigvals(C)

    diagnostics = {
        "C":          C.detach().clone(),
        "eig_C":      eig_C.detach().clone(),
        "C_mem_mem":  C[mem, mem].item(),
        "C_out_out":  C[out, out].item(),
        "C_mem_out":  C[mem, out].item(),
        "C_out_mem":  C[out, mem].item(),
        "corr_m_mem_n_mem": corr(model.m[:, mem], model.n[:, mem]),
        "corr_m_out_n_out": corr(model.m[:, out], model.n[:, out]),
        "corr_m_mem_n_out": corr(model.m[:, mem], model.n[:, out]),
        "corr_m_out_n_mem": corr(model.m[:, out], model.n[:, mem]),
        "corr_inA_n_out":   corr(model.wi.weight[:, 0], model.n[:, out]),
        "corr_inB_n_out":   corr(model.wi.weight[:, 1], model.n[:, out]),
        "corr_inC_n_out":   corr(model.wi.weight[:, 2], model.n[:, out]),
        "corr_inD_n_out":   corr(model.wi.weight[:, 3], model.n[:, out]),
        "corr_inA_m_mem":   corr(model.wi.weight[:, 0], model.m[:, mem]),
        "corr_inB_m_mem":   corr(model.wi.weight[:, 1], model.m[:, mem]),
        "corr_inC_u_test":  corr(model.wi.weight[:, 2], u_test),
        "corr_inD_u_test":  corr(model.wi.weight[:, 3], u_test),
    }

    if verbose:
        print("Low-rank recurrent core C = n^T m / N:")
        print(C.detach().cpu())
        print("eig(C):", eig_C.detach().cpu())
        print()
        print(f"C[{mem},{mem}] = {C[mem, mem].item():+.4f}  # sample memory")
        print(f"C[{out},{out}] = {C[out, out].item():+.4f}  # decision memory")
        print(f"C[{mem},{out}] = {C[mem, out].item():+.4f}")
        print(f"C[{out},{mem}] = {C[out, mem].item():+.4f}")
        print()
        print(f"corr(m{mem}, n{mem}) = {diagnostics['corr_m_mem_n_mem']:+.4f}")
        print(f"corr(m{out}, n{out}) = {diagnostics['corr_m_out_n_out']:+.4f}")
        print(f"corr(m{mem}, n{out}) = {diagnostics['corr_m_mem_n_out']:+.4f}")
        print(f"corr(m{out}, n{mem}) = {diagnostics['corr_m_out_n_mem']:+.4f}")
        print()
        print(f"corr(inA, n{out}) = {diagnostics['corr_inA_n_out']:+.4f}")
        print(f"corr(inB, n{out}) = {diagnostics['corr_inB_n_out']:+.4f}")
        print(f"corr(inC, n{out}) = {diagnostics['corr_inC_n_out']:+.4f}")
        print(f"corr(inD, n{out}) = {diagnostics['corr_inD_n_out']:+.4f}")

    return {
        "u_mem": u_mem, "p_m": p_m, "p_n": p_n,
        "m_mem": m_mem, "n_mem": n_mem,
        "u_test": u_test, "u_mix": u_mix,
        "u_out_private": u_priv, "u_noise": u_noise,
        "u_readout": u_read, "n_out": n_out, "m_out": m_out,
        "diagnostics": diagnostics,
    }

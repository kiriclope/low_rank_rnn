"""Validate the analytic input-noise mean-field correction against Monte-Carlo E_ξ[Ψ].

Ψ_σ(κ) = Ψ_0(κ) + ½(1/N)Σ n_i φ''(a_i) s_i²   (analytic, leading order)   vs
Ψ_MC(κ) = mean over K draws of Ψ(κ; x̄+ξ), ξ~N(0,σ²)  (exact in input noise).
They must agree to leading order (MC also carries O(σ⁴)). Uses the run's real σ = noise_sigma. Read-only.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt, make_input
from src.dynamics import low_rank_numpy_params, low_rank_field_np

DEV = "cpu"
sd  = "results/dual/sweep_cue"
meta = [m for m in _load_sweep_meta(sd) if m.run_id == "s0_r3cue"][0]
model = _build_model(meta, DEV); _load_ckpt(model, sd, "expert", meta.run_id, DEV)
p = low_rank_numpy_params(model)
sigma = meta.noise_sigma()
print(f"run={meta.run_id}  nonlinearity={meta.nonlinearity}  gain={p['gain']}  noise(raw)={meta.noise}  σ_eff={sigma:.4f}")

# clean autonomous input (attention on)
ff = make_input(meta.input_size, None, 1.0, device=DEV, dtype=torch.float32)
if meta.attention_input: ff[-1] = getattr(meta, "attention_scale", 1.0)
ff = ff.detach().cpu().numpy().astype(np.float64)

# per-neuron drive variance magnitude (for context)
s2 = (p["gain"]**2) * (np.asarray(p["Ai"])**2) * (sigma**2) * np.sum(p["Wi"]**2, axis=1)
print(f"drive-variance s_i²: mean={s2.mean():.3f}  median={np.median(s2):.3f}  (so effective-gain shrink ½⟨s²⟩≈{0.5*s2.mean():.3f})")

# test points: the four memory wells (from wells3) + origin
K_test = np.array([
    [-1.15, +0.30, +0.70],   # -sample go-rule (up)
    [+1.21, +0.13, +0.64],   # +sample go-rule (up)
    [+1.26, -0.20, -0.12],   # +sample nogo (down)
    [ 0.0,  0.0,   0.0],
])
rng = np.random.default_rng(0)
K_draws = 8000
xi = sigma * rng.standard_normal((K_draws, meta.input_size))
ff_noisy = ff[None, :] + xi                                     # (K, input) noisy draws

F0 = low_rank_field_np(p, K_test, ff_input=ff,       noise_sigma=0.0)      # deterministic
Fa = low_rank_field_np(p, K_test, ff_input=ff,       noise_sigma=sigma)    # analytic Ψ_σ
Fm = low_rank_field_np(p, K_test, ff_input=ff_noisy, noise_sigma=0.0)      # MC  E_ξ[Ψ]

lbl = ["-samp go ", "+samp go ", "+samp nogo", "origin   "]
print("\n  point       Δκ2 correction (lick component):  analytic   MC        |diff|")
for i in range(len(K_test)):
    ca, cm = (Fa - F0)[i], (Fm - F0)[i]
    print(f"  {lbl[i]}  κ2:  {ca[2]:+.4f}   {cm[2]:+.4f}   {abs(ca[2]-cm[2]):.4f}"
          f"     [full analytic Δ={np.round(ca,3)}  MC Δ={np.round(cm,3)}]")

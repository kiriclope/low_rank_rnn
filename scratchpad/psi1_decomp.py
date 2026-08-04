"""Decompose the autonomous memory-well κ₁ shift into the two competing terms (ring_lowerplane_log §19):

    ½(κ₁⁺+κ₁⁻) ≈ ⟨n₁, φ(g·b_attn)⟩/N            (attention-direct — route 1, the DOWN push)
               + (g²κ₀*²/2N)·⟨n₁·φ''(g·b_attn)·m₀²⟩ (memory-modulated even coupling — UP for tanh)

computed as Ψ₁(0) and the even part ½[Ψ₁(+κ₀*,0)+Ψ₁(−κ₀*,0)] of the reduced field, with attention
clamped ON at the TRAINED amplitude (meta.attention_scale) — and, as a control, OFF (b_attn=0, which
should zero both terms for odd φ). This is the "tug of war" whose sign sets whether the wells sit
below the no-lick line. Read-only.
"""
import os, sys, collections
import numpy as np
import torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt, make_input, TIMINGS
from src.tasks import generate_dual_trials
from src.dynamics import low_rank_numpy_params, low_rank_field_np

DEV = "cpu"
dt  = TIMINGS["dual"]


@torch.no_grad()
def _well_k0(model, meta):
    """|κ₀| the trained net actually holds in the deep delay (the well radius)."""
    X, y, _, _ = generate_dual_trials(
        128, timing=dt, input_size=meta.input_size, noise=meta.noise, target_rank=meta.rank,
        cue_on_go_input=meta.cue_on_go_input, cue_scale=meta.cue_scale, nogo_target=meta.nogo_target,
        input_scale=1.0, attention_input=meta.attention_input, attention_gated=meta.attention_gated,
        attention_scale=getattr(meta, "attention_scale", 1.0))
    out = model(X.to(DEV), y.to(DEV)).cpu().numpy()
    w0, w1 = int(7.0 / dt.dt), int(8.0 / dt.dt)
    return float(np.abs(out[:, w0:w1, 0].mean(1)).mean())   # mean held |κ₀|


@torch.no_grad()
def decomp(meta, sweep_dir):
    model = _build_model(meta, DEV)
    if not _load_ckpt(model, sweep_dir, "expert", meta.run_id, DEV):
        return None
    p = low_rank_numpy_params(model)
    att = getattr(meta, "attention_scale", 1.0)
    ff_on  = make_input(meta.input_size, None, 1.0, device=DEV, dtype=torch.float32).numpy().astype(np.float64)
    if meta.attention_input:
        ff_on = ff_on.copy(); ff_on[-1] = att
    ff_off = np.zeros_like(ff_on)
    k0 = _well_k0(model, meta)

    def Psi1(k0v, ff):
        F = low_rank_field_np(p, np.array([[k0v, 0.0]]), ff_input=ff[None, :])
        return float(F[0, 1])   # Ψ₁ = F₁ + κ₁(=0)

    return dict(
        k0star   = k0,
        direct   = Psi1(0.0, ff_on),                                   # attention-direct term
        even_on  = 0.5 * (Psi1(k0, ff_on)  + Psi1(-k0, ff_on)),        # net common shift (attn ON)
        even_off = 0.5 * (Psi1(k0, ff_off) + Psi1(-k0, ff_off)),       # control (attn OFF → ≈0 for odd φ)
    )


for sweep in sys.argv[1:]:
    sd = f"results/dual/{sweep}"
    metas = _load_sweep_meta(sd)
    print(f"\n== {sweep} :: Ψ₁ decomposition (attn-direct DOWN vs net-even; wells go down if net-even<0) ==")
    arm = collections.defaultdict(list)
    for m in sorted(metas, key=lambda x: x.run_id):
        try:
            r = decomp(m, sd)
        except Exception as e:
            print(f"  {m.run_id:16s} (load failed: {type(e).__name__})"); continue
        if r is None:
            print(f"  {m.run_id:16s} (no ckpt)"); continue
        att = getattr(m, "attention_scale", 1.0)
        arm[m.run_id.split("_", 1)[1]].append(r["even_on"])
        print(f"  {m.run_id:16s} attn_scale={att:>3} κ₀*={r['k0star']:.2f} | "
              f"Ψ₁(0)={r['direct']:+.3f}  net-even_ON={r['even_on']:+.3f}  (OFF={r['even_off']:+.3f})")
    print("  " + "-" * 70)
    for tag, v in sorted(arm.items()):
        v = np.array(v)
        print(f"  arm {tag:12s} mean net-even={v.mean():+.3f}  frac<0={np.mean(v < 0):.2f}  (n={len(v)})")

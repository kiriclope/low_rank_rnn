"""λ × ρ autonomous-flow grid for a TWO-POPULATION init (rank-2 lif, at init, no training).
Construction (the A↔B reflection tying of §35): draw N/2 prototype units isotropically
(m = σ(√ρ u + √(1−ρ) p_m), n likewise, σ = √(λ/ρ), same for both modes), then mirror them into the
second half with the MEMORY mode sign-flipped and the DECISION mode copied:
    m₀,n₀ (half 2) = −m₀,−n₀ (half 1)      m₁,n₁ (half 2) = +m₁,+n₁ (half 1)
and the input columns swapped A↔B, C↔D. Consequences, exact at finite N:
    n₀ᵀm₀/N and n₁ᵀm₁/N unchanged (= λ)   ·   n₀ᵀm₁ = n₁ᵀm₀ = 0 exactly (cross-overlaps killed)
    the network is EQUIVARIANT under A↔B ⇔ the reflection κ₀ → −κ₀ with κ₁ fixed, so any pair of
    memory attractors is forced to sit at the SAME κ₁ (both above or both below, never one each).
DEC_MEAN (env, default 0) adds a mean to n₁: ⟨n₁⟩ ≠ 0 is the only thing the reflection symmetry
allows to displace the pair along κ₁ (resting κ₁ ≈ φ(0)·⟨n₁⟩ for a non-negative φ).
Usage: init_flow_grid_2pop.py [config.json]   env: DEC_MEAN, OUT
"""
import sys, os, json, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.models import LowRankModel
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from src.flow_field import low_rank_numpy_params, low_rank_field_np
cfgp = sys.argv[1] if len(sys.argv) > 1 else "results/dual/sweep_lif_dpa_lambda_scan_rho1/s0_lamscan_7/config.json"
cfg = json.load(open(cfgp)); DEC = float(os.environ.get("DEC_MEAN", "0"))
OUT = os.environ.get("OUT", f"results/figures/init_flow_grids/grid4_2pop{'_decmean' if DEC else ''}.png")
DT = cfg["dt_base"] * cfg["tau_rec_frac"]; alpha = DT / cfg["tau"]; alpha_rec = DT / (cfg["tau"] * cfg["tau_rec_frac"])
sig = cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha)); N = cfg["hidden_size"]
XLIM = (-1.5, 1.5); NGRID = 121; NFP = 41; SEED = 0
LAMS = [3.5, 5.0, 7.0, 10.0]; RHOS = [0.3, 0.5, 0.8, 1.0]
# DEC_SCAN=1: rows become the decision-mode mean ⟨n₁⟩ at a fixed ρ (ROW_RHO), columns stay λ.
DEC_SCAN = os.environ.get("DEC_SCAN", "") == "1"
DECS = [float(x) for x in os.environ.get("DECS", "0,-0.1,-0.25,-0.5").split(",")]
ROW_RHO = float(os.environ.get("ROW_RHO", "0.8"))
def build2(lam, rho, dec=0.0):
    m = LowRankModel(input_size=cfg["input_size"], hidden_size=N, output_size=0, rank=2, gain=cfg["gain"],
                     alpha=alpha, alpha_rec=alpha_rec, noise=0.0, rwd=cfg["rwd"], rwd_scale=cfg["rwd_scale"],
                     use_fixed_weights=cfg["use_fixed_weights"], fixed_weight_scale=cfg["fixed_weight_scale"],
                     fixed_weight_orthogonalize=cfg["fixed_weight_orthogonalize"], fixed_weight_sparsity=cfg["fixed_weight_sparsity"],
                     nonlinearity=cfg["nonlinearity"], nl_gamma=cfg["nl_gamma"], use_unit_bias=cfg["use_unit_bias"],
                     unit_bias_trainable=cfg["unit_bias_trainable"], unit_bias_scale=cfg["unit_bias_scale"],
                     use_rec_scale=cfg["use_rec_scale"], device="cpu", integrate=cfg.get("integrate", "both"))
    g = torch.Generator().manual_seed(SEED); h = N // 2
    def z(v):
        v = v - v.mean(); return v / v.std()
    with torch.no_grad():
        m.m.normal_(0.0, 0.05, generator=g); m.n.normal_(0.0, 0.05, generator=g)
        m.wi.weight.normal_(0.0, 1.0, generator=g); m.wi.bias.zero_()
        s = math.sqrt(lam / rho)
        for k in range(2):                                   # both modes built identically (isotropic)
            u, pm, pn = (z(torch.randn(h, generator=g)) for _ in range(3))
            mk = s * (math.sqrt(rho) * u + math.sqrt(1 - rho) * pm)
            nk = s * (math.sqrt(rho) * u + math.sqrt(1 - rho) * pn)
            nk = nk * (lam / ((nk @ mk) / h))                # exact overlap on the prototype half
            sgn = -1.0 if k == 0 else 1.0                    # memory mode mirrored, decision mode copied
            m.m[:h, k] = mk; m.m[h:, k] = sgn * mk
            m.n[:h, k] = nk; m.n[h:, k] = sgn * nk
        wi = m.wi.weight
        wi[h:, 0] = wi[:h, 1]; wi[h:, 1] = wi[:h, 0]          # A ↔ B
        wi[h:, 2] = wi[:h, 3]; wi[h:, 3] = wi[:h, 2]          # C ↔ D
        wi[h:, 4:] = wi[:h, 4:]                               # go/nogo/cue shared
        if dec: m.n[:, 1] += dec
    m.eval(); return m
def panel(ax, model, title, vmax):
    cache, _ = _flow_panel_cache(model, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"],
                                 torch.zeros(1, 1, cfg["input_size"]), XLIM, XLIM, NGRID,
                                 field_input_noise=sig, n_fp_seeds=NFP, slow_tol=0.06)
    hm = _render_flow_panel(ax, cache, speed_vmax=vmax, sim_scattered=False, kappa_traj=None,
                            cond_idx={}, colors={}, xlim=XLIM, ylim=XLIM, model=model)
    P = low_rank_numpy_params(model); M = np.asarray(P["M"]); Nv = np.asarray(P["Nvec"])
    J = cfg["gain"] * (Nv.T @ M) / N
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False); pts = np.stack([np.cos(th), np.sin(th)], 1)
    ffz = np.zeros(cfg["input_size"])
    Fp = np.stack([np.asarray(low_rank_field_np(P, p, ff_input=ffz, noise_sigma=sig)).ravel()[:2] for p in pts])
    Fm = np.stack([np.asarray(low_rank_field_np(P, -p, ff_input=ffz, noise_sigma=sig)).ravel()[:2] for p in pts])
    odd = float(np.linalg.norm(Fp + Fm) / np.linalg.norm(Fp))
    ax.set_title(title, fontsize=8); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=6)
    ax.text(0.02, 0.02, f"J {J[0,0]:.1f}/{J[1,1]:.1f}  cross {J[0,1]:+.2f}/{J[1,0]:+.2f}\nodd viol {odd:.3f}",
            transform=ax.transAxes, fontsize=5.5, color="w", va="bottom")
    return hm
ROWVALS = DECS if DEC_SCAN else RHOS
models = {(r, l): (build2(l, ROW_RHO, r) if DEC_SCAN else build2(l, r, DEC)) for r in ROWVALS for l in LAMS}
sp = []
for mdl in models.values():
    _, s_ = _flow_panel_cache(mdl, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"],
                              torch.zeros(1, 1, cfg["input_size"]), XLIM, XLIM, 61, field_input_noise=sig, n_fp_seeds=9)
    sp.append(s_)
vmax = float(np.percentile(np.concatenate(sp), 98))
fig, axes = plt.subplots(len(ROWVALS), len(LAMS), figsize=(2.9 * len(LAMS), 2.8 * len(ROWVALS)), dpi=130)
for i, r in enumerate(ROWVALS):
    for j, l in enumerate(LAMS):
        ttl = (f"λ={l:g}  ⟨n₁⟩={r:g}" if DEC_SCAN else f"λ={l:g}  ρ={r:g}  σ={math.sqrt(l/r):.2f}")
        hm = panel(axes[i, j], models[(r, l)], ttl, vmax)
        if j == 0: axes[i, j].set_ylabel((f"⟨n₁⟩ = {r:g}" if DEC_SCAN else f"ρ = {r:g}") + "\nκ₁", fontsize=8)
        if i == len(ROWVALS) - 1: axes[i, j].set_xlabel("κ₀", fontsize=8)
fig.colorbar(hm, ax=axes.ravel().tolist(), shrink=0.5, label="‖Ψ(κ) − κ‖")
fig.suptitle((f"TWO-POPULATION init, ρ = {ROW_RHO:g}: DOSE of the decision-mode mean ⟨n₁⟩ (rows) × λ (cols) — "
              f"rank-2 {cfg['nonlinearity']}, gain {cfg['gain']}, N {N}, input-noise-averaged field (σ_eff {sig:.2f})"
              ) if DEC_SCAN else f"TWO-POPULATION init (A↔B reflection tying: memory mode mirrored, decision mode copied, "
             f"A↔B / C↔D inputs swapped){'  +  ⟨n₁⟩ = ' + str(DEC) if DEC else ''} — rank-2 {cfg['nonlinearity']}, "
             f"gain {cfg['gain']}, N {N}, input-noise-averaged field (σ_eff {sig:.2f})", fontsize=9)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight"); print("saved", OUT)

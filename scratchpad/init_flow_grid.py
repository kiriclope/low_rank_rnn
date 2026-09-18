"""Autonomous flow of rank-2 lif nets AT INIT, as a function of the init parameters — rendered with the
SAME machinery/conventions as the trained-net fp_stages panels (flow_rank2._flow_panel_cache /
_render_flow_panel: input-noise-averaged field, magma speed map, streamlines, fixed-point markers).

Grid 1 (isotropic family, both modes built memory-style):  rows rho in RHOS x cols lambda in LAMS.
        sigma^2 = ||m||^2/N = ||n||^2/N = lambda/rho.  Literature (unit-variance Gaussian) = the sigma=1 cells.
Grid 2 (breaking the ring the way init.py does): memory mode fixed (lambda0, rho0); cols lambda1/lambda0;
        row A = init.py's decision construction (n1 unit variance, m1 carries the overlap);
        row B = isotropic construction (decision mode built like the memory mode).
Usage: init_flow_grid.py [config.json for the recipe]  -> results/figures/init_flow_grids/{grid1,grid2}.{png,pdf}
"""
import sys, os, json, math, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.models import LowRankModel
from src.init import init_dpa_internal_readout_prepost
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
cfgp = sys.argv[1] if len(sys.argv) > 1 else "results/dual/sweep_lif_sub_design2_w1/s0_design2_w1/config.json"
cfg = json.load(open(cfgp)); OUT = "results/figures/init_flow_grids"; os.makedirs(OUT, exist_ok=True)
DT = cfg["dt_base"] * cfg["tau_rec_frac"]; alpha = DT / cfg["tau"]; alpha_rec = DT / (cfg["tau"] * cfg["tau_rec_frac"])
sig = cfg["noise"] * math.sqrt(1 - math.exp(-2 * alpha)); N = cfg["hidden_size"]; SEED = 0
XLIM = (-1.5, 1.5); NGRID = 121; NFP = 41

def build(lam0, rho0, lam1, rho1, isotropic_decision):
    m = LowRankModel(input_size=cfg["input_size"], hidden_size=N, output_size=0, rank=2, gain=cfg["gain"],
                     alpha=alpha, alpha_rec=alpha_rec, noise=0.0, rwd=cfg["rwd"], rwd_scale=cfg["rwd_scale"],
                     use_fixed_weights=cfg["use_fixed_weights"], fixed_weight_scale=cfg["fixed_weight_scale"],
                     fixed_weight_orthogonalize=cfg["fixed_weight_orthogonalize"], fixed_weight_sparsity=cfg["fixed_weight_sparsity"],
                     nonlinearity=cfg["nonlinearity"], nl_gamma=cfg["nl_gamma"], use_unit_bias=cfg["use_unit_bias"],
                     unit_bias_trainable=cfg["unit_bias_trainable"], unit_bias_scale=cfg["unit_bias_scale"],
                     use_rec_scale=cfg["use_rec_scale"], device="cpu", integrate=cfg.get("integrate", "both"))
    init_dpa_internal_readout_prepost(m, mem=0, out=1, memory_lambda=lam0, decision_lambda=lam1,
                                      target_mn_corr=rho0, target_out_mn_corr=rho1,
                                      sample_scale=cfg["sample_scale"], test_scale=cfg["test_scale"],
                                      decision_readout_mean=cfg["decision_readout_mean"], mix_strength=cfg["mix_strength"],
                                      noise_scale_mn=1.0, noise_scale_in=1.0, rwd_input_scale=cfg["rwd_input_scale"],
                                      seed=SEED, verbose=False)
    if isotropic_decision:   # rebuild the decision mode exactly like the memory mode, orthogonal to it
        g = torch.Generator().manual_seed(SEED + 1)
        with torch.no_grad():
            basis = [m.m[:, 0] / m.m[:, 0].norm(), m.n[:, 0] / m.n[:, 0].norm()]
            def z(v):
                for b in basis: v = v - (v @ b) * b
                return (v - v.mean()) / v.std()
            u = z(torch.randn(N, generator=g)); pm = z(torch.randn(N, generator=g)); pn = z(torch.randn(N, generator=g))
            a = lam1 ** 0.5; s = (lam1 * (1.0 / rho1 - 1.0)) ** 0.5
            m1 = a * u + s * pm; n1 = a * u + s * pn
            n1 = n1 * (lam1 / ((n1 @ m1) / N))
            m.m[:, 1] = m1; m.n[:, 1] = n1
    m.eval(); return m

def panel(ax, model, title, speed_vmax):
    spec = dict(name="Autonomous", dims=None, conds=[])
    ex = torch.zeros(1, 1, cfg["input_size"])
    cache, _ = _flow_panel_cache(model, spec, cfg["input_size"], ex, XLIM, XLIM, NGRID,
                                 field_input_noise=sig, n_fp_seeds=NFP, slow_tol=0.06)   # orange ring = slow attractor (|1-max|mult|| <= 0.06): the ring picks
    hm = _render_flow_panel(ax, cache, speed_vmax=speed_vmax, sim_scattered=False, kappa_traj=None,
                            cond_idx={}, colors={}, xlim=XLIM, ylim=XLIM, model=model)
    ax.set_title(title, fontsize=8); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=6)
    M = model.m.detach().numpy(); Nv = model.n.detach().numpy(); J = cfg["gain"] * (Nv.T @ M) / N
    ax.text(0.02, 0.02, f"J00 {J[0,0]:.2f} J11 {J[1,1]:.2f}\nσm {M[:,0].std():.2f}/{M[:,1].std():.2f} σn {Nv[:,0].std():.2f}/{Nv[:,1].std():.2f}",
            transform=ax.transAxes, fontsize=5.5, color="w", va="bottom")
    return hm

def speed_cap(models):   # same convention as the stacked plotter: shared vmax = 98th percentile of all speeds
    sp = []
    for mdl in models:
        cache, s = _flow_panel_cache(mdl, dict(name="Autonomous", dims=None, conds=[]), cfg["input_size"],
                                     torch.zeros(1, 1, cfg["input_size"]), XLIM, XLIM, 61, field_input_noise=sig, n_fp_seeds=9)
        sp.append(s)
    return float(np.percentile(np.concatenate(sp), 98))

# ── Grid 1
LAMS = [1.6, 3.0, 5.0, 7.0, 10.0]; RHOS = [0.3, 0.5, 0.8, 1.0]   # critical g·λ·φ'(0)=1 ↔ λ≈2.5 (lif φ'(0)≈0.4, gain 1); trained nets sit at J≈7
models = {(r, l): build(l, r, l, r, True) for r in RHOS for l in LAMS}
vmax = speed_cap(list(models.values()))
fig, axes = plt.subplots(len(RHOS), len(LAMS), figsize=(2.9 * len(LAMS), 2.8 * len(RHOS)), dpi=130)
for i, r in enumerate(RHOS):
    for j, l in enumerate(LAMS):
        lit = " ★lit σ=1" if abs(l / r - 1.0) < 1e-9 else ""; lit += "  (sub)" if cfg["gain"] * l * 0.4 < 1 else ""
        hm = panel(axes[i, j], models[(r, l)], f"λ={l}  ρ={r}  σ={math.sqrt(l / r):.2f}{lit}", vmax)
        if j == 0: axes[i, j].set_ylabel(f"ρ = {r}\nκ₁", fontsize=8)
        if i == len(RHOS) - 1: axes[i, j].set_xlabel("κ₀", fontsize=8)
fig.colorbar(hm, ax=axes.ravel().tolist(), shrink=0.5, label="‖Ψ(κ) − κ‖")
fig.suptitle(f"Rank-2 {cfg['nonlinearity']} at INIT, isotropic construction (both modes memory-style), gain {cfg['gain']}, N {N}, "
             f"input-noise-averaged field (σ_eff {sig:.2f}) — columns λ = nᵀm/N, rows ρ = corr(m,n); σ² = λ/ρ", fontsize=9)
fig.savefig(f"{OUT}/grid1_lambda_x_rho.png", bbox_inches="tight"); fig.savefig(f"{OUT}/grid1_lambda_x_rho.pdf", bbox_inches="tight"); plt.close(fig)
print("grid1 saved")

# ── Grid 2
LAM0, RHO0 = 7.0, cfg["target_mn_corr"]; RATIOS = [0.25, 0.5, 1.0, 2.0]   # memory mode at the TRAINED overlap (J00≈7)
rows = [("init.py construction (n₁ unit-variance, m₁ carries λ₁)", False), ("isotropic construction (decision built like memory)", True)]
models2 = {(k, q): build(LAM0, RHO0, LAM0 * q, RHO0, iso) for k, (_, iso) in enumerate(rows) for q in RATIOS}
vmax2 = speed_cap(list(models2.values()))
fig, axes = plt.subplots(2, len(RATIOS), figsize=(2.9 * len(RATIOS), 2.8 * 2), dpi=130)
for k, (lab, iso) in enumerate(rows):
    for j, q in enumerate(RATIOS):
        hm = panel(axes[k, j], models2[(k, q)], f"λ₁/λ₀ = {q}  (λ₀ {LAM0}, ρ {RHO0})", vmax2)
        if j == 0: axes[k, j].set_ylabel(lab + "\nκ₁", fontsize=7)
        if k == 1: axes[k, j].set_xlabel("κ₀", fontsize=8)
fig.colorbar(hm, ax=axes.ravel().tolist(), shrink=0.6, label="‖Ψ(κ) − κ‖")
fig.suptitle(f"Breaking the ring: memory mode fixed at (λ₀ {LAM0}, ρ {RHO0}); decision mode λ₁/λ₀ across columns; "
             f"top = our init.py decision construction, bottom = isotropic", fontsize=9)
fig.savefig(f"{OUT}/grid2_anisotropy.png", bbox_inches="tight"); fig.savefig(f"{OUT}/grid2_anisotropy.pdf", bbox_inches="tight"); plt.close(fig)
print("grid2 saved")

# ── Grid 3: the σ_n split at FIXED overlaps and correlation. General construction per mode:
#   n = σ_n(√ρ·u + √(1−ρ)·p_n),  m = σ_m(√ρ·u + √(1−ρ)·p_m),  σ_m = λ/(ρ·σ_n)   (nᵀm/N = λ, corr = ρ)
# init.py's memory mode is the σ_m = σ_n case; its decision mode has σ_n = 1 (readout_scale) and σ_m = λ/ρ.
# Mean-field: n enters the field only through the overlap; σ_m sets the saturation scale Δ = σ_m²R² — so a
# LARGER σ_n (smaller σ_m) should give a LARGER radius on that mode, at identical J.
def build_general(lam0, rho, sn0, lam1, sn1):
    m = build(lam0, rho, lam1, rho, True)
    g = torch.Generator().manual_seed(SEED + 7)
    with torch.no_grad():
        basis = []
        def z(v):
            for b in basis: v = v - (v @ b) * b
            v = (v - v.mean()) / v.std(); return v
        for k, (lam, sn) in enumerate(((lam0, sn0), (lam1, sn1))):
            u = z(torch.randn(N, generator=g)); basis.append(u / u.norm())
            pn = z(torch.randn(N, generator=g)); basis.append(pn / pn.norm())
            pm = z(torch.randn(N, generator=g)); basis.append(pm / pm.norm())
            sm = lam / (rho * sn)
            n = sn * (rho ** 0.5 * u + (1 - rho) ** 0.5 * pn); mm = sm * (rho ** 0.5 * u + (1 - rho) ** 0.5 * pm)
            n = n * (lam / ((n @ mm) / N))          # exact overlap at finite N
            m.m[:, k] = mm; m.n[:, k] = n
    m.eval(); return m
LAM, RHO3 = 7.0, 0.8; SN0 = [1.0, 1.7, 2.96]; SN1 = [1.0, 1.7, 2.96, 4.0]
models3 = {(a, b): build_general(LAM, RHO3, a, LAM, b) for a in SN0 for b in SN1}
vmax3 = speed_cap(list(models3.values()))
fig, axes = plt.subplots(len(SN0), len(SN1), figsize=(2.9 * len(SN1), 2.8 * len(SN0)), dpi=130)
for i, a in enumerate(SN0):
    for j, b in enumerate(SN1):
        hm = panel(axes[i, j], models3[(a, b)], f"σn₀ {a} · σn₁ {b}", vmax3)
        if j == 0: axes[i, j].set_ylabel(f"σ(n₀) = {a}\nκ₁", fontsize=8)
        if i == len(SN0) - 1: axes[i, j].set_xlabel("κ₀", fontsize=8)
fig.colorbar(hm, ax=axes.ravel().tolist(), shrink=0.6, label="‖Ψ(κ) − κ‖")
fig.suptitle(f"The σ_n split at fixed overlaps: λ₀ = λ₁ = {LAM}, ρ = {RHO3} on both modes; rows σ(n₀), cols σ(n₁); σ(m) = λ/(ρ·σ(n)). "
             f"Diagonal = isotropic (ring); off-diagonal = init.py-like asymmetry", fontsize=9)
fig.savefig(f"{OUT}/grid3_sigma_n.png", bbox_inches="tight"); fig.savefig(f"{OUT}/grid3_sigma_n.pdf", bbox_inches="tight"); plt.close(fig)
print("grid3 saved")

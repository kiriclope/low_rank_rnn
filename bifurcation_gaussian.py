"""
Generic Gaussian low-rank RNN: decision-mode bifurcation in the κ-plane.
Populations m0,m1,n0,n1 drawn from Gaussians; overlaps set the self-gains
  g·λ0 = g⟨n0 m0⟩ (memory),  g·λ1 = g⟨n1 m1⟩ (decision).
Real reduced field  F_r(κ) = (1/N) Σ_i n_{ir} tanh(g(m_i·κ + b_i)) − κ_r.
Sweep g·λ1 across the pitchfork (g·λ0 fixed, memory bistable); rows = symmetry
break OFF (odd, b=0) vs ON (b = attention-like per-unit bias, lowers κ1).
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import root

_ap = argparse.ArgumentParser(description="Generic Gaussian low-rank decision-mode bifurcation figure.")
_ap.add_argument("--N", type=int, default=512, help="units (match hidden_size)")
_ap.add_argument("--gain", type=float, default=2.0)
_ap.add_argument("--gl0", type=float, default=4.0, help="fixed memory self-gain (deep bistable)")
_ap.add_argument("--gl1", type=float, nargs="+", default=[0.6, 1.0, 2.0, 3.5],
                 help="decision self-gains to sweep (columns)")
_ap.add_argument("--beta", type=float, default=0.5, help="symmetry-breaking bias strength (row B)")
_ap.add_argument("--nu", type=float, default=0.40, help="readout-orthogonal noise")
_ap.add_argument("--seed", type=int, default=7)
_ap.add_argument("--out", default="results/figures/theory/bifurcation_gaussian",
                 help="output path stem (.pdf/.png appended)")
_args = _ap.parse_args()

rng = np.random.default_rng(_args.seed)
N   = _args.N
G   = _args.gain
NU  = _args.nu
GL0 = _args.gl0
LIM = 2.0

# --- population (drawn ONCE; only n1's alignment changes across columns) -----
m0 = rng.standard_normal(N)
m1 = rng.standard_normal(N)
z0 = rng.standard_normal(N)    # readout-orthogonal parts
z1 = rng.standard_normal(N)
w  = rng.standard_normal(N)    # independent attention component (=> deformation)
n0 = (GL0 / G) * m0 + NU * z0  # ⟨n0 m0⟩ = GL0/G  → g·λ0 = GL0
# attention-like bias loading: mostly along -m1 (lowers κ1) + independent part (deforms,
# not a pure offset). Gentle enough to keep the memory bistability intact.
attn = -(m1 + 0.25 * w)
BETA = _args.beta              # bias strength (row B)

def make_n1(gl1):
    return (gl1 / G) * m1 + NU * z1     # ⟨n1 m1⟩ = gl1/G → g·λ1 = gl1

def field(k, n1, b):
    k0, k1 = k[..., 0], k[..., 1]
    phi = np.tanh(G * (m0 * k0[..., None] + m1 * k1[..., None] + b))   # (...,N)
    F0 = phi @ n0 / N - k0
    F1 = phi @ n1 / N - k1
    return np.stack([F0, F1], axis=-1)

def jac(k, n1, b):
    arg = G * (m0 * k[0] + m1 * k[1] + b)
    pp  = 1.0 - np.tanh(arg) ** 2                # φ'
    J = np.empty((2, 2))
    J[0, 0] = G * np.mean(n0 * pp * m0) - 1.0
    J[0, 1] = G * np.mean(n0 * pp * m1)
    J[1, 0] = G * np.mean(n1 * pp * m0)
    J[1, 1] = G * np.mean(n1 * pp * m1) - 1.0
    return J

def find_fps(n1, b):
    seeds = np.array([[x, y] for x in np.linspace(-1.4, 1.4, 8)
                              for y in np.linspace(-1.4, 1.4, 8)])
    fps = []
    for s in seeds:
        sol = root(lambda k: field(k[None, :], n1, b)[0], s, tol=1e-9)
        if not sol.success:
            continue
        k = sol.x
        if np.abs(k).max() > LIM + 0.05:
            continue
        if not any(np.hypot(*(k - f)) < 6e-2 for f in fps):
            fps.append(k)
    out = []
    for f in fps:
        ev = np.linalg.eigvals(jac(f, n1, b)).real
        kind = ("attractor" if (ev < 0).all() else
                "repeller"  if (ev > 0).all() else "saddle")
        out.append((f, kind))
    return out

STYLE = {"attractor": dict(mfc="lime", mec="k", marker="o", ms=11),
         "saddle":    dict(mfc="orange", mec="k", marker="s", ms=8),
         "repeller":  dict(mfc="white", mec="crimson", marker="o", ms=8, mew=1.5)}

def draw(ax, gl1, use_bias, show_null_ref=None):
    n1 = make_n1(gl1)
    b  = BETA * attn if use_bias else np.zeros(N)
    ax_ = np.linspace(-LIM, LIM, 55)
    K0, K1 = np.meshgrid(ax_, ax_)
    F = field(np.stack([K0, K1], axis=-1), n1, b)
    d0, d1 = F[..., 0], F[..., 1]
    ax.streamplot(ax_, ax_, d0, d1, color=np.hypot(d0, d1), cmap="magma",
                  density=1.1, linewidth=0.6, arrowsize=0.65)
    ax.axhspan(-LIM, 0, color="0.85", alpha=0.30, zorder=0)
    ax.axhline(0, color="0.5", lw=0.7, ls=":"); ax.axvline(0, color="0.5", lw=0.7, ls=":")
    ax.contour(K0, K1, d1, levels=[0], colors="deepskyblue", linewidths=1.8)
    if show_null_ref is not None:
        ax.contour(K0, K1, show_null_ref, levels=[0], colors="deepskyblue",
                   linewidths=1.2, linestyles="--", alpha=0.7)
    for f, kind in find_fps(n1, b):
        ax.plot(f[0], f[1], zorder=6, **STYLE[kind])
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xticks([-1.5, 0, 1.5]); ax.set_yticks([-1.5, 0, 1.5])
    return d1

# ---------------------------------------------------------------------------
GL1S = list(_args.gl1)
NCOL = len(GL1S)
fig = plt.figure(figsize=(3.9 * NCOL, 8.4))
gs = gridspec.GridSpec(2, NCOL, hspace=0.16, wspace=0.14,
                       left=0.075, right=0.985, top=0.845, bottom=0.085)

# Row A: odd baseline (b=0). Keep each column's odd nullcline for the row-B overlay.
odd_nulls = []
for j, gl1 in enumerate(GL1S):
    ax = fig.add_subplot(gs[0, j])
    odd_nulls.append(draw(ax, gl1, use_bias=False))
    if j == 0:
        ax.set_ylabel("break OFF ($b=0$, odd)\n\n$\\kappa_1$", fontsize=11)

# Row B: symmetry break ON (attention-like bias) — lowers + deforms.
for j, gl1 in enumerate(GL1S):
    ax = fig.add_subplot(gs[1, j])
    draw(ax, gl1, use_bias=True, show_null_ref=odd_nulls[j])
    ax.set_xlabel(r"$\kappa_0$", fontsize=11)
    if j == 0:
        ax.set_ylabel("break ON (bias)\n\n$\\kappa_1$", fontsize=11)

def _regime(gl1):
    if gl1 < 0.85:  return "decision subcritical"
    if gl1 < 1.2:   return "decision critical"
    if gl1 < GL0 * 0.8: return "decision supercritical"
    return "decision over-driven"
for j, gl1 in enumerate(GL1S):
    xc = 0.075 + (j + 0.5) * (0.985 - 0.075) / NCOL
    fig.text(xc, 0.885, rf"$g\lambda_1 = {gl1:.1f}$", ha="center", va="center", fontsize=12.5)
    fig.text(xc, 0.862, _regime(gl1), ha="center", va="center", fontsize=9, color="0.35")

fig.suptitle("Generic Gaussian low-rank RNN — the decision mode's pitchfork ($g\\lambda_1{=}1$) sets how a "
             "fixed bias lowers the memory wells\n"
             f"($N={N}$,  $g={G}$,  deep memory $g\\lambda_0={GL0}$;  "
             "$F_r=\\frac{1}{N}\\sum_i n_{ir}\\tanh(g(m_i\\!\\cdot\\!\\kappa+b_i))-\\kappa_r$;  "
             "near-critical decision $\\Rightarrow$ soft mode $\\Rightarrow$ bias strongly lowers)",
             fontsize=11.5, y=0.975)
from matplotlib.lines import Line2D
h = [Line2D([0],[0], **{**STYLE["attractor"],"ls":""}, label="attractor"),
     Line2D([0],[0], **{**STYLE["saddle"],"ls":""}, label="saddle"),
     Line2D([0],[0], **{**STYLE["repeller"],"ls":""}, label="repeller"),
     Line2D([0],[0], color="deepskyblue", lw=2, label=r"$F_1=0$ nullcline"),
     Line2D([0],[0], color="deepskyblue", lw=1.4, ls="--", label="odd nullcline (b=0)")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))

os.makedirs(os.path.dirname(_args.out) or ".", exist_ok=True)
fig.savefig(f"{_args.out}.pdf", bbox_inches="tight")
fig.savefig(f"{_args.out}.png", dpi=140, bbox_inches="tight")
print("saved", f"{_args.out}.png")
# report well positions
for gl1 in GL1S:
    n1 = make_n1(gl1)
    for tag, b in [("odd", np.zeros(N)), ("bias", BETA*attn)]:
        atts = [f for f, k in find_fps(n1, b) if k == "attractor"]
        s = "  ".join(f"({f[0]:+.2f},{f[1]:+.2f})" for f in atts)
        print(f"  gλ1={gl1:.1f} {tag:4s}: {len(atts)} attractors  {s}")

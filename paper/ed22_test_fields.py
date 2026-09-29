"""Extended Data Fig. 22 — the choice lives in the test-driven fields.
The pairing choice is scored in the last 0.5 s of the test, while the test odor is on, so the choice states the task
requires are states of the C- and D-driven fields, not wells of the autonomous field (those are optional). Each element
of the DPA group maps a trial onto another trial type (inputs and noise relabeled), which fixes how the four end-of-test
states are related: σ₁ (A↔B, C↔D) B·D = D₁ A·C; σ₂ (A↔B alone, C fixed) B·C = −A·C, i.e. the C field is odd; σ₃ (C↔D
alone) A·D = D₃ A·C, i.e. D mirrors C across the line. Under the whole group the four states are one orbit, the
quadruple (±u, ±w), split between the two test odors.
a  the scheme per element; b  a whole-group-tied network at the DPA checkpoint: the input-noise-averaged field with no
input, with C on and with D on, and the four trial types' mean state at the end of the test; c  the three relations
scored on every network (paper/test_field_residuals.py): exact zeros where the tie forces them, and after release.
Env: TFJ (residual json), EXV=<sweep>:<rid> (example network). Run: LD_PRELOAD=... python ed22_test_fields.py"""
import sys, os, json, numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
torch.set_num_threads(8)
sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad'); sys.path.insert(0, '/home/leon/rnn/paper')
from style import *
from symtools import load_run, sig_of
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from test_field_residuals import four_states
os.chdir('/home/leon/rnn')
PLUM = '#AA4499'; VCOL = '0.2'; ECOL = {'σ₁': B_COL, 'σ₂': PLUM, 'σ₃': LICK_COL}
D = {'σ₁': np.diag([-1., 1.]), 'σ₂': -np.eye(2), 'σ₃': np.diag([1., -1.])}
IMG = {'σ₁': ('B', 'D'), 'σ₂': ('B', 'C'), 'σ₃': ('A', 'D')}          # where each element sends the (A, C) trial
RES = json.load(open(os.environ.get('TFJ', '/home/leon/dual/figures/paper_share/modelling/test_fields_log.json')))
EXV = os.environ.get('EXV', 'results/dual/sweep_lif_log:s2_lg_klein')
XL = (-1.5, 1.5)
def mark(ax, p, s, t, ms=5.5, **kw):
    """a trial state: sample colour (A indigo, B teal), test odor by marker (C circle, D square), lick states filled"""
    col = A_COL if s == 'A' else B_COL; lick = (s == 'A') == (t == 'C')
    ax.plot(*p, 'o' if t == 'C' else 's', ms=ms, mfc=col if lick else 'white', mec=col, mew=1.2, zorder=7, **kw)

RES2_ON = bool(os.environ.get('TFJ2'))
fig = plt.figure(figsize=(9.6, 12.0 if RES2_ON else 9.4))
gs = fig.add_gridspec(4 if RES2_ON else 3, 20, height_ratios=[0.62, 0.95, 0.9, 0.9][:4 if RES2_ON else 3], hspace=(0.55 if RES2_ON else 0.42), wspace=2.2, left=0.06, right=0.985, top=0.965, bottom=0.06)

# ── a: the scheme ──
ax = fig.add_subplot(gs[0, 0:6]); ax.axis('off'); ax_a = ax
ax.set_title('the choice is made during the test', loc='left', fontsize=TITLE_FS)
ax.text(0.0, 0.93, 'The pairing choice is scored in the last 0.5 s\nof the test, while the odor is on: the choice\n'
        'states the task requires belong to the C- and\nD-driven fields. Autonomous choice wells are\noptional.\n\n'
        'Each element maps a trial onto another trial\ntype, inputs and noise relabeled, so it fixes\n'
        'how the four end-of-test states are related.\nUnder the whole group they are one orbit, the\n'
        'quadruple, split between the two test odors.', transform=ax.transAxes, va='top', fontsize=SMALL, color='0.3')
sg = gs[0, 6:20].subgridspec(1, 4, wspace=0.45); axs_a = [fig.add_subplot(sg[0, k]) for k in range(4)]
p0 = np.array([0.62, 0.66])
NOTE = {'σ₁': 'B·D = D₁ A·C:\nthe lick states mirror', 'σ₂': 'B·C = −A·C:\nthe C field is odd',
        'σ₃': 'A·D = D₃ A·C:\nD mirrors C', 'V': 'one quadruple,\nsplit between C and D'}
TIT = {'σ₁': 'σ₁  A↔B and C↔D', 'σ₂': 'σ₂  A↔B alone', 'σ₃': 'σ₃  C↔D alone', 'V': 'V  the whole group'}
for ax, nm in zip(axs_a, ('σ₁', 'σ₂', 'σ₃', 'V')):
    col = ECOL.get(nm, VCOL)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(0.9)
    ax.axvline(0, color='0.88', lw=0.6); ax.axhline(0, color=LICK_COL, lw=0.8, ls='--')
    mark(ax, p0, 'A', 'C'); ax.text(p0[0] + 0.12, p0[1] + 0.13, 'A·C', fontsize=SMALL, color=A_COL, ha='left')
    for el in (('σ₁', 'σ₂', 'σ₃') if nm == 'V' else (nm,)):
        q = D[el] @ p0; s, t = IMG[el]; mark(ax, q, s, t)
        ax.annotate('', q, p0, arrowprops=dict(arrowstyle='->', color=ECOL[el], lw=1.0, linestyle=':', shrinkA=5, shrinkB=5))
        ax.text(q[0] + (0.12 if q[0] > 0 else -0.12), q[1] + (0.13 if q[1] > 0 else -0.2), f'{s}·{t}', fontsize=SMALL,
                color=A_COL if s == 'A' else B_COL, ha='left' if q[0] > 0 else 'right')
    ax.set_title(TIT[nm], loc='left', fontsize=TITLE_FS, color=col)
    ax.text(0.5, -0.06, NOTE[nm], transform=ax.transAxes, ha='center', va='top', fontsize=SMALL, color=col)
axs_a[0].text(1.3, 0.1, 'lick', fontsize=SMALL * 0.9, color=LICK_COL, ha='right'); axs_a[0].text(1.3, -0.25, 'no lick', fontsize=SMALL * 0.9, color=LICK_COL, ha='right')

# ── b: a whole-group-tied network at the DPA checkpoint ──
sw, rid = EXV.split(':'); m, cfg = load_run(sw, rid, stage='dpa', device='cpu'); sig = sig_of(cfg)
S = four_states(m, cfg)
sgb = gs[1, 0:20].subgridspec(1, 3, wspace=0.28); axs_b = [fig.add_subplot(sgb[0, k]) for k in range(3)]
for ax, (lab, dims) in zip(axs_b, (('no input (autonomous)', []), ('test C on', [2]), ('test D on', [3]))):
    cache, spd = _flow_panel_cache(m, dict(name=lab, dims=dims, conds=[]), cfg['input_size'], torch.zeros(1, 1, cfg['input_size']),
                                   XL, XL, 61, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
    _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={},
                       colors={}, xlim=XL, ylim=XL, model=m)
    ax.axhline(0, color=LICK_COL, lw=0.8, ls='--', zorder=6)
    for (s, t), p in S.items():
        if not dims or t == ('C' if dims == [2] else 'D'): mark(ax, p, s, t, ms=6.5)
    ax.set_title(lab, loc='left', fontsize=TITLE_FS); ax.set_xlabel('$\\kappa_0$ sample'); ax.set_aspect('equal')
    print('flow', lab, 'done', flush=True)
axs_b[0].set_ylabel('$\\kappa_1$ choice')
axs_b[1].text(0.5, -0.2, f'{rid} (whole group tied, λ = 7), DPA checkpoint, input-noise-averaged field.  Markers: mean state at the end of the test '
          '(circle: test C, square: test D; indigo A, teal B; filled: a lick trial).', transform=axs_b[1].transAxes, ha='center', va='top', fontsize=SMALL, color='0.3')

# ── c (λ = 7) and d (λ = 2): the relations scored on every network ──
YMAX = 0.55
def relations_row(axs_c, RES, arms, fail, lam):
    GROUPS = [('σ₁ tied', arms[0], 'σ₁'), ('σ₂ tied', arms[1], 'σ₂'), ('σ₃ tied', arms[2], 'σ₃'), ('V tied', arms[3], 'V'), ('free', arms[4], None)]
    for ax, st, ttl in zip(axs_c, ('dpa', 'expert'), (f'λ = {lam}: after DPA — the tie held', f'λ = {lam}: after Dual — the tie released')):
        for gi, (glab, arm, el) in enumerate(GROUPS):
            keys = sorted(k for k in RES if k.endswith(f'|{st}') and k.split('|')[0].split('_', 1)[1] == arm)
            for ri, (rk, rel) in enumerate((('r1', 'σ₁'), ('r2', 'σ₂'), ('r3', 'σ₃'))):
                x0 = gi * 4 + ri; forced = (el == rel) or (el == 'V')
                if forced and st == 'dpa': ax.add_patch(plt.Rectangle((x0 - 0.42, -0.012), 0.84, 0.035, color=ECOL[rel], alpha=0.18, lw=0))
                for j, k in enumerate(keys):
                    v = RES[k][rk]; bad = k.split('|')[0] in fail
                    xx = x0 + (j - (len(keys) - 1) / 2) * 0.09
                    if v > YMAX: ax.plot(xx, YMAX - 0.01, '^', ms=3.5, color='0.6' if bad else ECOL[rel], alpha=0.8, zorder=4)
                    else: ax.plot(xx, v, 'o', ms=3.2, mfc='white' if bad else ECOL[rel], mec='0.6' if bad else ECOL[rel], mew=0.8, alpha=0.9, zorder=4)
            ax.text(gi * 4 + 1, -0.075, glab, ha='center', va='top', fontsize=SMALL, color=ECOL.get(el, VCOL) if el else '0.3', transform=ax.get_xaxis_transform())
        ax.set_xlim(-0.8, len(GROUPS) * 4 - 1.2); ax.set_ylim(-0.02, YMAX); ax.set_xticks([])
        ax.set_title(ttl, loc='left', fontsize=TITLE_FS); ax.axhline(0, color='0.8', lw=0.6)
    axs_c[0].set_ylabel('relation residual\n(fraction of |κ|)'); axs_c[1].set_yticklabels([])
RES2 = json.load(open(os.environ['TFJ2'])) if os.environ.get('TFJ2') else None
nrow = 4 if RES2 else 3
sgc = gs[2, 0:20].subgridspec(1, 2, wspace=0.12); axs_c = [fig.add_subplot(sgc[0, k]) for k in range(2)]
relations_row(axs_c, RES, ('lg_pair', 'lg_inv', 'lg_test', 'lg_klein', 'log'), {'s0_log', 's7_log'}, 7)
for rel, col in (('σ₁', B_COL), ('σ₂', PLUM), ('σ₃', LICK_COL)): axs_c[0].plot([], [], 'o', ms=4, color=col, label=f'{rel} relation')
axs_c[0].legend(loc='upper left', frameon=False, fontsize=SMALL, ncol=3, handletextpad=0.2, columnspacing=0.8)
axs_c[0].text(0.02, 0.84, 'shaded: the relation the tie forces (exact 0)\nopen grey: the two free seeds that failed DPA\n▲ off scale (> 0.55)',
              transform=axs_c[0].transAxes, ha='left', va='top', fontsize=SMALL * 0.9, color='0.35')
letters = [(ax_a, 'a'), (axs_b[0], 'b'), (axs_c[0], 'c')]
if RES2:
    sgd = gs[3, 0:20].subgridspec(1, 2, wspace=0.12); axs_d = [fig.add_subplot(sgd[0, k]) for k in range(2)]
    relations_row(axs_d, RES2, ('ls_pair', 'ls_inv', 'ls_test', 'ls_klein', 'lsub'), set(), 2); letters.append((axs_d[0], 'd'))
for a, L in letters: panel_letter(fig, a, L, x=0.012)
save(fig, os.environ.get('STEM', 'ed22'))

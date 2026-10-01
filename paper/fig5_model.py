"""Fig. 5 of the paper — the circuit model of the gated no-lick repositioning, in the house style.
a  the model and its plane; b  accuracy across the curriculum (n seeds); c  the flow at the three checkpoints (one seed,
all seeds' memory wells overlaid); d  per-seed well depth DPA → Dual (the push); e  the four-group and the wells each tie
predicts; f  the wells of the tied networks at the DPA checkpoint; g  after the whole curriculum, per tie; h  populations.
Env: FREE=<sweep>:<arm>:<seeds>  TIES=pair:<sweep>,klein:<sweep>,inv:<sweep>,test:<sweep>   (defaults below)
Run: LD_PRELOAD=... python fig5_model.py"""
import sys, os, json, glob, numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad'); sys.path.insert(0, '/home/leon/rnn/paper')
from style import *
RED, BLUE, GREEN, GREY, STAGE_SHADE = '#d62728', '#1f77b4', '#2ca02c', '#555555', '#332288'   # Fig. 1's colours (overlaps/fig_behavior_main.py)
COND_COL = {'DPA': RED, 'Go': BLUE, 'NoGo': GREEN}
from scipy import stats
from symtools import load_run, sig_of, field_fn, residuals, disk
from src.flow_rank2 import _flow_panel_cache, _render_flow_panel
from bifurcation_probe import find_wells
FREE = os.environ.get('FREE', 'results/dual/sweep_lif_recipe7_bfix:recipe7:0,1,2,3,4,5,6,7'); fsw, farm, fseeds = FREE.split(':'); fseeds = [int(x) for x in fseeds.split(',')]
TIES = dict(kv.split(':') for kv in os.environ.get('TIES', 'pair:results/dual/sweep_lif_symdpa,klein:results/dual/sweep_lif_symdpa,inv:results/dual/sweep_lif_symdpa,test:results/dual/sweep_lif_symdpa').split(','))
os.chdir('/home/leon/rnn'); XL = (-1.5, 1.5)
import hashlib, pickle
_WC = '/home/leon/dual/figures/paper_share/modelling/.wells_cache.pkl'; _wells_cache = pickle.load(open(_WC, 'rb')) if os.path.exists(_WC) and not os.environ.get('NOCACHE') else {}
def wells_of(m, cfg, sig):
    """stable fixed points of the noise-averaged field; cached on the parameters' hash (find_wells is the slow step, ~30 s per network)"""
    key = hashlib.md5(np.concatenate([m.m.detach().cpu().numpy().ravel(), m.n.detach().cpu().numpy().ravel(), [sig]]).tobytes()).hexdigest()
    if key not in _wells_cache:
        _wells_cache[key] = [f for f, kd, t in find_wells(m, cfg, xlim=2.5, n_seeds=41, noise_sigma=sig, with_eigs=True) if str(kd).lower().startswith(('stable', 'attract')) and np.linalg.norm(f) > 0.15]
        pickle.dump(_wells_cache, open(_WC, 'wb'))
    return _wells_cache[key]
from src.tasks import make_timings, generate_dpa_trials
from bifurcation_probe import run_dt_alpha
def memory_pair(w, m=None, cfg=None):
    """the A and B memory wells = the attractors nearest the mean end-of-delay state of A and B trials (the occupied
    wells, as in readout_arm.py); falls back to the lowest well on each side if no trials can be run"""
    L = [f for f in w if f[0] < -0.5]; R = [f for f in w if f[0] > 0.5]
    if m is None or not (L and R): return (min(R, key=lambda f: f[1]) if R else None), (min(L, key=lambda f: f[1]) if L else None)
    dt, alpha, _ = run_dt_alpha(cfg); T = make_timings(dt)['dpa']; eta = cfg['noise'] * float(np.sqrt(1 - np.exp(-alpha) ** 2)); torch.manual_seed(0)
    X, y = generate_dpa_trials(48, T, cfg['input_size'], noise=eta, target_rank=2, windowed_targets=True, decay_to_zero=False, response_in_cue=cfg['response_in_cue'], prelick_free=True, hold_window=0.5, hold_anchor='sample', post_response_window=cfg.get('dpa_post_response_window'))
    with torch.no_grad(): k = m(X)[:, int(T.n_stim_on[1]) - 1].numpy()
    isA = X[:, int(T.n_stim_on[0]):int(T.n_stim_off[0]), 0].mean(1).numpy() > 0.5
    kA, kB = k[isA].mean(0), k[~isA].mean(0)
    near = lambda pt: min(w, key=lambda f: np.linalg.norm(np.asarray(f) - pt))
    return near(kA), near(kB)
BELOW = -0.25   # "below the line" = both memory wells lower than a quarter of the noise s.d.
def acc_of(sw, rid):
    for l in open(f'{sw}/results.jsonl'):
        d = json.loads(l)
        if d['run_id'] == rid: return d['accuracy']
fig = plt.figure(figsize=(9.6, 15.0)); gs = fig.add_gridspec(6, 20, height_ratios=[1.1, 0.6, 0.72, 0.7, 0.7, 0.9], hspace=0.5, wspace=2.4, left=0.06, right=0.985, top=0.975, bottom=0.035)
def cell(r, c0, c1): return fig.add_subplot(gs[r, c0:c1])
stages = ['after DPA', 'after GNG', 'after Dual']

# ── a: the model, drawn as the network of the Neuron paper's Fig. 1B (neuron_symmetry/figures/fig1_framework.py; Leon
#      2026-10-01: "change the network's scheme to something similar to the first fig in ~/neuron_symmetry"): a population of
#      units (colored by sample preference), the three input channels, the rank-2 recurrence, the two readouts ──
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon
ax = cell(0, 0, 6); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect('equal', adjustable='box'); ax.set_anchor('W')
ax.set_title('a rank-2 recurrent network', loc='left', fontsize=TITLE_FS)
CX, CY, CR = 5.3, 4.9, 2.2                                 # the population: a true circle (square panel, equal axes; Leon 2026-10-01)
ax.add_patch(Circle((CX, CY), CR, fc='0.95', ec='0.7', lw=0.8))
_rng = np.random.default_rng(4); _pts = []
while len(_pts) < 26:
    _p = _rng.uniform([CX - CR, CY - CR], [CX + CR, CY + CR])
    if np.hypot(*(_p - (CX, CY))) < CR - 0.32 and all(np.hypot(*(_p - q)) > 0.5 for q in _pts): _pts.append(_p)
for _p in _pts: ax.add_patch(Circle(_p, 0.19, fc=A_COL if _p[0] < CX else B_COL, ec='none', alpha=0.85))
def _edge(ang): return (CX + CR * np.cos(ang), CY + CR * np.sin(ang))
for y, lab, col in ((6.5, 'sample\nA, B', A_COL), (4.9, 'Go/NoGo,\ncue', '#4878d0'), (3.3, 'test\nC, D', '0.4')):
    ang = np.pi - 0.85 * (y - CY) / CR
    ax.add_patch(FancyArrowPatch((2.0, y), _edge(ang), arrowstyle='-|>', mutation_scale=7, lw=0.9, color=col, shrinkA=0, shrinkB=1))
    ax.text(0.05, y, lab, ha='left', va='center', fontsize=SMALL, color=col, linespacing=1.0)
ax.add_patch(FancyArrowPatch(_edge(np.pi / 2 + 0.42), _edge(np.pi / 2 - 0.42), connectionstyle='arc3,rad=-0.75', arrowstyle='-|>', mutation_scale=7, lw=0.9, color='0.15'))
ax.text(CX, 8.05, '$W = (m_0 n_0^{\\mathrm{T}} + m_1 n_1^{\\mathrm{T}})/N$', ha='center', va='bottom', fontsize=SMALL)
for y, lab, ang in ((5.75, '$\\kappa_0$\nsample', 0.35), (4.05, '$\\kappa_1$\nchoice', -0.35)):
    ax.add_patch(FancyArrowPatch(_edge(ang), (8.4, y), arrowstyle='-|>', mutation_scale=7, lw=0.9, color='0.15', shrinkA=1, shrinkB=0))
    ax.text(8.55, y, lab, va='center', ha='left', fontsize=SMALL, linespacing=1.0)
ax.text(8.55, 2.95, 'lick iff\n$\\kappa_1 > 0$', va='center', ha='left', fontsize=SMALL, color=LICK_COL, linespacing=1.0)
ax.text(CX, 2.1, '$N$ = 1024 units; unit $i$: loadings\n$m_{0i}, m_{1i}, n_{0i}, n_{1i}$, input weights $w_i$', ha='center', va='top', fontsize=STAT_FS, color='0.4')

# ── b: the curriculum, drawn as the task of Fig. 1a (dual/overlaps/fig_behavior_main.py: dual_task_scheme.svg colors; Leon
#      2026-10-01: "for the curriculum use something similar to the task in fig1 of this paper"): the three stages as colored
#      boxes in training order (as Fig. 1a's curriculum), each beside its trial — odor pairs as diagonally split boxes, the cue
#      yellow, the response a lick (drop) or no lick (crossed drop) — with what the stage trains and freezes ──
ax = cell(0, 6, 20); ax.axis('off'); ax.set_xlim(-3.6, 12.4); ax.set_ylim(-1.7, 10.5); ax_b = ax; ax.set_title('the curriculum', loc='left', fontsize=TITLE_FS)
SA, SB, TC, TD, GO_, NG_, CUE_, DROP = '#332288', '#44AA99', '#ffaaaa', '#666666', '#4878d0', '#6acc64', '#ffc000', '#2797d1'
STAGE_COL = {'DPA': '#d62728', 'GNG': '#1f77b4', 'Dual': '#ff7f0e'}
TH = 1.1                                                    # timeline (tube) height
def split_box(x0, x1, y, c1, c2):
    """an odor-pair box split on the diagonal, as Fig. 1a: first odor upper left, second lower right"""
    y0, y1 = y - TH / 2, y + TH / 2
    ax.add_patch(Polygon([(x0, y0), (x0, y1), (x1, y1)], closed=True, fc=c1, ec='none', zorder=3))
    ax.add_patch(Polygon([(x0, y0), (x1, y0), (x1, y1)], closed=True, fc=c2, ec='none', zorder=3))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, TH, fc='none', ec='0.15', lw=0.7, zorder=4))
def drop(x, y, r=0.17, crossed=False):
    ax.add_patch(Circle((x, y), r, fc=DROP, ec='none', zorder=5)); ax.add_patch(Polygon([(x - r * 0.96, y + r * 0.28), (x, y + r * 2.1), (x + r * 0.96, y + r * 0.28)], closed=True, fc=DROP, ec='none', zorder=5))
    if crossed:
        for sgn in (1, -1): ax.plot([x - 1.6 * r, x + 1.6 * r], [y + r * 0.6 - sgn * 1.6 * r, y + r * 0.6 + sgn * 1.6 * r], color='k', lw=1.0, zorder=6)
def tube(y, T):
    for yy in (y - TH / 2, y + TH / 2): ax.plot([0, T], [yy, yy], color='0.15', lw=0.9, zorder=2)
    for xx in (0, T): ax.plot([xx, xx], [y - TH / 2, y + TH / 2], color='0.15', lw=0.9, zorder=2)
    ax.text(T + 0.15, y, f'{T:g} s', fontsize=STAT_FS, color='0.4', va='center', ha='left')
def label(x, y, t): ax.text(x, y + TH / 2 + 0.1, t, ha='center', va='bottom', fontsize=STAT_FS, color='0.25')
ROWS = [('DPA', 8.7, 11.0, 'all parameters free · 250 epochs'), ('GNG', 5.45, 6.0, '$m_0$, $n_0$ and the sample/test inputs frozen · 100 epochs'),
        ('Dual', 2.2, 11.0, 'all inputs frozen · 150 epochs')]
for k_, (nm, y, T, what) in enumerate(ROWS):
    ax.add_patch(FancyBboxPatch((-3.45, y - 0.5), 2.45, 1.0, boxstyle='round,pad=0,rounding_size=0.25', fc=STAGE_COL[nm], ec='0.1', lw=1.0, zorder=3))
    ax.text(-2.225, y, nm, ha='center', va='center', fontsize=PS * 8, color='w', fontweight='bold', zorder=4)
    if k_ < 2: ax.add_patch(FancyArrowPatch((-2.225, y - 0.55), (-2.225, ROWS[k_ + 1][1] + 0.55), arrowstyle='-|>', mutation_scale=9, lw=1.3, color='0.1'))
    tube(y, T)
    if nm in ('DPA', 'Dual'):
        split_box(2, 3, y, SA, SB); label(2.5, y, 'sample A/B' if nm == 'DPA' else 'A/B')
        split_box(8, 9, y, TC, TD); label(8.5, y, 'test C/D' if nm == 'DPA' else 'C/D')
        drop(9.5, y - 0.1); ax.text(10.0, y - 0.02, 'or', fontsize=STAT_FS * 0.9, ha='center', va='center', color='0.3'); drop(10.5, y - 0.1, crossed=True)
    if nm in ('GNG', 'Dual'):
        g0, c0 = (2, 4) if nm == 'GNG' else (4, 6)
        split_box(g0, g0 + 1, y, GO_, NG_); label(g0 + 0.5, y, 'Go/NoGo')
        ax.add_patch(Rectangle((c0, y - TH / 2), 0.5, TH, fc=CUE_, ec='0.15', lw=0.7, zorder=3)); label(c0 + 0.25, y, 'cue')
        if nm == 'GNG': drop(c0 + 0.85, y - 0.12, r=0.16); drop(c0 + 1.45, y - 0.12, r=0.16, crossed=True)
    ax.text(0, y - TH / 2 - 0.15, what, fontsize=STAT_FS, color='0.35', va='top')
ax.text(0, -0.55, f'drop, a lick; crossed, no lick (the sign of $\\kappa_1$)\n{len(fseeds)} networks from {len(fseeds)} random initializations',
        fontsize=STAT_FS, color='0.35', ha='left', va='top')

# ── c: the four-group of DPA — left, the task table and the three relabelings; right, one sub-panel per element and one for the
#      whole group: the action on the plane (a state and its images) and the memory pair it allows ──
PLUM = '#AA4499'; TCOL = '0.35'; VCOL = '0.2'
ax = fig.add_subplot(gs[1:3, 0:6]); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(-11.5, 10); ax_e0 = ax
ax.set_title('the DPA task and its symmetries', loc='left', fontsize=TITLE_FS)
for i_, r_ in enumerate(['A', 'B']):
    for j_, c_ in enumerate(['C', 'D']):
        lick = (i_ == j_); x, y = 3.0 + j_ * 2.2, 8.4 - (i_ + 1) * 1.0
        ax.add_patch(Rectangle((x, y), 2.2, 1.0, fc=(RED if lick else '0.92'), ec='w', lw=0.8, alpha=(0.85 if lick else 1)))
        ax.text(x + 1.1, y + 0.5, 'lick' if lick else 'no lick', ha='center', va='center', fontsize=SMALL, color=('w' if lick else '0.35'))
    ax.text(2.8, 8.4 - (i_ + 0.5) * 1.0, f'sample {r_}', ha='right', va='center', fontsize=SMALL, color=(A_COL if r_ == 'A' else B_COL), fontweight='bold')
for j_, c_ in enumerate(['C', 'D']): ax.text(3.0 + (j_ + 0.5) * 2.2, 8.55, f'test {c_}', ha='center', va='bottom', fontsize=SMALL, color=TCOL, fontweight='bold')
ax.text(0.3, 5.7, 'lick iff the pair matches; three relabelings\nleave the objective unchanged:', fontsize=SMALL, va='top', color='0.3')
for k_, (nm, what, col) in enumerate([('σ₁', 'A↔B and C↔D, response kept', B_COL), ('σ₂', 'A↔B alone, response flipped', PLUM), ('σ₃', 'C↔D alone, response flipped', LICK_COL)]):
    ax.text(0.3, 3.9 - k_ * 1.0, nm, fontsize=PS*8, color=col, fontweight='bold', va='top'); ax.text(1.6, 3.9 - k_ * 1.0, what, fontsize=SMALL, va='top')
ax.text(0.3, 0.4, 'With the identity e they form the Klein\nfour-group V = Z₂ × Z₂ (σ₁σ₂ = σ₃).', fontsize=SMALL, va='top', color='0.3')
ax.text(0.3, -2.6, 'Upper row: what each element predicts\nfor the two memory wells.', fontsize=SMALL, va='top', color='0.3')
ax.text(0.3, -6.9, 'Lower row: the flow of a network\nsymmetric under that element only\n(constructed; rings, attractors;\ncrosses, saddles).', fontsize=SMALL, va='top', color='0.3')
ELEMS = [('σ₁', 'A↔B and C↔D', [np.diag([-1, 1])], B_COL, 'mirror pair, one height;\ninvariant: the κ₁ axis'),
         ('σ₂', 'A↔B,\nlick↔no lick', [-np.eye(2)], PLUM, 'antipodal pair;\ninvariant: the origin'),
         ('σ₃', 'C↔D,\nlick↔no lick', [np.diag([1, -1])], LICK_COL, 'on the κ₀ axis, a′ free;\ninvariant: the κ₀ axis'),
         ('V', 'the whole group', [np.diag([-1, 1]), -np.eye(2), np.diag([1, -1])], VCOL, 'pinned pair (±a, 0);\ninvariant: both axes')]
def draw_matrix(ax, D, x, y, label, col='0.3'):
    """a 2×2 matrix with its entries, drawn at axes coordinates (x, y) = top-left, with bracket strokes"""
    ax.text(x, y - 0.075, label, transform=ax.transAxes, ha='left', va='center', fontsize=SMALL, color=col)
    x0 = x + 0.17; w, h = 0.13, 0.085
    for i_ in range(2):
        for j_ in range(2): ax.text(x0 + 0.06 + j_ * w, y - 0.035 - i_ * h, f'{int(D[i_, j_]):d}'.replace('-', '−'), transform=ax.transAxes, ha='center', va='center', fontsize=SMALL, color=col)
    for xb, d_ in ((x0 - 0.01, 1), (x0 + 0.12 + w, -1)):
        ax.plot([xb + 0.02 * d_, xb, xb, xb + 0.02 * d_], [y + 0.01, y + 0.01, y - 0.08 - h + 0.005, y - 0.08 - h + 0.005], transform=ax.transAxes, color=col, lw=0.8, clip_on=False)
ICOL = [B_COL, PLUM, LICK_COL]
TA = os.environ.get('TIEARM', 'symdpa_{}')   # tie arm name pattern: symdpa_{} (hinge scaffolds) or lg_{} (sweep_lif_log)
EX = [('σ₁', TIES['pair'], TA.format('pair'), 0), ('σ₂', TIES['inv'], TA.format('inv'), 0), ('σ₃', TIES['test'], TA.format('test'), 1), ('V', TIES['klein'], TA.format('klein'), 2)]
EXW, CM = {}, {}   # per element: the memory pair of the network drawn under the scheme (each scheme is the prediction evaluated there)
CONSTRUCT = bool(os.environ.get('CONSTRUCT'))   # the flows are CONSTRUCTED networks: the whole-group-tied net with ONE free parameter of each element
if CONSTRUCT:                                    # switched on (training at the DPA stage lands every tie on the same symmetric solution)
    import copy
    base_, cfgB = load_run(TIES['klein'], 's2_' + TA.format('klein'), stage='dpa', device='cpu')
    CPERT = {'σ₁': ('⟨n₁⟩ shifted', lambda m, b: m.n[:, 1].add_(-0.035)),             # σ₁-even, σ₃-odd: the pair moves off the line together
             'σ₂': ('modes mixed', lambda m, b: (m.m[:, 0].add_(0.01 * b.m[:, 1]), m.n[:, 1].add_(0.01 * b.n[:, 0]))),   # σ₂-odd columns mixed: rotation
             'σ₃': ('⟨n₀⟩ shifted', lambda m, b: m.n[:, 0].add_(0.05)),               # σ₃-even, σ₁-odd: a ≠ a′ on the line
             'V': ('none switched on', lambda m, b: None)}
    for nm_ in ('σ₁', 'σ₂', 'σ₃', 'V'):
        m_ = copy.deepcopy(base_)
        with torch.no_grad(): CPERT[nm_][1](m_, base_)
        CM[nm_] = m_; EXW[nm_] = memory_pair(wells_of(m_, cfgB, sig_of(cfgB)), m_, cfgB)
else:
    for nm_, sw_, arm_, sd_ in EX:
        m_, cf_ = load_run(sw_, f's{sd_}_{arm_}', stage='dpa', device='cpu'); EXW[nm_] = memory_pair(wells_of(m_, cf_, sig_of(cf_)), m_, cf_)
INVCOL = {'σ₁': B_COL, 'σ₂': PLUM, 'σ₃': LICK_COL, 'V': VCOL}
def draw_invariant(ax, nm, col, lw=1.3, alpha=0.9):
    """Fix(D): the set every element of the tie leaves in place — the flow is tangent to it (σ₁: the κ₁ axis; σ₃: the κ₀ axis; σ₂: the origin)"""
    if nm in ('σ₁', 'V'): ax.axvline(0, color=col, lw=lw, alpha=alpha, zorder=2)
    if nm in ('σ₃', 'V'): ax.axhline(0, color=col, lw=lw * 1.8, alpha=alpha * 0.45, zorder=2)
    if nm == 'σ₂': ax.plot(0, 0, 'o', mfc='none', mec=col, mew=1.3, ms=7, zorder=6)
sg = gs[1, 6:20].subgridspec(1, 4, wspace=0.45); axs_e = [fig.add_subplot(sg[0, k]) for k in range(4)]
SCH_TIT = {'σ₁': 'σ₁  A↔B and C↔D', 'σ₂': 'σ₂  A↔B', 'σ₃': 'σ₃  C↔D', 'V': 'V  all three'}
SCH_RULE = {'σ₁': 'B mirrors A;\none height (free)', 'σ₂': 'B = −A;\nthe angle is free', 'σ₃': 'A and B on the line;\nunrelated distances', 'V': 'B mirrors A,\non the line'}
WELLC = '#F39C12'   # the flow panels' attractor ring colour
for ax, (nm, rel, Ds, col, note) in zip(axs_e, ELEMS):
    # the prediction, drawn as what to look for in the flow below: the two wells (at that network's positions), the map from A
    # to B, the invariant set, and a one-line rule
    A_, B_ = [tuple(float(v) for v in f[:2]) if f is not None else d for f, d in zip(EXW[nm], ((0.95, 0.0), (-0.95, 0.0)))]
    ax.set_xlim(*XL); ax.set_ylim(*XL); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.0); sp.set_visible(True)
    ax.axhline(0, color=LICK_COL, lw=0.8, ls='--', zorder=1); ax.axvline(0, color='0.9', lw=0.6, zorder=0)
    if nm in ('σ₁', 'V'): ax.axvspan(-0.045, 0.045, color=col, alpha=0.35, lw=0, zorder=1)
    if nm in ('σ₃', 'V'): ax.axhspan(-0.045, 0.045, color=col, alpha=0.35, lw=0, zorder=1)
    if nm == 'σ₂': ax.plot(0, 0, 'o', ms=7, mfc=col, mec='none', alpha=0.6, zorder=2)
    if nm != 'σ₃':
        ax.add_patch(FancyArrowPatch(A_, B_, arrowstyle='<->', mutation_scale=9, color=col, lw=1.2, shrinkA=9, shrinkB=9,
                                     connectionstyle=f"arc3,rad={0.0 if nm == 'σ₂' else -0.35}", zorder=3))
    else:   # σ₃ maps each well onto itself: nothing ties the two distances from the origin
        for p_, lab in ((A_, 'a'), (B_, 'a′')):
            ax.add_patch(FancyArrowPatch((0, -0.42), (p_[0], -0.42), arrowstyle='<->', mutation_scale=7, color=col, lw=0.9, shrinkA=0, shrinkB=0, zorder=3))
            ax.text(p_[0] / 2, -0.52, lab, ha='center', va='top', fontsize=SMALL, color=col)
    for p_, lab, c_ in ((A_, 'A', A_COL), (B_, 'B', B_COL)):
        ax.plot(*p_, 'o', ms=9, mfc='none', mec=WELLC, mew=1.8, zorder=5); ax.plot(*p_, 'o', ms=4.5, mfc=c_, mec='none', zorder=6)
        ax.text(p_[0], p_[1] + (0.32 if p_[1] >= 0 else -0.32), lab, ha='center', va='center', fontsize=SMALL, color=c_, fontweight='bold', zorder=7)
    ax.set_title(SCH_TIT[nm], loc='left', fontsize=TITLE_FS, color=col)
    ax.text(0.5, -0.05, SCH_RULE[nm], transform=ax.transAxes, ha='center', va='top', fontsize=SMALL, color=col)
axs_e[0].text(1.42, 0.08, 'lick', fontsize=SMALL * 0.85, color=LICK_COL, ha='right', va='bottom')
axs_e[0].set_ylabel('$\\kappa_1$ choice')

# ── c, lower row: the simulated flow of a network tied to each element (DPA checkpoint), under its scheme ──
sg2 = gs[2, 6:20].subgridspec(1, 4, wspace=0.45); axs_cf = [fig.add_subplot(sg2[0, k]) for k in range(4)]
for ax, (nm, sw, arm, sd), (_, _, _, col, _) in zip(axs_cf, EX, ELEMS):
    m0, cfg = (CM[nm], cfgB) if CONSTRUCT else load_run(sw, f's{sd}_{arm}', stage='dpa', device='cpu'); sig = sig_of(cfg)
    cache, spd = _flow_panel_cache(m0, dict(name=nm, dims=None, conds=[]), cfg['input_size'], torch.zeros(1, 1, cfg['input_size']), XL, XL, 61, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
    _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m0)
    ax.axhline(0, color='w', lw=0.8, ls=(0, (4, 3))); draw_invariant(ax, nm, INVCOL[nm] if nm != 'V' else '0.85', lw=1.1)
    for s_ in ([] if CONSTRUCT else range(4)):   # every seed's memory pair under this tie (trained mode only)
        try: m, cf = load_run(sw, f's{s_}_{arm}', stage='dpa', device='cpu')
        except Exception: continue
        A, B = memory_pair(wells_of(m, cf, sig_of(cf)), m, cf)
        for f, c_ in ((A, A_COL), (B, B_COL)):
            if f is not None: ax.plot(f[0], f[1], 'o', ms=3.2, mfc=c_, mec='w', mew=0.4, zorder=7)
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(0.9)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel('$\\kappa_0$ sample'); ax.set_ylabel('$\\kappa_1$ choice' if nm == 'σ₁' else ''); ax.set_title(((f'{nm} only' if nm != 'V' else 'all of V') if CONSTRUCT else f'{nm} tied'), loc='left', fontsize=TITLE_FS, color=col)
    print('tied flow', nm, 'done', flush=True)

# ── d: what each stage does to the group — the explanation of the push. Left, the account; middle, the choice readout learns to hear
#      Go/NoGo in the GNG stage (forbidden under σ₂/σ₃: the deafness lemma); right, the equivariance residual of the autonomous field per element ──
ax = cell(3, 0, 10); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax_f0 = ax
ax.set_title('the group across the curriculum', loc='left', fontsize=TITLE_FS)
STG = [('DPA', 'V = {e, σ₁, σ₂, σ₃}', 'all four hold;\nσ₃ and V pin the\nwells on the\nlick line', ['e', 'σ₁', 'σ₂', 'σ₃'], []),
       ('GNG', 'Z₂ = {e, σ₁}', 'one-sided objective;\nσ₂, σ₃ flip the\nresponse: broken', ['e', 'σ₁'], ['σ₂', 'σ₃']),
       ('Dual', 'Z₂ = {e, σ₁}', 'no-lick cost = a\ndownward field;\nσ₁ ⇒ both wells\nmove as one', ['e', 'σ₁'], ['σ₂', 'σ₃'])]
ECOL = {'e': '0.3', 'σ₁': B_COL, 'σ₂': PLUM, 'σ₃': LICK_COL}
for k_, (stg, grp, txt, kept, broken) in enumerate(STG):
    x0 = 0.2 + k_ * 4.0
    ax.text(x0, 9.8, stg, fontsize=PS*8, fontweight='bold', va='top'); ax.text(x0, 8.9, grp, fontsize=STAT_FS, va='top', color='0.3')
    for e_i, el in enumerate(['e', 'σ₁', 'σ₂', 'σ₃']):
        xx = x0 + 0.25 + e_i * 0.8; ax.text(xx, 7.6, el, fontsize=PS*7.5, color=ECOL[el], ha='center', va='center', alpha=(1 if el in kept else 0.35))
        if el in broken: ax.plot([xx - 0.25, xx + 0.25], [7.3, 7.9], color='k', lw=0.9)
    # mini plane
    px, py, ps = x0 + 1.7, 5.0, 1.4
    ax.plot([px - ps, px + ps], [py, py], color=LICK_COL, lw=0.7, ls='--'); ax.plot([px, px], [py - ps, py + ps], color='0.85', lw=0.6)
    ya = {0: 0.0, 1: 0.0, 2: -0.9}[k_]
    ax.plot(px + 0.95, py + ya, 'o', color=A_COL, ms=4, mec='w', mew=0.4, zorder=5); ax.plot(px - 0.95, py + ya, 'o', color=B_COL, ms=4, mec='w', mew=0.4, zorder=5)
    if k_ == 2:
        for sx in (0.95, -0.95): ax.annotate('', (px + sx, py - 0.75), (px + sx, py + 0.6), arrowprops=dict(arrowstyle='->', color='0.3', lw=0.9))
    if k_ == 0:
        ax.annotate('', (px + 0.95, py - 0.55), (px + 0.95, py + 0.55), arrowprops=dict(arrowstyle='<->', color=LICK_COL, lw=0.8, linestyle=':')); ax.text(px + 1.15, py + 0.6, 'σ₃', fontsize=SMALL, color=LICK_COL)
    if k_ == 1:
        ax.annotate('', (px + 1.4, py + 1.0), (px + 1.4, py + 0.1), arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.9)); ax.annotate('', (px - 1.4, py - 1.0), (px - 1.4, py - 0.1), arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.9))
        ax.text(px + 1.55, py + 0.5, 'Go', fontsize=SMALL, color=BLUE, va='center'); ax.text(px - 1.55, py - 0.5, 'NoGo', fontsize=SMALL, color=GREEN, va='center', ha='right')
    ax.text(x0, 2.9, txt, fontsize=STAT_FS, va='top', color='0.3')
# f2: overlaps of the choice readout with the Go and NoGo columns
OV = json.load(open(os.environ.get('OVJ', '/home/leon/dual/figures/paper_share/modelling/overlaps_bfix.json')))
ax = cell(3, 10, 15); ax_f1 = ax; ax.set_box_aspect(1); ax.set_anchor('W')
for s_ in fseeds:
    for c_, col, mk in ((4, BLUE, 's'), (5, GREEN, 'o')):
        ax.plot(range(3), [OV[f'{s_}|{st}'][str(c_)][1] for st in ('dpa', 'naive', 'expert')], '-', marker=mk, color=col, lw=0.7, ms=2.8, alpha=0.55)
        ax.plot(range(3), [OV[f'{s_}|{st}'][str(c_)][0] for st in ('dpa', 'naive', 'expert')], '-', marker=mk, color=col, lw=0.6, ms=2.2, alpha=0.25, mfc='white')
for c_, col in ((4, BLUE), (5, GREEN)): ax.plot(range(3), [np.median([OV[f'{s_}|{st}'][str(c_)][1] for s_ in fseeds]) for st in ('dpa', 'naive', 'expert')], '-', color=col, lw=2.0, zorder=5)
ax.axhline(0, color='0.75', lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels(['after\nDPA', 'after\nGNG', 'after\nDual']); ax.set_ylabel('overlap $n_j^{\\mathrm{T}} w_c / N$')
ax.set_title('Go/NoGo inputs reach\nthe choice readout', loc='left', fontsize=TITLE_FS)
ax.set_ylim(-4.2, 4.6); ax.set_xlim(-0.25, 2.25)
from matplotlib.lines import Line2D as _L2
ax.legend(handles=[_L2([0], [0], color='0.35', marker='s', ms=3.0, lw=0.9, label='$n_1$, choice'),
                   _L2([0], [0], color='0.35', marker='s', ms=3.0, lw=0.9, mfc='white', alpha=0.6, label='$n_0$, sample'),
                   _L2([0], [0], color=BLUE, lw=2.0, label='Go'), _L2([0], [0], color=GREEN, lw=2.0, label='NoGo')],
          loc='upper left', ncol=2, frameon=False, fontsize=SMALL * 0.92, handlelength=1.3, columnspacing=0.7, handletextpad=0.4, borderaxespad=0.1)
# f3: the equivariance residual of the autonomous field per element
LED = {}
for l in open(os.environ.get('SVT', '/home/leon/dual/figures/paper_share/modelling/symviol_bfix.tsv')).read().strip().split('\n')[1:]:
    r = l.split('\t')
    if len(r) < 5 or not r[2][0].isdigit(): continue
    LED[(int(r[0].split('_')[0][1:]), r[1])] = [float(r[2]), float(r[3]), float(r[4])]
ax = cell(4, 10, 15); ax_f2 = ax; ax.set_box_aspect(1); ax.set_anchor('W')
_lab_y = {}
for e_i, (el, col) in enumerate((('σ₁', B_COL), ('σ₂', PLUM), ('σ₃', LICK_COL))):
    ys = np.array([[LED[(s_, st)][e_i] for st in ('dpa', 'naive', 'expert')] for s_ in fseeds if all((s_, st) in LED for st in ('dpa', 'naive', 'expert'))])
    for y in ys: ax.plot(range(3), y, '-o', color=col, lw=0.6, ms=2.4, alpha=0.4)
    _md = np.median(ys, 0); ax.plot(range(3), _md, '-', color=col, lw=2.0, zorder=5); _lab_y[el] = (_md[-1], col)
_ord = sorted(_lab_y.items(), key=lambda kv: kv[1][0]); _yy = [v[0] for _, v in _ord]
for _i in range(1, len(_yy)): _yy[_i] = max(_yy[_i], _yy[_i - 1] + 0.085)       # keep the end labels apart
for (el, (_, col)), _y in zip(_ord, _yy): ax.text(2.1, _y, el, color=col, fontsize=PS*7.5, fontweight='bold', va='center', ha='left')
ax.set_xticks(range(3)); ax.set_xticklabels(['after\nDPA', 'after\nGNG', 'after\nDual']); ax.set_ylabel('equivariance residual'); ax.set_ylim(0, 1.15); ax.set_xlim(-0.25, 2.45)
ax.set_title('the autonomous field\nkeeps σ₁ only', loc='left', fontsize=TITLE_FS)

# ── wells of the free networks at the three checkpoints (cached), for the push ──
wells_free = {}
for stage in ('dpa', 'naive', 'expert'):
    for s in fseeds:
        m, cf = load_run(fsw, f's{s}_{farm}', stage=stage, device='cpu'); sig = sig_of(cf); A, B = memory_pair(wells_of(m, cf, sig), m, cf); wells_free[(stage, s)] = (A, B, sig)
stages = ['after DPA', 'after GNG', 'after Dual']

# ── d, lower row: the free network's autonomous flow at the three checkpoints, under the DPA / GNG / Dual columns of the account ──
sgd = gs[4, 0:9].subgridspec(1, 3, wspace=0.35); axs_d = [fig.add_subplot(sgd[0, k]) for k in range(3)]
for k, (stage, lab) in enumerate([('dpa', 'after DPA'), ('naive', 'after GNG'), ('expert', 'after Dual')]):
    ax = axs_d[k]; m0, cfg = load_run(fsw, f's{fseeds[0]}_{farm}', stage=stage, device='cpu'); sig = sig_of(cfg)
    cache, spd = _flow_panel_cache(m0, dict(name=lab, dims=None, conds=[]), cfg['input_size'], torch.zeros(1, 1, cfg['input_size']), XL, XL, 61, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
    _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m0)
    ax.axhline(0, color='w', lw=0.8, ls=(0, (4, 3)))
    for s_ in fseeds:
        A, B, _ = wells_free[(stage, s_)]
        for f, col in ((A, A_COL), (B, B_COL)):
            if f is not None: ax.plot(f[0], f[1], 'o', ms=3.2, mfc=col, mec='w', mew=0.4, zorder=7)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel('$\\kappa_0$ sample'); ax.set_ylabel('$\\kappa_1$ choice' if k == 0 else ''); ax.set_title(lab, loc='left', fontsize=TITLE_FS)
    print('flow', stage, 'done', flush=True)

# ── e: the push, per seed (tab10 = per network, as the paper's per-mouse colours), mean ± 95% CI, Wilcoxon DPA → Dual ──
ax = cell(3, 15, 20); ax_d = ax; ax.set_box_aspect(1); ax.set_anchor('W')
H = {idx: np.array([[wells_free[(st, s_)][idx][1] / wells_free[(st, s_)][2] if wells_free[(st, s_)][idx] is not None else np.nan for st in ('dpa', 'naive', 'expert')] for s_ in fseeds]) for idx in (0, 1)}
NETCOL = SEED_COL if len(fseeds) <= len(SEED_COL) else __import__('seaborn').color_palette('tab20', len(fseeds))   # 16 networks: tab20
for i, s_ in enumerate(fseeds):
    ax.plot(np.arange(3) - 0.07, H[0][i], '-o', color=NETCOL[i], alpha=0.6, lw=0.7, ms=2.8); ax.plot(np.arange(3) + 0.07, H[1][i], '-o', color=NETCOL[i], alpha=0.6, lw=0.7, ms=2.8, mfc='white')
rng_ = np.random.default_rng(0)
for idx, dx in ((0, -0.07), (1, 0.07)):
    mu = np.nanmean(H[idx], 0); boots = np.array([np.nanmean(H[idx][rng_.integers(0, len(fseeds), len(fseeds))], 0) for _ in range(2000)]); lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    ax.errorbar(np.arange(3) + dx, mu, yerr=[mu - lo, hi - mu], fmt='-', color='k', lw=1.5, capsize=2, elinewidth=1.0, zorder=5, marker=('o' if idx == 0 else 'o'), mfc=('k' if idx == 0 else 'white'), ms=4)
def _wil(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); return stats.wilcoxon(a[ok], b[ok]).pvalue, int(ok.sum())
pvals = {}; nwil = {}
for idx in (0, 1): pvals[idx], nwil[idx] = _wil(H[idx][:, 2], H[idx][:, 0])
print('e: heights (η) per network, A | B, dpa/naive/expert:'); [print(f'   s{s_}: ' + ' | '.join(' '.join('—' if np.isnan(v) else f'{v:+.2f}' for v in H[idx][i]) for idx in (0, 1))) for i, s_ in enumerate(fseeds)]
print(f'e: Wilcoxon DPA→Dual A p = {pvals[0]:.4f} (n = {nwil[0]}), B p = {pvals[1]:.4f} (n = {nwil[1]}); expert heights A {np.nanmin(H[0][:, 2]):+.2f}…{np.nanmax(H[0][:, 2]):+.2f}, B {np.nanmin(H[1][:, 2]):+.2f}…{np.nanmax(H[1][:, 2]):+.2f}; mean A {np.nanmean(H[0][:, 2]):+.2f}, B {np.nanmean(H[1][:, 2]):+.2f}')
_both = all(pvals[i] < 0.05 for i in (0, 1))
for idx, xx in (((0, 2.0),) if _both else ((0, 1.85), (1, 2.15))):
    sig = pvals[idx] < 0.05; ax.text(xx, 0.55, '∗' if sig else 'n.s.', ha='center', fontsize=(PS*11 if sig else PS*6.5), fontweight='bold', color=('k' if sig else '0.55'))
_pf = lambda p_: f'{p_:.4f}' if p_ < 0.001 else f'{p_:.3f}'
ax.text(0.03, 0.03, (f'Wilcoxon DPA → Dual,\nA and B: p = {_pf(pvals[0])}' if abs(pvals[0] - pvals[1]) < 1e-9 else f'Wilcoxon DPA → Dual\nA p = {_pf(pvals[0])}, B p = {_pf(pvals[1])}'), transform=ax.transAxes, fontsize=STAT_FS, color='0.3')
ax.axhline(0, color=LICK_COL, lw=0.8, ls='--'); ax.set_xticks(range(3)); ax.set_xticklabels(['after\nDPA', 'after\nGNG', 'after\nDual']); ax.set_ylabel('well height, $\\kappa_1/\\eta$'); ax.set_ylim(-2.3, 0.9); ax.set_xlim(-0.35, 2.35)
nb = sum(1 for s_ in fseeds if wells_free[('expert', s_)][0] is not None and wells_free[('expert', s_)][1] is not None and wells_free[('expert', s_)][0][1] / wells_free[('expert', s_)][2] < BELOW and wells_free[('expert', s_)][1][1] / wells_free[('expert', s_)][2] < BELOW)
ax.set_title(f'the push ({nb}/{len(fseeds)} below)', loc='left', fontsize=TITLE_FS); print(f'e: both wells below the line in {nb}/{len(fseeds)}; networks with a memory pair after DPA: {sum(1 for s_ in fseeds if all(f is not None for f in wells_free[("dpa", s_)][:2]))}, after Dual: {sum(1 for s_ in fseeds if all(f is not None for f in wells_free[("expert", s_)][:2]))}'); ax.text(0.02, 0.95, 'A filled, B open', transform=ax.transAxes, ha='left', va='top', fontsize=SMALL, color='0.3')

# ── e, bottom: where the push comes from (2026-10-01, Leon: "a big element missing from the figure which is where the push
#    down comes from"). The forces the objective exerts on a memory well at height h = κ₁/η, each the derivative of its
#    −log-likelihood term (Methods, eq. 6): the paired test, where a lick is required at h + k, pulls the well up,
#    φ(h+k)/Φ(h+k); the unpaired test, where a lick is wrong at h − k, pushes it down, φ(h−k)/Φ(k−h); and, in the dual stage
#    only, the delay, where a lick is wrong at the well itself, pushes it down, φ(h)/Φ(−h), the Gaussian hazard. k is the mean
#    test-evoked displacement of the networks that learned (draft_numbers_log16.json, depth law); equal weights, so the
#    balance drawn is the leading term of the depth law, d* = k/2.
from scipy.stats import norm as _NRM
LEARN_ = [s_ for s_ in fseeds if (acc_of(fsw, f's{s_}_{farm}') or {}).get('after_dpa', {}).get('dpa', 0) >= 0.95]
_dnj = os.environ.get('DNJ', '/home/leon/dual/figures/paper_share/modelling/draft_numbers_log16.json')
_DL = json.load(open(_dnj)).get('depth_law', {}) if os.path.exists(_dnj) else {}
_ks = [v[1] for r_, v in _DL.items() if int(r_.split('_')[0][1:]) in LEARN_]
KTEST = float(np.mean(_ks)) if _ks else 2.5
ax = cell(4, 15, 20); ax_push = ax; ax.set_box_aspect(1); ax.set_anchor('W')
hh = np.linspace(-3.4, 1.4, 800)
pull = _NRM.pdf(hh + KTEST) / _NRM.cdf(hh + KTEST); push_u = _NRM.pdf(hh - KTEST) / _NRM.cdf(KTEST - hh); push_d = _NRM.pdf(hh) / _NRM.cdf(-hh)
net_dpa, net_dual = pull - push_u, pull - push_u - push_d          # net force on the well, up positive
ax.axhline(0, color='0.6', lw=0.7, zorder=1); ax.axvline(0, color=LICK_COL, lw=0.8, ls='--', zorder=1)
ax.fill_between(hh, net_dual, net_dpa, color=LICK_COL, alpha=0.13, lw=0, zorder=2)
ax.plot(hh, net_dpa, color='0.45', lw=1.5, zorder=4); ax.plot(hh, net_dual, color=LICK_COL, lw=1.7, zorder=4)
def _zero(y_):
    k_ = np.where(np.sign(y_[:-1]) != np.sign(y_[1:]))[0][-1]
    return float(hh[k_] - y_[k_] * (hh[k_ + 1] - hh[k_]) / (y_[k_ + 1] - y_[k_]))
H_DPA, H_DUAL = _zero(net_dpa), _zero(net_dual)
ax.plot(H_DPA, 0, 'o', ms=6.5, mfc='white', mec='0.35', mew=1.3, zorder=6); ax.plot(H_DUAL, 0, 'o', ms=6.5, mfc=LICK_COL, mec='white', mew=0.8, zorder=6)
ax.annotate('', (H_DUAL + 0.12, 0.16), (H_DPA - 0.05, 0.16), arrowprops=dict(arrowstyle='->', color=LICK_COL, lw=1.4, shrinkA=0, shrinkB=0))
ax.text((H_DPA + H_DUAL) / 2, 0.21, 'push', color=LICK_COL, ha='center', va='bottom', fontsize=STAT_FS)
YT = 1.12
for idx in (0, 1):                                         # the trained wells of the networks that learned: after DPA (gray), after Dual (red)
    for i, s_ in enumerate(fseeds):
        if s_ not in LEARN_: continue
        for c_, colr in ((0, '0.5'), (2, LICK_COL)):
            v = H[idx][i, c_]
            if not np.isnan(v): ax.plot([v, v], [YT - 0.09, YT], color=colr, lw=0.9, alpha=0.8, zorder=3)
ax.text(-1.95, 0.72, 'up: paired\ntest', color='0.25', fontsize=STAT_FS, va='center', ha='left')
ax.text(-3.3, -0.66, 'down: no lick,\nunpaired test', color='0.45', fontsize=STAT_FS, va='center', ha='left')
ax.text(-3.3, -1.1, '+ delay (dual)', color=LICK_COL, fontsize=STAT_FS, va='center', ha='left')
ax.text(H_DPA + 0.12, -0.08, 'DPA', color='0.4', fontsize=STAT_FS, va='top', ha='left')
ax.text(H_DUAL - 0.1, -0.08, 'dual', color=LICK_COL, fontsize=STAT_FS, va='top', ha='right')
ax.set_xlim(-3.4, 1.4); ax.set_ylim(-1.3, 1.2); ax.set_xlabel('well height, $\\kappa_1/\\eta$'); ax.set_ylabel('net force on the well')
ax.set_title('where the push comes from', loc='left', fontsize=TITLE_FS)
_hw = np.array([H[idx][i, 2] for idx in (0, 1) for i, s_ in enumerate(fseeds) if s_ in LEARN_]); _hw = _hw[~np.isnan(_hw)]
print(f'e (bottom): k = {KTEST:.2f} η ({len(_ks)} networks); zero net force: DPA objective {H_DPA:+.2f}, dual objective {H_DUAL:+.2f} (−k/2 = {-KTEST / 2:+.2f}); trained wells after Dual {_hw.min():+.2f}…{_hw.max():+.2f} (mean {_hw.mean():+.2f})')

# ── f, g, h: depth ↔ performance by perturbation — a delay-only drive along m1 moves the wells on the lick axis (the test-driven field is untouched);
#      DPA and GNG accuracy against the well location, two wells (A filled, B open) per drive strength, per network (tab10) ──
PJ = json.load(open(os.environ.get('PJF', '/home/leon/dual/figures/paper_share/modelling/perturb_depth.json')))
P = {}
for k, v in PJ.items():
    s_, var, d = k.split('|'); P.setdefault((int(s_), var), []).append((float(d), v))
ax_g = cell(5, 0, 6); ax_h = cell(5, 7, 13); ax_i = cell(5, 14, 20)
# PSEEDS: the networks drawn in f–h (default all of FREE; '1,2,3,4,5,6' = the six that learned all three stages of sweep_lif_log,
# 2026-09-30: s0 and s7 never learned DPA and fail Go ~0.5 η early, which flattened the Go/NoGo plateau). BINW: bin width in η.
# 2026-09-30 (Leon: "Figure 5 needs some work to be as good as the rest"): the default is now the networks that learned the DPA
# stage (after_dpa DPA accuracy >= 0.95 in results.jsonl, as compare_log16.py), in 0.25 eta bins, as Fig. 6h; PSEEDS=0,...,7 BINW=0.5
# restores the earlier pooled build.
PSEEDS = ([int(x) for x in os.environ['PSEEDS'].split(',')] if os.environ.get('PSEEDS')
          else [s_ for s_ in fseeds if (acc_of(fsw, f's{s_}_{farm}') or {}).get('after_dpa', {}).get('dpa', 0) >= 0.95 and (s_, 'drive') in P])
BINW = float(os.environ.get('BINW', '0.25')); print('f–h networks:', PSEEDS, 'bin', BINW)
def binned(ax, xk, yk, col):
    """all (network, sample, drive) points, gray; binned mean ± 95% bootstrap CI in fixed 0.5 η bins, coloured"""
    X = np.array([r[xk + lab] for s_ in PSEEDS for d, r in P[(s_, 'drive')] for lab in ('A', 'B')]); Y = np.array([r[yk + lab] for s_ in PSEEDS for d, r in P[(s_, 'drive')] for lab in ('A', 'B')])
    XA = np.array([r[xk + 'A'] for s_ in PSEEDS for d, r in P[(s_, 'drive')]]); YA = np.array([r[yk + 'A'] for s_ in PSEEDS for d, r in P[(s_, 'drive')]])
    XB = np.array([r[xk + 'B'] for s_ in PSEEDS for d, r in P[(s_, 'drive')]]); YB = np.array([r[yk + 'B'] for s_ in PSEEDS for d, r in P[(s_, 'drive')]])
    ax.scatter(XA, YA, s=4, color='0.55', alpha=0.45, edgecolors='none', zorder=2); ax.scatter(XB, YB, s=6, facecolors='none', edgecolors='0.55', linewidths=0.4, alpha=0.5, zorder=2)
    edges = np.arange(np.floor(X.min() / BINW) * BINW, np.ceil(X.max() / BINW) * BINW + 0.01, BINW); xc, mu, lo, hi = [], [], [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        sel = (X >= a_) & (X < b_)
        if sel.sum() < 4: continue
        boots = np.array([np.mean(Y[sel][rng_.integers(0, sel.sum(), sel.sum())]) for _ in range(1000)]); xc.append((a_ + b_) / 2); mu.append(Y[sel].mean()); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
    mu = np.array(mu); ax.errorbar(xc, mu, yerr=[mu - np.array(lo), np.array(hi) - mu], fmt='-o', color=col, lw=1.4, ms=3.0, capsize=1.5, elinewidth=0.8, zorder=6)
    ax._binned = (np.array(xc), mu); print(f'binned {yk}:', ' '.join(f'{a:+.2f}:{b:.2f}' for a, b in zip(xc, mu)))
    return stats.spearmanr(X, Y)
for s_ in PSEEDS:   # dual performance = P(both responses of a dual trial correct) = DPA accuracy × GNG accuracy, per (sample, drive)
    for d, r in P[(s_, 'drive')]:
        for lab in ('A', 'B'): r['dual_perf_' + lab] = r['dpa_acc_' + lab] * r['gng_acc_' + lab]
rho_d, p_d = binned(ax_g, 'depth_', 'dpa_acc_', RED); rho_g, p_g = binned(ax_h, 'depthU_', 'gng_acc_', BLUE); rho_i, p_i = binned(ax_i, 'depthU_', 'dual_perf_', '0.15')
for a_, ttl, yl in ((ax_g, 'DPA trials, well at the test', 'DPA accuracy'), (ax_h, 'Go/NoGo, well at the cue', 'Go/NoGo accuracy'), (ax_i, 'dual trials, both correct', 'P(both correct)')):
    a_.axvline(0, color=LICK_COL, lw=0.8, ls='--'); a_.set_ylim(0.35, 1.03); a_.set_xlabel('well on the choice axis, $\\kappa_1/\\eta$'); a_.set_ylabel(yl); a_.set_title(ttl, loc='left', fontsize=TITLE_FS)
TRAINED_CUE = float(np.mean([r['depthU_' + l] for s_ in PSEEDS for d, r in P[(s_, 'drive')] if abs(d) < 1e-9 for l in 'AB']))   # the trained well at the cue (drive 0)
ax_i.axvline(TRAINED_CUE, color='0.5', lw=0.8, ls=':'); print(f'trained state at the cue: {TRAINED_CUE:+.2f} eta')
xc_, mu_ = ax_i._binned; ax_i.plot(-xc_, mu_, ':', color=LICK_COL, lw=1.2, zorder=5)   # the image of the curve under σ₃ (κ₁ → −κ₁): equal to the curve only if σ₃ held
for a_, lft, rgt in ((ax_g, '← misses', 'false alarms →'), (ax_h, '← Go misses', 'NoGo licks →'), (ax_i, '', '')):
    a_.text(0.03, 0.985, lft, transform=a_.transAxes, ha='left', va='top', fontsize=STAT_FS, color='0.35')
    a_.text(0.97, 0.985, rgt, transform=a_.transAxes, ha='right', va='top', fontsize=STAT_FS, color='0.35')
    a_.set_ylim(0.35, 1.08)
ax_g.legend(handles=[_L2([0], [0], marker='o', ls='none', color='0.55', ms=3.5, label='A'), _L2([0], [0], marker='o', ls='none', mfc='white', mec='0.55', ms=3.5, label='B')],
            title=f'{len(PSEEDS)} networks × 81 drives; bins {BINW:g} η', title_fontsize=STAT_FS, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=SMALL, handletextpad=0.2, columnspacing=0.8)
ax_i.legend(handles=[_L2([0], [0], color='0.5', lw=0.9, ls=':', label='trained state at the cue'), _L2([0], [0], color=LICK_COL, lw=1.2, ls=':', label='σ₃ image of the curve')],
            loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, fontsize=SMALL, handlelength=1.8)

fig.canvas.draw()                                     # apply the square box aspects before placing the letters
for L, a in zip('abcdefgh', [fig.axes[0], ax_b, ax_e0, ax_f0, ax_d, ax_g, ax_h, ax_i]): panel_letter(fig, a, L)
import re as _re, matplotlib.text as _mt
for _t in fig.findobj(_mt.Text):                       # typographic minus for numbers written as text (tick labels already use it)
    _x = _t.get_text()
    if _x and '$' not in _x: _t.set_text(_re.sub(r'(^|[\s=(\[,:])-(?=\d)', '\\1\u2212', _x))
save(fig, os.environ.get('FIG5_STEM', 'fig5_model'))
if os.environ.get('FIG5_STEM'): sys.exit(0)   # a review render of Fig. 5 only; ED 18 below is left untouched

# ── Extended Data 18: what left Fig. 5 — a lick probabilities per trial type, b the free network's flow at the three checkpoints, c the six populations ──
fig = plt.figure(figsize=(9.6, 6.4)); gs = fig.add_gridspec(2, 20, height_ratios=[1.0, 1.0], hspace=0.6, wspace=2.4, left=0.06, right=0.985, top=0.95, bottom=0.07)
def cell(r, c0, c1): return fig.add_subplot(gs[r, c0:c1])

# ── b: lick probabilities per trial type at each checkpoint (simulated, trained noise), as Fig. 1g ──
from src.tasks import generate_dual_trials
def lick_probs(sw, rid, stage):
    m, cfg = load_run(sw, rid, stage=stage, device='cpu'); dt, alpha, _ = run_dt_alpha(cfg); eta = cfg['noise'] * float(np.sqrt(1 - np.exp(-alpha) ** 2)); torch.manual_seed(1)
    Td = make_timings(dt)['dual']; half = int(round(0.5 / dt))
    X, y, _, names = generate_dual_trials(384, Td, cfg['input_size'], noise=eta, target_rank=2, cue_on_go_input=cfg['cue_on_go_input'], cue_scale=cfg['cue_scale'], nogo_target=cfg['nogo_target'], go_target=cfg['go_target'], response_in_cue=cfg['response_in_cue'], windowed_targets=cfg['windowed_targets'], decay_to_zero=cfg['decay_to_zero'], gng_response=cfg['gng_response'], post_response_window=cfg.get('dpa_post_response_window'))
    with torch.no_grad(): k1 = m(X)[..., 1].numpy()
    to, co = int(Td.n_stim_off[3]), int(Td.n_stim_off[2]); ric = cfg['response_in_cue']
    L = (lambda seg: (seg > 0).any(1)) if os.environ.get('LICK', 'any') == 'any' else (lambda seg: seg.mean(1) > 0)   # a lick = κ₁ > 0 at any step of the window
    test_lick = L(k1[:, (to - half):to] if ric else k1[:, to:to + half]); cue_lick = L(k1[:, (co - half):co] if ric else k1[:, co:co + half])
    samp = np.array([n[0] for n in names]); tst = np.array([n[-1] for n in names]); pair = ((samp == 'A') & (tst == 'C')) | ((samp == 'B') & (tst == 'D'))
    go = np.array(['_go_' in n for n in names]); ng = np.array(['_nogo_' in n for n in names])
    Tp = make_timings(dt)['dpa']; Xd, _ = generate_dpa_trials(256, Tp, cfg['input_size'], noise=eta, target_rank=2, windowed_targets=True, decay_to_zero=False, response_in_cue=ric, prelick_free=True, hold_window=0.5, hold_anchor='sample', post_response_window=cfg.get('dpa_post_response_window'))
    with torch.no_grad(): kd = m(Xd)[..., 1].numpy()
    tod = int(Tp.n_stim_off[1]); dl = (kd[:, (tod - half):tod].mean(1) if ric else kd[:, tod:tod + half].mean(1)) > 0
    sA = Xd[:, int(Tp.n_stim_on[0]):int(Tp.n_stim_off[0]), 0].mean(1).numpy() > 0.5; tC = Xd[:, int(Tp.n_stim_on[1]):int(Tp.n_stim_off[1]), 2].mean(1).numpy() > 0.5; pd_ = (sA & tC) | (~sA & ~tC)
    return {('DPA', 'paired'): dl[pd_].mean(), ('DPA', 'unpaired'): dl[~pd_].mean(), ('Go', 'paired'): test_lick[go & pair].mean(), ('Go', 'unpaired'): test_lick[go & ~pair].mean(),
            ('NoGo', 'paired'): test_lick[ng & pair].mean(), ('NoGo', 'unpaired'): test_lick[ng & ~pair].mean(), ('Go', 'cue'): cue_lick[go].mean(), ('NoGo', 'cue'): cue_lick[ng].mean()}
ax = cell(0, 0, 10); x = np.arange(3); ax_e18a = ax
LP = {s_: [lick_probs(fsw, f's{s_}_{farm}', st) for st in ('dpa', 'naive', 'expert')] for s_ in fseeds}
def series(key, first=0): return np.array([[LP[s_][i][key] if i >= first else np.nan for i in range(3)] for s_ in fseeds])
SPEC = [(('DPA', 'paired'), RED, '-', 'o', 'DPA paired'), (('DPA', 'unpaired'), RED, '-', 'o', 'DPA unpaired', 'white'), (('Go', 'paired'), BLUE, '-', 'o', 'Go paired'), (('Go', 'unpaired'), BLUE, '-', 'o', 'Go unpaired', 'white'),
        (('NoGo', 'paired'), GREEN, '-', 'o', 'NoGo paired'), (('NoGo', 'unpaired'), GREEN, '-', 'o', 'NoGo unpaired', 'white'), (('Go', 'cue'), BLUE, '--', 's', 'Go, at the cue'), (('NoGo', 'cue'), GREEN, '--', 's', 'NoGo, at the cue')]
for j, spec in enumerate(SPEC):
    key, col, ls, mk, lab = spec[:5]; mfc = spec[5] if len(spec) > 5 else col; v = series(key, first=(1 if key[1] == 'cue' else 0)); dx = (j - 3.5) * 0.028
    mu, se = np.nanmean(v, 0), np.nanstd(v, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(v), 0))
    ax.errorbar(x + dx, mu, yerr=se, fmt=ls, marker=mk, color=col, mfc=mfc, mec=col, lw=1.2, ms=3.5, capsize=1.5, elinewidth=0.7, label=lab)
ax.axhline(0.5, color='0.6', lw=0.6, ls=':'); ax.set_xticks(x); ax.set_xticklabels(stages); ax.set_ylim(-0.03, 1.03); ax.set_ylabel('P(lick)')
ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.16), fontsize=SMALL, handlelength=1.6, columnspacing=0.8)
ax.set_title(f'lick probability by trial type, {len(fseeds)} seeds (mean ± SEM)', loc='left', fontsize=TITLE_FS)
print('lick probabilities done', flush=True)

# ── c: the flow at the three checkpoints ──
axs_c = [cell(1, 0, 6), cell(1, 6, 12), cell(1, 12, 18)]
for k, (stage, lab) in enumerate([('dpa', 'after DPA'), ('naive', 'after GNG'), ('expert', 'after Dual')]):
    ax = axs_c[k]; m0, cfg = load_run(fsw, f's{fseeds[0]}_{farm}', stage=stage, device='cpu'); sig = sig_of(cfg)
    cache, spd = _flow_panel_cache(m0, dict(name=lab, dims=None, conds=[]), cfg['input_size'], torch.zeros(1, 1, cfg['input_size']), XL, XL, 61, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
    _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m0)
    ax.axhline(0, color='w', lw=0.8, ls=(0, (4, 3)))
    for s in fseeds:
        m, cf = load_run(fsw, f's{s}_{farm}', stage=stage, device='cpu'); A, B, _ = wells_free[(stage, s)]
        for f, col in ((A, A_COL), (B, B_COL)):
            if f is not None: ax.plot(f[0], f[1], 'o', ms=3.2, mfc=col, mec='w', mew=0.4, zorder=7)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel('$\\kappa_0$ sample'); ax.set_ylabel('$\\kappa_1$ choice' if k != 1 else ''); ax.set_title(lab, loc='left', fontsize=TITLE_FS)
    print('flow', stage, 'done', flush=True)

# ── ED18 c: populations of a whole-group network ──
ax = cell(0, 12, 19); m, cf = load_run(TIES['klein'], 's0_' + TA.format('klein'), stage='dpa', device='cpu'); M = m.m.detach().numpy(); N = len(M)
axu = (np.abs(M[:, 0]) < 0.8) & (np.abs(M[:, 1]) > 2.5); lat = ~axu
QC = {(1, 1): A_COL, (-1, 1): B_COL, (-1, -1): B_COL, (1, -1): A_COL}
for q, col in QC.items():
    sel = lat & (np.sign(M[:, 0]) == q[0]) & (np.sign(M[:, 1]) == q[1]); ax.scatter(M[sel, 0], M[sel, 1], s=6, color=col, alpha=(0.75 if q[1] > 0 else 0.35), edgecolors='none')
ax.scatter(M[axu, 0], M[axu, 1], s=7, color='0.45', alpha=0.8, edgecolors='none')
ax.axhline(0, color='0.85', lw=0.6); ax.axvline(0, color='0.85', lw=0.6); ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect('equal'); ax.set_xlabel('$m_0$ (sample write)'); ax.set_ylabel('$m_1$ (choice write)')
ax.text(0.0, -0.24, f'six populations:\nfour mixed (color, sample\nsign; opaque, choice +) and a\npure-choice pair ({axu.sum()} units, gray)', transform=ax.transAxes, va='top', fontsize=SMALL, color='0.3')
ax.set_title('units, whole-group tie', loc='left', fontsize=TITLE_FS); ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect('equal', adjustable='box')

ax_f = ax

for L, a in zip('abc', [ax_e18a, ax_f, axs_c[0]]): panel_letter(fig, a, L)
save(fig, 'ed18'); print('ED18 done')

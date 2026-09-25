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
fig = plt.figure(figsize=(9.6, 13.8)); gs = fig.add_gridspec(6, 20, height_ratios=[0.75, 0.6, 0.72, 0.6, 0.72, 0.9], hspace=0.42, wspace=2.4, left=0.06, right=0.985, top=0.975, bottom=0.035)
def cell(r, c0, c1): return fig.add_subplot(gs[r, c0:c1])
stages = ['after DPA', 'after GNG', 'after Dual']

# ── a: the model, drawn as in Mastrogiuseppe & Ostojic (2018), Fig. 1: inputs → population → the connectivity as a matrix image = the outer products → readout ──
ax = cell(0, 0, 10); ax.axis('off'); ax.set_xlim(0, 20); ax.set_ylim(0, 10); ax.set_title('rank-2 recurrent network', loc='left', fontsize=TITLE_FS)
mA, cfgA = load_run(fsw, f's{fseeds[0]}_{farm}', stage='dpa', device='cpu'); M = mA.m.detach().numpy(); Nv = mA.n.detach().numpy()
order = np.argsort(np.arctan2(M[:, 1], M[:, 0])); sub = order[::len(order) // 40][:40]; Wsub = (M[sub] @ Nv[sub].T) / len(M)
rng_ = np.random.default_rng(3); NODES = np.array([[3.6, 7.6], [5.2, 8.3], [6.6, 7.2], [3.2, 5.6], [4.9, 6.1], [6.4, 5.2], [3.8, 3.7], [5.4, 4.2], [6.9, 3.4], [4.6, 2.5]])
pairs = [(i, j) for i in range(len(NODES)) for j in range(len(NODES)) if i != j and np.linalg.norm(NODES[i] - NODES[j]) < 2.3]
for i, j in pairs:
    if rng_.random() < 0.55: ax.add_patch(FancyArrowPatch(NODES[i], NODES[j], arrowstyle='-|>', color='0.55', lw=0.55, mutation_scale=5, shrinkA=4.5, shrinkB=4.5, connectionstyle='arc3,rad=0.2', zorder=2))
for p_ in NODES: ax.add_patch(Circle(p_, 0.34, fc='#F6F5F0', ec='0.3', lw=0.8, zorder=3))
for k, (lab, c, tgt) in enumerate((('sample A/B', A_COL, (0, 3)), ('Go/NoGo, cue', BLUE, (3, 6)), ('test C/D', GREY, (6, 9)))):
    y = 8.4 - k * 2.4; ax.text(0.05, y + 0.55, lab, fontsize=SMALL, color=c)
    for t in tgt: ax.add_patch(FancyArrowPatch((1.1, y), NODES[t], arrowstyle='-|>', color=c, lw=0.7, mutation_scale=5, shrinkB=4.5, connectionstyle='arc3,rad=0.1', zorder=2))
ax.text(5.1, 1.35, '$N$ = 1024 units, $W_{\\mathrm{rec}}$ of rank 2', ha='center', fontsize=SMALL, color='0.3')
for t in (2, 5, 8): ax.add_patch(FancyArrowPatch(NODES[t], (8.6, 5.6), arrowstyle='-', color='0.55', lw=0.55, shrinkA=4.5, zorder=2))
ax.add_patch(FancyArrowPatch((8.6, 5.6), (9.2, 5.6), arrowstyle='-|>', color='0.3', lw=0.8, mutation_scale=6))
im = ax.inset_axes([0.445, 0.28, 0.2, 0.5]); im.imshow(Wsub, cmap='RdBu_r', vmin=-np.abs(Wsub).max(), vmax=np.abs(Wsub).max(), interpolation='nearest'); im.set_xticks([]); im.set_yticks([]); im.set_title('$W_{\\mathrm{rec}}$', fontsize=SMALL, pad=2)
ax.text(13.6, 5.6, '=', ha='center', va='center', fontsize=PS*10, color='0.3')
for k, (c, lab) in enumerate(((A_COL, 'sample'), (RED, 'choice'))):
    xx = 14.4 + k * 2.6; ax.add_patch(Rectangle((xx, 3.9), 0.4, 2.6, fc=c, ec='none', alpha=0.85)); ax.add_patch(Rectangle((xx + 0.55, 6.7), 1.7, 0.4, fc=c, ec='none', alpha=0.55))
    ax.text(xx + 0.2, 3.55, f'$m_{k}$', ha='center', va='top', fontsize=SMALL); ax.text(xx + 1.4, 7.25, f'$n_{k}^{{\\mathrm{{T}}}}$', ha='center', fontsize=SMALL); ax.text(xx + 1.1, 2.6, lab, ha='center', fontsize=SMALL, color=c)
    if k == 0: ax.text(xx + 2.45, 5.2, '+', ha='center', va='center', fontsize=PS*9, color='0.3')
ax.text(19.6, 1.9, '$W_{\\mathrm{rec}} = (m_0 n_0^{\\mathrm{T}} + m_1 n_1^{\\mathrm{T}})/N$', ha='right', fontsize=SMALL, color='0.3'); ax.text(19.6, 0.7, 'state $\\kappa_j = n_j^{\\mathrm{T}} r/N$;  lick iff $\\kappa_1 > 0$', ha='right', fontsize=SMALL, color='0.3')

# ── b: the curriculum — the three stages as trial timelines, what each stage trains and freezes ──
ax = cell(0, 10, 20); ax.axis('off'); ax.set_xlim(-2.6, 12.2); ax.set_ylim(-0.6, 10.2); ax_b = ax; ax.set_title('the curriculum', loc='left', fontsize=TITLE_FS)
GREY2 = '0.55'
def epoch(ax, y, t0, t1, col, lab, txt='w', hatch=None):
    ax.add_patch(Rectangle((t0, y - 0.32), t1 - t0, 0.64, fc=col, ec='none', hatch=hatch, alpha=(0.9 if hatch is None else 0.35))); ax.text((t0 + t1) / 2, y, lab, ha='center', va='center', fontsize=STAT_FS, color=txt)
ST = [('DPA', 9.0, 11.0, 'all parameters free · 250 epochs'), ('GNG', 5.9, 6.0, 'sample mode $m_0, n_0$ and sample/test\ninputs frozen · 100 epochs'), ('Dual', 2.4, 11.0, 'all inputs frozen · 150 epochs')]
for k_, (nm, y, T, what) in enumerate(ST):
    ax.text(-2.5, y, nm, fontsize=PS*8, fontweight='bold', va='center'); ax.plot([0, T], [y - 0.5, y - 0.5], color='0.4', lw=0.7); ax.text(0, y - 0.62, '0', fontsize=STAT_FS, color='0.4', ha='center', va='top'); ax.text(T, y - 0.62, f'{T:g} s', fontsize=STAT_FS, color='0.4', ha='center', va='top')
    ax.text(-2.5, y - 1.05, what, fontsize=STAT_FS, color='0.35', va='top')
    if nm in ('DPA', 'Dual'):
        ax.add_patch(Rectangle((2, y - 0.32), 1, 0.32, fc=A_COL, ec='none')); ax.add_patch(Rectangle((2, y), 1, 0.32, fc=B_COL, ec='none')); ax.text(2.5, y + 0.45, 'sample A/B' if nm == 'DPA' else 'A/B', ha='center', va='bottom', fontsize=STAT_FS, color='0.4')
        epoch(ax, y, 8, 9, GREY2, ''); ax.text(8.5, y + 0.45, 'test C/D', ha='center', va='bottom', fontsize=STAT_FS, color='0.4'); ax.add_patch(Rectangle((8.5, y - 0.32), 0.5, 0.64, fc='none', ec=RED, lw=1.0, hatch='////')); ax.text(9.15, y, 'pair?', fontsize=STAT_FS, color=RED, va='center')
    if nm in ('GNG', 'Dual'):
        t0 = 2 if nm == 'GNG' else 4; ax.add_patch(Rectangle((t0, y - 0.32), 1, 0.32, fc=BLUE, ec='none')); ax.add_patch(Rectangle((t0, y), 1, 0.32, fc=GREEN, ec='none')); ax.text(t0 + 0.5, y + 0.45, 'Go / NoGo' if nm == 'GNG' else 'Go/NoGo', ha='center', va='bottom', fontsize=STAT_FS, color='0.4')
        c0 = 4 if nm == 'GNG' else 6; epoch(ax, y, c0, c0 + 0.5, '0.75', ''); ax.text(c0 + 0.25, y + 0.45, 'cue', ha='center', va='bottom', fontsize=STAT_FS, color='0.4'); ax.add_patch(Rectangle((c0, y - 0.32), 0.5, 0.64, fc='none', ec=RED, lw=1.0, hatch='////'))
        if nm == 'GNG': ax.text(c0 + 0.7, y, 'lick?', fontsize=STAT_FS, color=RED, va='center')
    if k_ < 2: ax.annotate('', (-1.9, y - 1.9), (-1.9, y - 1.2), arrowprops=dict(arrowstyle='->', color='0.5', lw=0.9))
ax.text(5.0, -0.25, 'blue Go, green NoGo; hatched red: the response window (lick iff $\\kappa_1 > 0$)\n8 networks from 8 random initializations', fontsize=STAT_FS, color='0.35', ha='center', va='top')

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
ax.text(0.3, -4.5, 'Above: each element acts on the\nplane by a sign-flip matrix D; the\nmemory wells a network that\nrespects it may have. Choice wells\nin the autonomous field are\noptional: the choice is scored\nduring the test (Extended Data).\n\nBelow: the simulated flow at the\nDPA checkpoint of a network trained\nwith that element (or the whole\ngroup) held exactly, with the memory\nwells of every network trained\nunder the same tie.', fontsize=STAT_FS, va='top', color='0.3')
ELEMS = [('σ₁', 'A↔B and C↔D', [np.diag([-1, 1])], B_COL, 'mirror pair,\none height'),
         ('σ₂', 'A↔B,\nlick↔no lick', [-np.eye(2)], PLUM, 'antipodal pair'),
         ('σ₃', 'C↔D,\nlick↔no lick', [np.diag([1, -1])], LICK_COL, 'on the κ₀ axis,\nunrelated a, a′'),
         ('V', 'the whole group', [np.diag([-1, 1]), -np.eye(2), np.diag([1, -1])], VCOL, 'pinned pair (±a, 0)\nor the quadruple')]
def draw_matrix(ax, D, x, y, label, col='0.3'):
    """a 2×2 matrix with its entries, drawn at axes coordinates (x, y) = top-left, with bracket strokes"""
    ax.text(x, y - 0.075, label, transform=ax.transAxes, ha='left', va='center', fontsize=SMALL, color=col)
    x0 = x + 0.17; w, h = 0.13, 0.085
    for i_ in range(2):
        for j_ in range(2): ax.text(x0 + 0.06 + j_ * w, y - 0.035 - i_ * h, f'{int(D[i_, j_]):d}'.replace('-', '−'), transform=ax.transAxes, ha='center', va='center', fontsize=SMALL, color=col)
    for xb, d_ in ((x0 - 0.01, 1), (x0 + 0.12 + w, -1)):
        ax.plot([xb + 0.02 * d_, xb, xb, xb + 0.02 * d_], [y + 0.01, y + 0.01, y - 0.08 - h + 0.005, y - 0.08 - h + 0.005], transform=ax.transAxes, color=col, lw=0.8, clip_on=False)
ICOL = [B_COL, PLUM, LICK_COL]
sg = gs[1, 6:20].subgridspec(1, 4, wspace=0.45); axs_e = [fig.add_subplot(sg[0, k]) for k in range(4)]
for ax, (nm, rel, Ds, col, note) in zip(axs_e, ELEMS):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(0.9)
    ax.axvline(0, color='0.88', lw=0.6); ax.axhline(0, color=LICK_COL, lw=0.8, ls='--')
    p_ = np.array([0.45, 0.5]); ax.plot(*p_, 'o', color='k', ms=4, zorder=5)
    for D, c_ in zip(Ds, ([col] if nm != 'V' else ICOL)):
        q_ = D @ p_; ax.plot(*q_, 'o', mfc='none', mec=c_, mew=1.2, ms=5, zorder=5); ax.annotate('', q_, p_, arrowprops=dict(arrowstyle='->', color=c_, lw=1.0, linestyle=':', shrinkA=4, shrinkB=4))
    wa, wb = {'σ₁': ((0.95, -0.3), (-0.95, -0.3)), 'σ₂': ((0.95, 0.3), (-0.95, -0.3)), 'σ₃': ((0.95, 0.0), (-0.65, 0.0)), 'V': ((0.95, 0.0), (-0.95, 0.0))}[nm]
    ax.plot([wa[0], wb[0]], [wa[1], wb[1]], ':', color=col, lw=0.8, zorder=3)
    ax.plot(*wa, 'o', color=A_COL, ms=4.5, mec='w', mew=0.4, zorder=6); ax.plot(*wb, 'o', color=B_COL, ms=4.5, mec='w', mew=0.4, zorder=6)
    ax.text(wa[0], wa[1] - 0.2, 'A', ha='center', va='top', fontsize=SMALL, color=A_COL); ax.text(wb[0], wb[1] - 0.2, 'B', ha='center', va='top', fontsize=SMALL, color=B_COL)
    if nm == 'V':   # the quadruple the group also allows, faint
        for sx in (1, -1):
            for sy in (1, -1): ax.plot(0.72 * sx, 0.62 * sy, 'o', mfc='none', mec=(A_COL if sx > 0 else B_COL), mew=0.8, ms=4, alpha=0.5, zorder=4)
    ax.set_title(f'{nm}\n{rel}', loc='left', fontsize=TITLE_FS, color=col)
    if nm == 'V': ax.text(0.04, 0.97, '{I, D₁, D₂, D₃}', transform=ax.transAxes, ha='left', va='top', fontsize=SMALL, color='0.3')
    else:
        D0 = Ds[0]; k_ = [e_[0] for e_ in ELEMS].index(nm) + 1; ent = lambda v: ('-' if v < 0 else '') + str(abs(int(v)))
        ax.text(0.04, 0.97, f'$D_{k_}=\\left[\\genfrac{{}}{{}}{{0}}{{}}{{{ent(D0[0,0])}}}{{{ent(D0[1,0])}}}\\ \\ \\genfrac{{}}{{}}{{0}}{{}}{{{ent(D0[0,1])}}}{{{ent(D0[1,1])}}}\\right]$', transform=ax.transAxes, ha='left', va='top', fontsize=PS*7, color=col)
    ax.text(0.5, -0.06, note, transform=ax.transAxes, ha='center', va='top', fontsize=SMALL, color=col)
axs_e[0].set_ylabel('$\\kappa_1$ choice')

# ── c, lower row: the simulated flow of a network tied to each element (DPA checkpoint), under its scheme ──
TA = os.environ.get('TIEARM', 'symdpa_{}')   # tie arm name pattern: symdpa_{} (hinge scaffolds) or lg_{} (sweep_lif_log)
EX = [('σ₁', TIES['pair'], TA.format('pair'), 0), ('σ₂', TIES['inv'], TA.format('inv'), 0), ('σ₃', TIES['test'], TA.format('test'), 1), ('V', TIES['klein'], TA.format('klein'), 2)]
sg2 = gs[2, 6:20].subgridspec(1, 4, wspace=0.45); axs_cf = [fig.add_subplot(sg2[0, k]) for k in range(4)]
for ax, (nm, sw, arm, sd), (_, _, _, col, _) in zip(axs_cf, EX, ELEMS):
    m0, cfg = load_run(sw, f's{sd}_{arm}', stage='dpa', device='cpu'); sig = sig_of(cfg)
    cache, spd = _flow_panel_cache(m0, dict(name=nm, dims=None, conds=[]), cfg['input_size'], torch.zeros(1, 1, cfg['input_size']), XL, XL, 61, field_input_noise=sig, n_fp_seeds=41, slow_tol=0.06)
    _render_flow_panel(ax, cache, speed_vmax=float(np.percentile(spd, 98)), sim_scattered=False, kappa_traj=None, cond_idx={}, colors={}, xlim=XL, ylim=XL, model=m0)
    ax.axhline(0, color='w', lw=0.8, ls=(0, (4, 3)))
    for s_ in range(4):   # every seed's memory pair under this tie
        try: m, cf = load_run(sw, f's{s_}_{arm}', stage='dpa', device='cpu')
        except Exception: continue
        A, B = memory_pair(wells_of(m, cf, sig_of(cf)), m, cf)
        for f, c_ in ((A, A_COL), (B, B_COL)):
            if f is not None: ax.plot(f[0], f[1], 'o', ms=3.2, mfc=c_, mec='w', mew=0.4, zorder=7)
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(0.9)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlabel('$\\kappa_0$ sample'); ax.set_ylabel('$\\kappa_1$ choice' if nm == 'σ₁' else ''); ax.set_title(f'{nm} tied', loc='left', fontsize=TITLE_FS, color=col)
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
ax = cell(3, 10, 20); ax_f1 = ax
for s_ in fseeds:
    for c_, col, mk in ((4, BLUE, 's'), (5, GREEN, 'o')):
        ax.plot(range(3), [OV[f'{s_}|{st}'][str(c_)][1] for st in ('dpa', 'naive', 'expert')], '-', marker=mk, color=col, lw=0.7, ms=2.8, alpha=0.55)
        ax.plot(range(3), [OV[f'{s_}|{st}'][str(c_)][0] for st in ('dpa', 'naive', 'expert')], '-', marker=mk, color=col, lw=0.6, ms=2.2, alpha=0.25, mfc='white')
for c_, col in ((4, BLUE), (5, GREEN)): ax.plot(range(3), [np.median([OV[f'{s_}|{st}'][str(c_)][1] for s_ in fseeds]) for st in ('dpa', 'naive', 'expert')], '-', color=col, lw=2.0, zorder=5)
ax.axhline(0, color='0.75', lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels(stages, rotation=20); ax.set_ylabel('overlap $n_j^{\\mathrm{T}} w_c / N$')
ax.set_title('choice readout hears Go/NoGo', loc='left', fontsize=TITLE_FS)
ax.set_ylim(-5.4, 3.8); ax.text(0.02, 0.03, 'filled $n_1$ (choice): σ₂, σ₃ force 0\nopen $n_0$ (sample): σ₁ forces 0\nblue Go, green NoGo; thick, median', transform=ax.transAxes, va='bottom', fontsize=STAT_FS, color='0.3')
# f3: the equivariance residual of the autonomous field per element
LED = {}
for l in open(os.environ.get('SVT', '/home/leon/dual/figures/paper_share/modelling/symviol_bfix.tsv')).read().strip().split('\n')[1:]:
    r = l.split('\t')
    if len(r) < 5 or not r[2][0].isdigit(): continue
    LED[(int(r[0].split('_')[0][1:]), r[1])] = [float(r[2]), float(r[3]), float(r[4])]
ax = cell(4, 10, 20); ax_f2 = ax
for e_i, (el, col) in enumerate((('σ₁', B_COL), ('σ₂', PLUM), ('σ₃', LICK_COL))):
    ys = np.array([[LED[(s_, st)][e_i] for st in ('dpa', 'naive', 'expert')] for s_ in fseeds if all((s_, st) in LED for st in ('dpa', 'naive', 'expert'))])
    for y in ys: ax.plot(range(3), y, '-o', color=col, lw=0.6, ms=2.4, alpha=0.4)
    ax.plot(range(3), np.median(ys, 0), '-', color=col, lw=2.0, zorder=5, label=el)
ax.set_xticks(range(3)); ax.set_xticklabels(stages, rotation=20); ax.set_ylabel('equivariance residual'); ax.set_ylim(0, 1.15)
ax.set_title('the field keeps σ₁', loc='left', fontsize=TITLE_FS); ax.legend(loc='upper left', frameon=False, fontsize=SMALL, handlelength=1.2)

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
ax = cell(5, 0, 5); ax_d = ax
H = {idx: np.array([[wells_free[(st, s_)][idx][1] / wells_free[(st, s_)][2] if wells_free[(st, s_)][idx] is not None else np.nan for st in ('dpa', 'naive', 'expert')] for s_ in fseeds]) for idx in (0, 1)}
for i, s_ in enumerate(fseeds):
    ax.plot(np.arange(3) - 0.07, H[0][i], '-o', color=SEED_COL[i], alpha=0.6, lw=0.7, ms=2.8); ax.plot(np.arange(3) + 0.07, H[1][i], '-o', color=SEED_COL[i], alpha=0.6, lw=0.7, ms=2.8, mfc='white')
rng_ = np.random.default_rng(0)
for idx, dx in ((0, -0.07), (1, 0.07)):
    mu = np.nanmean(H[idx], 0); boots = np.array([np.nanmean(H[idx][rng_.integers(0, len(fseeds), len(fseeds))], 0) for _ in range(2000)]); lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    ax.errorbar(np.arange(3) + dx, mu, yerr=[mu - lo, hi - mu], fmt='-', color='k', lw=1.5, capsize=2, elinewidth=1.0, zorder=5, marker=('o' if idx == 0 else 'o'), mfc=('k' if idx == 0 else 'white'), ms=4)
def _wil(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); return stats.wilcoxon(a[ok], b[ok]).pvalue, int(ok.sum())
pvals = {}; nwil = {}
for idx in (0, 1): pvals[idx], nwil[idx] = _wil(H[idx][:, 2], H[idx][:, 0])
for idx, xx in ((0, 1.45), (1, 1.85)):
    sig = pvals[idx] < 0.05; ax.text(xx, 0.62, '∗' if sig else 'n.s.', ha='center', fontsize=(12 if sig else 8), fontweight='bold', color=('k' if sig else '0.55'))
ax.text(0.02, 0.03, f'Wilcoxon DPA→Dual\nA p = {pvals[0]:.3f}, B p = {pvals[1]:.3f}', transform=ax.transAxes, fontsize=STAT_FS, color='0.3')
ax.axhline(0, color=LICK_COL, lw=0.8, ls='--'); ax.set_xticks(range(3)); ax.set_xticklabels(stages, rotation=20); ax.set_ylabel('well height, $\\kappa_1/\\eta$'); ax.set_ylim(-2.3, 0.9)
nb = sum(1 for s_ in fseeds if wells_free[('expert', s_)][0] is not None and wells_free[('expert', s_)][1] is not None and wells_free[('expert', s_)][0][1] / wells_free[('expert', s_)][2] < BELOW and wells_free[('expert', s_)][1][1] / wells_free[('expert', s_)][2] < BELOW)
ax.set_title(f'the push ({nb}/{len(fseeds)} below)', loc='left', fontsize=TITLE_FS); ax.text(0.02, 0.95, 'A filled, B open', transform=ax.transAxes, ha='left', va='top', fontsize=SMALL, color='0.3')

# ── f, g, h: depth ↔ performance by perturbation — a delay-only drive along m1 moves the wells on the lick axis (the test-driven field is untouched);
#      DPA and GNG accuracy against the well location, two wells (A filled, B open) per drive strength, per network (tab10) ──
PJ = json.load(open(os.environ.get('PJF', '/home/leon/dual/figures/paper_share/modelling/perturb_depth.json')))
P = {}
for k, v in PJ.items():
    s_, var, d = k.split('|'); P.setdefault((int(s_), var), []).append((float(d), v))
ax_g = cell(5, 5, 10); ax_h = cell(5, 10, 15); ax_i = cell(5, 15, 20)
def binned(ax, xk, yk, col):
    """all (network, sample, drive) points, gray; binned mean ± 95% bootstrap CI in fixed 0.5 η bins, coloured"""
    X = np.array([r[xk + lab] for s_ in fseeds for d, r in P[(s_, 'drive')] for lab in ('A', 'B')]); Y = np.array([r[yk + lab] for s_ in fseeds for d, r in P[(s_, 'drive')] for lab in ('A', 'B')])
    XA = np.array([r[xk + 'A'] for s_ in fseeds for d, r in P[(s_, 'drive')]]); YA = np.array([r[yk + 'A'] for s_ in fseeds for d, r in P[(s_, 'drive')]])
    XB = np.array([r[xk + 'B'] for s_ in fseeds for d, r in P[(s_, 'drive')]]); YB = np.array([r[yk + 'B'] for s_ in fseeds for d, r in P[(s_, 'drive')]])
    ax.scatter(XA, YA, s=4, color='0.55', alpha=0.45, edgecolors='none', zorder=2); ax.scatter(XB, YB, s=6, facecolors='none', edgecolors='0.55', linewidths=0.4, alpha=0.5, zorder=2)
    edges = np.arange(np.floor(X.min() * 2) / 2, np.ceil(X.max() * 2) / 2 + 0.01, 0.5); xc, mu, lo, hi = [], [], [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        sel = (X >= a_) & (X < b_)
        if sel.sum() < 4: continue
        boots = np.array([np.mean(Y[sel][rng_.integers(0, sel.sum(), sel.sum())]) for _ in range(1000)]); xc.append((a_ + b_) / 2); mu.append(Y[sel].mean()); lo.append(np.percentile(boots, 2.5)); hi.append(np.percentile(boots, 97.5))
    mu = np.array(mu); ax.errorbar(xc, mu, yerr=[mu - np.array(lo), np.array(hi) - mu], fmt='-o', color=col, lw=1.4, ms=3.5, capsize=2, elinewidth=0.9, zorder=6)
    ax._binned = (np.array(xc), mu)
    return stats.spearmanr(X, Y)
for s_ in fseeds:   # dual performance = P(both responses of a dual trial correct) = DPA accuracy × GNG accuracy, per (sample, drive)
    for d, r in P[(s_, 'drive')]:
        for lab in ('A', 'B'): r['dual_perf_' + lab] = r['dpa_acc_' + lab] * r['gng_acc_' + lab]
rho_d, p_d = binned(ax_g, 'depth_', 'dpa_acc_', RED); rho_g, p_g = binned(ax_h, 'depthU_', 'gng_acc_', BLUE); rho_i, p_i = binned(ax_i, 'depthU_', 'dual_perf_', '0.15')
for a_, ttl, yl in ((ax_g, 'DPA vs well at the test', 'DPA accuracy'), (ax_h, 'GNG vs well at the cue', 'GNG accuracy'), (ax_i, 'dual, both correct', 'DPA × GNG accuracy')):
    a_.axvline(0, color=LICK_COL, lw=0.8, ls='--'); a_.set_ylim(0.35, 1.03); a_.set_xlabel('well on the choice axis, $\\kappa_1/\\eta$'); a_.set_ylabel(yl); a_.set_title(ttl, loc='left', fontsize=TITLE_FS)
ax_i.axvline(-0.9, color='0.5', lw=0.8, ls=':')
xc_, mu_ = ax_i._binned; ax_i.plot(-xc_, mu_, ':', color=LICK_COL, lw=1.2, zorder=5)   # the image of the curve under σ₃ (κ₁ → −κ₁): equal to the curve only if σ₃ held
ax_g.text(0.5, 0.04, f'left: misses; right: false alarms\nA filled, B open: coincide (σ₁)\nbins 0.5 η, mean ± 95% CI', transform=ax_g.transAxes, ha='center', fontsize=STAT_FS, color='0.3')
ax_h.text(0.02, 0.04, f'NoGo licks only\nρ = {rho_g:.2f}, p = {p_g:.0e}', transform=ax_h.transAxes, fontsize=STAT_FS, color='0.3')
ax_i.text(0.98, 0.97, 'gray dotted: trained well\nred dotted: σ₃ image of the curve\nlick = κ₁ > 0 at any step of the window', transform=ax_i.transAxes, ha='right', va='top', fontsize=STAT_FS, color='0.3')

for L, a in zip('abcdefgh', [fig.axes[0], ax_b, ax_e0, ax_f0, ax_d, ax_g, ax_h, ax_i]): panel_letter(fig, a, L)
save(fig, 'fig5_model')

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
ax.text(0.0, -0.24, f'six populations:\nfour mixed (colour, sample\nsign; opaque, choice +) and a\npure-choice pair ({axu.sum()} units, gray)', transform=ax.transAxes, va='top', fontsize=SMALL, color='0.3')
ax.set_title('units, whole-group tie', loc='left', fontsize=TITLE_FS); ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect('equal', adjustable='box')

ax_f = ax

for L, a in zip('abc', [ax_e18a, ax_f, axs_c[0]]): panel_letter(fig, a, L)
save(fig, 'ed18'); print('ED18 done')

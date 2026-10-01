"""Every number the modelling draft quotes about the Fig. 5 networks, recomputed on sweep_lif_log (free ×8, ties ×4 each),
with the same definitions as the figures (fig5_model.py, perturb_depth.py). Prints a report and writes draft_numbers_log.json.
Run: LD_PRELOAD=... python draft_numbers_log.py"""
import sys, os, json, numpy as np, torch; torch.set_num_threads(16)
sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad'); sys.path.insert(0, '/home/leon/rnn/paper')
from scipy import stats
from symtools import load_run, sig_of
from bifurcation_probe import find_wells, run_dt_alpha
from src.tasks import make_timings, generate_dpa_trials, generate_dual_trials
os.chdir('/home/leon/rnn'); SW = 'results/dual/sweep_lif_log'; M = '/home/leon/dual/figures/paper_share/modelling'
NS = int(os.environ.get('NSEEDS', '8')); FREE = [f's{s}_log' for s in range(NS)]; OUT = {}   # NSEEDS=16: the 16-network set (2026-10-01)
SVT, OVJ, PJF = (os.environ.get(k, d) for k, d in (('SVT', 'symviol_log.tsv'), ('OVJ', 'overlaps_log.json'), ('PJF', 'perturb_depth.json')))
OUTJ = os.environ.get('OUTJ', 'draft_numbers_log.json'); SKIP_C = bool(os.environ.get('SKIP_C')); PJ_LEARNERS = bool(os.environ.get('PJ_LEARNERS'))
def rng_(v): v = np.asarray(v, float); return f'{np.nanmin(v):+.2f} … {np.nanmax(v):+.2f}'

# ── A. accuracies (results.jsonl, the window criterion used by the sweep) ──
R = {json.loads(l)['run_id']: json.loads(l) for l in open(f'{SW}/results.jsonl')}
acc = {rid: R[rid]['accuracy'] for rid in R}
print('A. accuracies (free)'); OUT['acc'] = {}
for rid in FREE:
    a = acc[rid]; row = dict(dpa=a['after_dpa']['dpa'], gng_dpa=a['after_gng']['dpa'], dual_dpaonly=a['after_dual'].get('dpa', float('nan')),
                            dual_dpa=a['after_dual']['dual_dpa'], go=a['after_dual']['dual_go'], nogo=a['after_dual']['dual_nogo'])
    OUT['acc'][rid] = row; print(f'   {rid}: ' + ' '.join(f'{k} {v:.3f}' for k, v in row.items()))
LEARN = [r for r in FREE if OUT['acc'][r]['dpa'] >= 0.9]; OUT['learners'] = LEARN; print('   learners:', LEARN)

# ── B. lick probabilities at the three checkpoints (ED 18a definitions; any-time lick in the window) ──
def lick_probs(sw, rid, stage):
    m, cfg = load_run(sw, rid, stage=stage, device='cpu'); dt, alpha, _ = run_dt_alpha(cfg); eta = cfg['noise'] * float(np.sqrt(1 - np.exp(-alpha) ** 2)); torch.manual_seed(1)
    Td = make_timings(dt)['dual']; half = int(round(0.5 / dt))
    X, y, _, names = generate_dual_trials(384, Td, cfg['input_size'], noise=eta, target_rank=2, cue_on_go_input=cfg['cue_on_go_input'], cue_scale=cfg['cue_scale'],
                                          nogo_target=cfg['nogo_target'], go_target=cfg['go_target'], response_in_cue=cfg['response_in_cue'],
                                          gng_response=cfg.get('gng_response', True), gng_memory=cfg.get('dual_gng_memory', False), attention_input=cfg.get('attention_input', False))
    with torch.no_grad(): k1 = m(X)[..., 1].numpy()
    to, co = int(Td.n_stim_off[3]), int(Td.n_stim_off[2]); ric = cfg['response_in_cue']
    L = lambda seg: (seg > 0).any(1)
    test_lick = L(k1[:, (to - half):to] if ric else k1[:, to:to + half]); cue_lick = L(k1[:, (co - half):co] if ric else k1[:, co:co + half])
    samp = np.array([n[0] for n in names]); tst = np.array([n[-1] for n in names]); pair = ((samp == 'A') & (tst == 'C')) | ((samp == 'B') & (tst == 'D'))
    go = np.array(['_go_' in n for n in names]); ng = np.array(['_nogo_' in n for n in names])
    return dict(go_dpa=(test_lick[go & pair].mean() + 1 - test_lick[go & ~pair].mean()) / 2,
                nogo_dpa=(test_lick[ng & pair].mean() + 1 - test_lick[ng & ~pair].mean()) / 2,
                go_cue=cue_lick[go].mean(), nogo_cue_correct=1 - cue_lick[ng].mean())
print('B. dual-trial behaviour at the naive checkpoint (before dual training), any-time lick')
LPn = {rid: lick_probs(SW, rid, 'naive') for rid in FREE}; OUT['naive_lick'] = LPn
for rid in FREE: print(f'   {rid}: DPA choice on Go trials {LPn[rid]["go_dpa"]:.2f}, NoGo correct at the cue {LPn[rid]["nogo_cue_correct"]:.2f}')
for grp, ids in ((f'all {len(FREE)}', FREE), ('learners', LEARN)):
    print(f'   median ({grp}): DPA on Go trials {np.median([LPn[r]["go_dpa"] for r in ids]):.2f}; NoGo at the cue {np.median([LPn[r]["nogo_cue_correct"] for r in ids]):.2f}')

# ── C. memory-well heights (occupied wells of the noise-averaged field, as fig5 memory_pair), in η ──
def memory_pair(m, cfg, sig):
    w = [np.asarray(f[:2], float) for f, kd in find_wells(m, cfg, xlim=2.5, n_seeds=41, noise_sigma=sig) if str(kd).lower().startswith(('stable', 'attract')) and np.linalg.norm(f) > 0.15]
    dt, alpha, _ = run_dt_alpha(cfg); T = make_timings(dt)['dpa']; torch.manual_seed(0)
    X, _ = generate_dpa_trials(48, T, cfg['input_size'], noise=sig, target_rank=2, windowed_targets=True, decay_to_zero=False, response_in_cue=cfg['response_in_cue'],
                               prelick_free=True, hold_window=0.5, hold_anchor='sample', post_response_window=cfg.get('dpa_post_response_window'))
    with torch.no_grad(): k = m(X)[:, int(T.n_stim_on[1]) - 1].numpy()
    isA = X[:, int(T.n_stim_on[0]):int(T.n_stim_off[0]), 0].mean(1).numpy() > 0.5
    kA, kB = k[isA].mean(0), k[~isA].mean(0); L_ = [f for f in w if f[0] < -0.5]; R_ = [f for f in w if f[0] > 0.5]
    if not (L_ and R_): return None, None
    near = lambda pt: min(w, key=lambda f: np.linalg.norm(f - pt))
    return near(kA), near(kB)
print('C. memory-well heights (η) at dpa / naive / expert' + (' — SKIPPED (SKIP_C; Fig. 5e prints them)' if SKIP_C else ''))
H = {}
for rid in ([] if SKIP_C else FREE):
    H[rid] = {}
    for st in ('dpa', 'naive', 'expert'):
        m, cfg = load_run(SW, rid, stage=st, device='cpu'); sig = sig_of(cfg); A, B = memory_pair(m, cfg, sig)
        H[rid][st] = [None if A is None else float(A[1] / sig), None if B is None else float(B[1] / sig)]
    print(f'   {rid}: ' + ' | '.join(f'{st} ' + ' '.join('—' if v is None else f'{v:+.2f}' for v in H[rid][st]) for st in ('dpa', 'naive', 'expert')), flush=True)
OUT['heights_eta'] = H
if not SKIP_C:
    for st in ('dpa', 'naive', 'expert'):
        vals = [v for r in LEARN for v in H[r][st] if v is not None]; print(f'   {st} (learners): {rng_(vals)}')
    both = [r for r in FREE if all(v is not None and v < -0.25 for v in H[r]['expert'])]; print(f'   both wells below −0.25 η after Dual: {len(both)}/{len(FREE)} {both}')
    dA = [H[r]['expert'][0] - H[r]['dpa'][0] for r in FREE if None not in (H[r]['expert'][0], H[r]['dpa'][0])]; dB = [H[r]['expert'][1] - H[r]['dpa'][1] for r in FREE if None not in (H[r]['expert'][1], H[r]['dpa'][1])]
    print(f'   Wilcoxon DPA→Dual: A p = {stats.wilcoxon(dA).pvalue:.3f} (n={len(dA)}), B p = {stats.wilcoxon(dB).pvalue:.3f} (n={len(dB)})')

# ── D–F. ⟨n₁⟩, input overlaps, residual medians (tables computed for Fig. 5d) ──
SV = [l.rstrip('\n').split('\t') for l in open(f'{M}/{SVT}')][1:]
sv = {(r[0], r[1]): r for r in SV if len(r) > 6 and r[2][0].isdigit()}
print('D. <n1> by stage (learners):', {st: rng_([float(sv[(r, st)][6]) for r in LEARN]) for st in ('dpa', 'naive', 'expert')})
print(f'F. residual medians σ1/σ2/σ3 (all {len(FREE)}):', {st: [round(float(np.median([float(sv[(r, st)][c]) for r in FREE])), 2) for c in (2, 3, 4)] for st in ('dpa', 'naive', 'expert')})
OV = json.load(open(f'{M}/{OVJ}'))
for st in ('dpa', 'naive', 'expert'):
    g = [OV[f'{s}|{st}']['4'][1] for s in range(NS)]; n_ = [OV[f'{s}|{st}']['5'][1] for s in range(NS)]; n0 = [abs(OV[f'{s}|{st}'][c][0]) for s in range(NS) for c in ('4', '5')]
    print(f'E. {st}: n1·w_Go {rng_(g)}, n1·w_NoGo {rng_(n_)}, max |n0·w_Go/NoGo| {max(n0):.2f}')

# ── G. ties: after-Dual accuracies (DPA-checkpoint wells are in the tie table) ──
print('G. ties after Dual (window criterion)')
for arm in ('lg_pair', 'lg_inv', 'lg_test', 'lg_klein'):
    ids = [f's{s}_{arm}' for s in range(4)]
    print(f'   {arm}: dual DPA {rng_([acc[r]["after_dual"]["dual_dpa"] for r in ids])}, Go {rng_([acc[r]["after_dual"]["dual_go"] for r in ids])}, NoGo {rng_([acc[r]["after_dual"]["dual_nogo"] for r in ids])}')

# ── H. the perturbation (perturb_depth.json, same pooling and 0.5 η bins as Fig. 5f–h) ──
PJ = json.load(open(f'{M}/{PJF}')); P = {}
for k, v in PJ.items():
    s_, var, d = k.split('|')
    if var == 'drive' and (not PJ_LEARNERS or f's{s_}_log' in LEARN): P.setdefault(int(s_), []).append((float(d), v))
print(f'H. perturbation networks: {sorted(P)}' + (' (learners only)' if PJ_LEARNERS else ''))
pts = [(r, lab) for s_ in P for d, r in P[s_] for lab in ('A', 'B')]
X_t = np.array([r['depth_' + l] for r, l in pts]); X_c = np.array([r['depthU_' + l] for r, l in pts])
Y_d = np.array([r['dpa_acc_' + l] for r, l in pts]); Y_g = np.array([r['gng_acc_' + l] for r, l in pts]); Y_x = Y_d * Y_g
def bins(X, Y):
    e = np.arange(np.floor(X.min() * 2) / 2, np.ceil(X.max() * 2) / 2 + 0.01, 0.5); out = []
    for a_, b_ in zip(e[:-1], e[1:]):
        sel = (X >= a_) & (X < b_)
        if sel.sum() >= 4: out.append(((a_ + b_) / 2, float(Y[sel].mean()), int(sel.sum())))
    return out
print('H. perturbation, binned means (center η: mean, n)')
for nm, X, Y in (('DPA vs well at the test', X_t, Y_d), ('GNG vs well at the cue', X_c, Y_g), ('dual (product) vs well at the cue', X_c, Y_x)):
    print(f'   {nm}: ' + '  '.join(f'{c:+.2f}:{m_:.2f}' for c, m_, n in bins(X, Y)))
d0 = [r for s_ in P for d, r in P[s_] if abs(d) < 1e-9]
print(f'   trained position (drive 0): at the test {rng_([r["depth_" + l] for r in d0 for l in "AB"])} η (mean {np.mean([r["depth_" + l] for r in d0 for l in "AB"]):+.2f}); '
      f'at the cue mean {np.mean([r["depthU_" + l] for r in d0 for l in "AB"]):+.2f} η')
print(f'   at drive 0: DPA {np.mean([r["dpa_acc"] for r in d0]):.3f}, Go {np.mean([r["go"] for r in d0]):.3f}, NoGo {np.mean([r["nogo"] for r in d0]):.3f} '
      f'(range {rng_([r["nogo"] for r in d0])}), dual product {np.mean([r["dpa_acc_" + l] * r["gng_acc_" + l] for r in d0 for l in "AB"]):.3f}')
below = X_t < np.mean([r["depth_" + l] for r in d0 for l in "AB"])
print(f'   Spearman DPA vs well below the trained position: ρ = {stats.spearmanr(X_t[below], Y_d[below]).correlation:.2f}; GNG vs well at the cue ρ = {stats.spearmanr(X_c, Y_g).correlation:.2f}')
# NoGo and Go separately per drive (network level), binned by the mean well at the cue
Xn = np.array([(r['depthU_A'] + r['depthU_B']) / 2 for s_ in P for d, r in P[s_]]); Yng = np.array([r['nogo'] for s_ in P for d, r in P[s_]]); Ygo = np.array([r['go'] for s_ in P for d, r in P[s_]])
print('   NoGo vs well at the cue: ' + '  '.join(f'{c:+.2f}:{m_:.2f}' for c, m_, n in bins(Xn, Yng)))
print('   Go   vs well at the cue: ' + '  '.join(f'{c:+.2f}:{m_:.2f}' for c, m_, n in bins(Xn, Ygo)))
OUT['trained_well_cue_eta'] = float(np.mean([r["depthU_" + l] for r in d0 for l in "AB"]))

# ── I. the depth law: d from the delay, k from the paired response (DPA-only trials, expert, trained noise) ──
print('I. depth law d/(k/2)')
ratios = {}
for rid in FREE:
    m, cfg = load_run(SW, rid, stage='expert', device='cpu'); sig = sig_of(cfg); dt, _, _ = run_dt_alpha(cfg); T = make_timings(dt)['dpa']; torch.manual_seed(2)
    X, _ = generate_dpa_trials(1024, T, cfg['input_size'], noise=sig, target_rank=2, windowed_targets=True, decay_to_zero=False, response_in_cue=cfg['response_in_cue'],
                               prelick_free=True, hold_window=0.5, hold_anchor='sample', post_response_window=cfg.get('dpa_post_response_window'))
    m.noise = 0.0
    with torch.no_grad(): k1 = m(X)[..., 1].numpy()
    half = int(round(0.5 / dt)); t0, t1 = int(T.n_stim_on[1]), int(T.n_stim_off[1])
    sA = X[:, int(T.n_stim_on[0]):int(T.n_stim_off[0]), 0].mean(1).numpy() > 0.5; tC = X[:, t0:t1, 2].mean(1).numpy() > 0.5; pr = (sA & tC) | (~sA & ~tC)
    dly = k1[:, t0 - half:t0].mean(1); rsp = k1[:, t1 - half:t1].mean(1)
    d = -dly.mean() / sig; kk = (rsp[pr] - dly[pr]).mean() / sig; ratios[rid] = (d, kk, d / (kk / 2))
    print(f'   {rid}: d {d:.2f} η, k {kk:.2f} η, d/(k/2) {d / (kk / 2):.2f}', flush=True)
rl = [ratios[r][2] for r in LEARN]; print(f'   learners: d/(k/2) = {np.mean(rl):.2f} ± {np.std(rl, ddof=1):.2f} (n={len(rl)}); k {rng_([ratios[r][1] for r in LEARN])}, d {rng_([ratios[r][0] for r in LEARN])}')
OUT['depth_law'] = {r: list(map(float, v)) for r, v in ratios.items()}
json.dump(OUT, open(f'{M}/{OUTJ}', 'w'), indent=1, default=float); print('saved', OUTJ)

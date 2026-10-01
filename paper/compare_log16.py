"""compare_log16.py — the Fig. 5f–h comparison over every network of the log recipe (sweep_lif_log free arm), seed by
seed (Leon 2026-09-30: "run more of these simulations (more seeds), so that we have more nets to compare").

Reads the perturbation runs of perturb_depth.py (the drive along m1, 81 strengths, 2048 trials): perturb_depth.json
(s0–s7, the published Fig. 5f–h data) and, if present, perturb_depth_log_s8_15.json (the extra seeds). A network counts
as a learner if it learned the DPA stage (after_dpa DPA accuracy ≥ 0.95 in results.jsonl; s0 and s7 of the first batch
fail at 0.74 / 0.73).

Per network: the trained well at the test (DPA trials) and at the cue (dual trials); Go and NoGo at the trained position;
the Go/NoGo plateau (wells at the cue where GNG accuracy ≥ 0.97); the deepest well at which Go still holds (Go ≥ 0.95);
the position of best dual performance (DPA × GNG, smoothed over 5 drives) and its distance from the trained well.
Writes: perturb_depth_log16.json (the merged runs), diag_gng_vs_well_log16.png (per-network Go/NoGo curves, pooled
learner curve, each network's optimum against its trained well) and prints the table.
"""
import sys, os, json, numpy as np
sys.path.insert(0, '/home/leon/rnn/paper')
from style import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

M = '/home/leon/dual/figures/paper_share/modelling'
RES = '/home/leon/rnn/results/dual/sweep_lif_log/results.jsonl'
PJ = json.load(open(f'{M}/perturb_depth.json'))
if os.path.exists(f'{M}/perturb_depth_log_s8_15.json'):
    PJ.update(json.load(open(f'{M}/perturb_depth_log_s8_15.json')))
json.dump(PJ, open(f'{M}/perturb_depth_log16.json', 'w'), indent=0)
P = {}
for k, v in PJ.items():
    s, var, d = k.split('|')
    if var == 'drive':
        P.setdefault(int(s), {})[float(d)] = v
acc = {int(r['run_id'].split('_')[0][1:]): r['accuracy'] for r in map(json.loads, open(RES)) if r['run_id'].endswith('_log')}
SEEDS = sorted(s for s in P if s in acc)
LEARN = [s for s in SEEDS if acc[s]['after_dpa']['dpa'] >= 0.95]
NOT = [s for s in SEEDS if s not in LEARN]
print(f'networks with perturbation runs: {SEEDS}; learners (after_dpa DPA ≥ 0.95): {LEARN}; not: {NOT}')

rows = []
for s in SEEDS:
    R = sorted(P[s].items()); z = P[s][0.0]
    du = np.array([np.mean([r['depthU_A'], r['depthU_B']]) for d, r in R])
    go = np.array([r['go'] for d, r in R]); ng = np.array([r['nogo'] for d, r in R])
    gng = np.array([np.mean([r['gng_acc_' + l] for l in 'AB']) for d, r in R])
    dual = np.array([np.mean([r['dpa_acc_' + l] * r['gng_acc_' + l] for l in 'AB']) for d, r in R])
    ds = np.convolve(dual, np.ones(5) / 5, mode='same'); i = int(np.argmax(ds[2:-2])) + 2
    zu = np.mean([z['depthU_A'], z['depthU_B']]); zd = np.mean([z['depth_A'], z['depth_B']])
    pl = du[gng >= 0.97]; goh = du[go >= 0.95]
    rows.append(dict(seed=s, learner=s in LEARN, dpa_stage=acc[s]['after_dpa']['dpa'], well_test=zd, well_cue=zu,
                     go0=z['go'], nogo0=z['nogo'], dual0=np.mean([z['dpa_acc_' + l] * z['gng_acc_' + l] for l in 'AB']),
                     plateau=(pl.min(), pl.max()) if len(pl) else (np.nan, np.nan), go_limit=goh.min() if len(goh) else np.nan,
                     best=ds[i], best_well=du[i], d_best=du[i] - zu))
print(f'\n{"net":>5} {"DPA@stage":>9} {"well@test":>9} {"well@cue":>8} {"Go@0":>5} {"NoGo@0":>6} {"dual@0":>6} | {"plateau GNG≥0.97 (well@cue)":>27} {"Go holds to":>11} | {"best dual":>9} {"at well":>7} {"best−trained":>12}')
for r in rows:
    print(f'{"" if r["learner"] else "*"}s{r["seed"]:<3} {r["dpa_stage"]:9.2f} {r["well_test"]:+9.2f} {r["well_cue"]:+8.2f} {r["go0"]:5.2f} {r["nogo0"]:6.2f} {r["dual0"]:6.2f} | '
          f'{r["plateau"][0]:+13.2f} … {r["plateau"][1]:+.2f}    {r["go_limit"]:+11.2f} | {r["best"]:9.2f} {r["best_well"]:+7.2f} {r["d_best"]:+12.2f}')
L = [r for r in rows if r['learner']]
if L:
    db = np.array([r['d_best'] for r in L]); gain = np.array([r['best'] - r['dual0'] for r in L])
    print(f'\nlearners (n = {len(L)}): best − trained well at the cue {db.mean():+.2f} ± {db.std(ddof=1) if len(L) > 1 else 0:.2f} η '
          f'(range {db.min():+.2f} … {db.max():+.2f}; deeper in {int((db < -0.1).sum())}/{len(L)}); dual gain at the optimum '
          f'{gain.mean():+.3f} (range {gain.min():+.3f} … {gain.max():+.3f}); trained wells at the cue {np.mean([r["well_cue"] for r in L]):+.2f} '
          f'(range {min(r["well_cue"] for r in L):+.2f} … {max(r["well_cue"] for r in L):+.2f})')

# ── figure: as diag_gng_vs_well.py, over every network ──
rng = np.random.default_rng(0)
cols = {s: (SEED_COL[s % 10] if s in LEARN else '0.6') for s in SEEDS}
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9)); fig.subplots_adjust(wspace=0.36, left=0.06, right=0.99, top=0.84, bottom=0.17)
ax = axs[0]
for s in SEEDS:
    R = sorted(P[s].items()); du = [np.mean([r['depthU_A'], r['depthU_B']]) for d, r in R]
    ax.plot(du, [r['go'] for d, r in R], '-', color=cols[s], lw=1.0, alpha=0.9); ax.plot(du, [r['nogo'] for d, r in R], '--', color=cols[s], lw=1.0, alpha=0.9)
    z = P[s][0.0]; ax.plot(np.mean([z['depthU_A'], z['depthU_B']]), 0.5, 'o', color=cols[s], ms=4)
ax.axvline(0, color=LICK_COL, lw=0.8, ls='--'); ax.set_xlim(-3.6, 1.6); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel('well at the cue, $\\kappa_1/\\eta$'); ax.set_ylabel('accuracy')
ax.set_title(f'a  Go (solid), NoGo (dashed), {len(SEEDS)} networks', loc='left', fontsize=TITLE_FS)
ax.text(0.97, 0.55, f'color: learners (n = {len(LEARN)})\ngray: never learned DPA (n = {len(NOT)})\ndots: trained wells', transform=ax.transAxes, ha='right', fontsize=STAT_FS, color='0.3')
ax = axs[1]
def pooled(seeds, bw):
    X = np.array([r['depthU_' + l] for s in seeds for d, r in P[s].items() for l in 'AB']); Y = np.array([r['gng_acc_' + l] for s in seeds for d, r in P[s].items() for l in 'AB'])
    xc, mu, lo, hi = [], [], [], []
    for a in np.arange(np.floor(X.min() / bw) * bw, X.max(), bw):
        sel = (X >= a) & (X < a + bw)
        if sel.sum() < 4: continue
        bt = [Y[sel][rng.integers(0, sel.sum(), sel.sum())].mean() for _ in range(1000)]
        xc.append(a + bw / 2); mu.append(Y[sel].mean()); lo.append(np.percentile(bt, 2.5)); hi.append(np.percentile(bt, 97.5))
    return np.array(xc), np.array(mu), np.array(lo), np.array(hi)
for seeds, col, lab in ((SEEDS, '0.45', f'all {len(SEEDS)}'), (LEARN, '#1f77b4', f'{len(LEARN)} learners')):
    xc, mu, lo, hi = pooled(seeds, 0.25); ax.fill_between(xc, lo, hi, color=col, alpha=0.2, lw=0); ax.plot(xc, mu, '-o', color=col, ms=3, lw=1.3, label=lab)
trL = np.mean([P[s][0.0]['depthU_' + l] for s in LEARN for l in 'AB'])
ax.axvline(trL, color='0.5', lw=0.8, ls=':'); ax.axvline(0, color=LICK_COL, lw=0.8, ls='--'); ax.text(trL + 0.05, 0.64, 'trained', fontsize=STAT_FS, color='0.4')
ax.set_xlim(-3.6, 1.6); ax.set_ylim(0.45, 1.02); ax.set_xlabel('well at the cue, $\\kappa_1/\\eta$'); ax.set_ylabel('Go/NoGo accuracy')
ax.legend(frameon=False, fontsize=STAT_FS, loc='upper right'); ax.set_title('b  pooled, 0.25 η bins, mean ± 95% CI', loc='left', fontsize=TITLE_FS)
ax = axs[2]
for s in SEEDS:
    R = sorted(P[s].items()); z = P[s][0.0]; zu = np.mean([z['depthU_A'], z['depthU_B']])
    du = np.array([np.mean([r['depthU_A'], r['depthU_B']]) for d, r in R]) - zu
    dual = np.array([np.mean([r['dpa_acc_' + l] * r['gng_acc_' + l] for l in 'AB']) for d, r in R])
    ds = np.convolve(dual, np.ones(5) / 5, mode='same'); i = int(np.argmax(ds[2:-2])) + 2
    ax.plot(du, dual, '-', color=cols[s], lw=1.0); ax.plot(du[i], dual[i], 'o', mfc='white', mec=cols[s], ms=6, mew=1.3, zorder=5)
ax.axvline(0, color='0.5', lw=0.8, ls=':'); ax.set_xlim(-2.2, 2.2); ax.set_ylim(0.2, 1.02)
ax.set_xlabel('well at the cue − trained well, $\\kappa_1/\\eta$'); ax.set_ylabel('dual performance (DPA × Go/NoGo)')
ax.set_title("c  each network's optimum vs its trained well", loc='left', fontsize=TITLE_FS)
if L:
    ax.text(0.98, 0.97, f'learners: optimum {db.mean():+.2f} η from the trained well\n(range {db.min():+.2f} … {db.max():+.2f}; n = {len(L)})', transform=ax.transAxes, ha='right', va='top', fontsize=STAT_FS, color='0.3')
fig.suptitle(f'Fig. 5f–h comparison over {len(SEEDS)} networks of the log recipe ({len(LEARN)} learners)', x=0.06, ha='left', fontsize=TITLE_FS * 1.05, fontweight='bold')
fig.savefig(f'{M}/diag_gng_vs_well_log16.png', dpi=200, bbox_inches='tight'); print('saved', f'{M}/diag_gng_vs_well_log16.png')

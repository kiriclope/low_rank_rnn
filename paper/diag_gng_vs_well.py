"""diag_gng_vs_well.py — why the Go/NoGo curve of Fig. 5g falls for deep wells without a plateau (Leon 2026-09-30:
"I did not expect it to drop that much for lower wells but to plateau and then drop, maybe some simulations are not great").
Reads perturb_depth.json (the drive runs behind Fig. 5f–h and Fig. 6h; sweep_lif_log, 8 networks × 81 drives).
a, Go (solid) and NoGo (dashed) accuracy per network against the well the cue finds; dots, the trained position.
b, Go/NoGo accuracy pooled as Fig. 5g draws it (all eight networks, 0.5 η bins) and for the six networks that learned all
   three stages (0.25 η bins); mean ± 95% bootstrap CI.
c, dual performance (DPA × Go/NoGo accuracy) per network against the well at the cue, relative to its trained position;
   open circle, each network's optimum.
Output: /home/leon/dual/figures/paper_share/modelling/diag_gng_vs_well.png"""
import sys, json, numpy as np
sys.path.insert(0, '/home/leon/rnn/paper')
from style import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
PJ = json.load(open('/home/leon/dual/figures/paper_share/modelling/perturb_depth.json'))
P = {}
for k, v in PJ.items():
    s, var, d = k.split('|')
    if var == 'drive': P.setdefault(int(s), {})[float(d)] = v
LEARN, NOT = [1, 2, 3, 4, 5, 6], [0, 7]
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9)); fig.subplots_adjust(wspace=0.36, left=0.06, right=0.99, top=0.84, bottom=0.17)
ax = axs[0]
for s in LEARN + NOT:
    rows = sorted(P[s].items()); du = [np.mean([r['depthU_A'], r['depthU_B']]) for d, r in rows]
    col = SEED_COL[s] if s in LEARN else '0.6'
    ax.plot(du, [r['go'] for d, r in rows], '-', color=col, lw=1.1, alpha=0.9)
    ax.plot(du, [r['nogo'] for d, r in rows], '--', color=col, lw=1.1, alpha=0.9)
    z = P[s][0.0]; ax.plot(np.mean([z['depthU_A'], z['depthU_B']]), 0.5, 'o', color=col, ms=4)
ax.axvline(0, color=LICK_COL, lw=0.8, ls='--'); ax.set_xlim(-3.6, 1.6); ax.set_ylim(-0.03, 1.03)
ax.set_xlabel('well at the cue, $\\kappa_1/\\eta$'); ax.set_ylabel('accuracy')
ax.set_title('a  Go (solid), NoGo (dashed), per network', loc='left', fontsize=TITLE_FS)
ax.text(0.97, 0.55, 'color: the six learners\ngray: s0, s7 (never learned DPA)\ndots: trained wells', transform=ax.transAxes, ha='right', fontsize=STAT_FS, color='0.3')
ax = axs[1]
def pooled(seeds, bw):
    X = np.array([r['depthU_' + l] for s in seeds for d, r in P[s].items() for l in 'AB'])
    Y = np.array([r['gng_acc_' + l] for s in seeds for d, r in P[s].items() for l in 'AB'])
    xc, mu, lo, hi = [], [], [], []
    for a in np.arange(np.floor(X.min() / bw) * bw, X.max(), bw):
        sel = (X >= a) & (X < a + bw)
        if sel.sum() < 4: continue
        bt = [Y[sel][rng.integers(0, sel.sum(), sel.sum())].mean() for _ in range(1000)]
        xc.append(a + bw / 2); mu.append(Y[sel].mean()); lo.append(np.percentile(bt, 2.5)); hi.append(np.percentile(bt, 97.5))
    return np.array(xc), np.array(mu), np.array(lo), np.array(hi)
for seeds, bw, col, lab in ((range(8), 0.5, '0.45', 'all eight, 0.5 η bins (Fig. 5g now)'), (LEARN, 0.25, BLUE if 'BLUE' in dir() else '#1f77b4', 'six learners, 0.25 η bins')):
    xc, mu, lo, hi = pooled(seeds, bw); ax.fill_between(xc, lo, hi, color=col, alpha=0.2, lw=0); ax.plot(xc, mu, '-o', color=col, ms=3, lw=1.3, label=lab)
trL = np.mean([P[s][0.0]['depthU_' + l] for s in LEARN for l in 'AB'])
ax.axvline(trL, color='0.5', lw=0.8, ls=':'); ax.axvline(0, color=LICK_COL, lw=0.8, ls='--')
ax.set_xlim(-3.6, 1.6); ax.set_ylim(0.45, 1.02); ax.set_xlabel('well at the cue, $\\kappa_1/\\eta$'); ax.set_ylabel('Go/NoGo accuracy')
ax.legend(frameon=False, fontsize=STAT_FS, loc='upper right'); ax.set_title('b  pooled: the plateau appears without s0, s7', loc='left', fontsize=TITLE_FS)
ax.text(trL + 0.05, 0.64, 'trained', fontsize=STAT_FS, color='0.4')
ax = axs[2]
for s in LEARN + NOT:
    rows = sorted(P[s].items()); z = P[s][0.0]; z_du = np.mean([z['depthU_A'], z['depthU_B']])
    du = np.array([np.mean([r['depthU_A'], r['depthU_B']]) for d, r in rows]) - z_du
    dual = np.array([np.mean([r['dpa_acc_' + l] * r['gng_acc_' + l] for l in 'AB']) for d, r in rows])
    ds = np.convolve(dual, np.ones(5) / 5, mode='same'); i = int(np.argmax(ds[2:-2])) + 2
    col = SEED_COL[s] if s in LEARN else '0.6'
    ax.plot(du, dual, '-', color=col, lw=1.1); ax.plot(du[i], dual[i], 'o', mfc='white', mec=col, ms=6, mew=1.3, zorder=5)
ax.axvline(0, color='0.5', lw=0.8, ls=':'); ax.set_xlim(-2.2, 2.2); ax.set_ylim(0.2, 1.02)
ax.set_xlabel('well at the cue − trained well, $\\kappa_1/\\eta$'); ax.set_ylabel('dual performance (DPA × Go/NoGo)')
ax.set_title('c  each network\'s optimum vs its trained well', loc='left', fontsize=TITLE_FS)
ax.text(0.98, 0.97, 'learners: optimum 0.6–0.8 η deeper\n(0.97–1.00 vs 0.88–0.96 trained)', transform=ax.transAxes, ha='right', va='top', fontsize=STAT_FS, color='0.3')
fig.suptitle('Fig. 5g check: the Go/NoGo curve plateaus once the two networks that never learned DPA are left out', x=0.06, ha='left', fontsize=TITLE_FS * 1.05, fontweight='bold')
fig.savefig('/home/leon/dual/figures/paper_share/modelling/diag_gng_vs_well.png', dpi=200, bbox_inches='tight'); print('saved')

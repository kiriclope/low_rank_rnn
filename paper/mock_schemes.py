"""Mock-up of cleaner prediction panels for Fig. 5c: per element, only the predicted memory wells (drawn like the flow's
attractors), one arrow for how the element maps A onto B, the invariant set, and a one-line rule. Positions = the
constructed DPA-stage networks drawn below them in Fig. 5c. Run: python mock_schemes.py"""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/leon/rnn/paper')
from style import *
from matplotlib.patches import FancyArrowPatch
PLUM = '#AA4499'; VCOL = '0.25'; WELL = '#F39C12'
P = {'σ₁': ((0.84, -0.24), (-0.84, -0.24)), 'σ₂': ((0.84, 0.22), (-0.84, -0.22)), 'σ₃': ((0.93, 0.0), (-0.83, 0.0)), 'V': ((0.89, 0.0), (-0.89, 0.0))}
COL = {'σ₁': B_COL, 'σ₂': PLUM, 'σ₃': LICK_COL, 'V': VCOL}
TIT = {'σ₁': 'σ₁  A↔B and C↔D', 'σ₂': 'σ₂  A↔B', 'σ₃': 'σ₃  C↔D', 'V': 'V  all three'}
RULE = {'σ₁': 'B mirrors A;\none height (free)', 'σ₂': 'B = −A;\nthe angle is free', 'σ₃': 'A and B on the line;\nunrelated distances', 'V': 'B mirrors A,\non the line'}
fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.9)); fig.subplots_adjust(wspace=0.12, left=0.02, right=0.98, top=0.86, bottom=0.2)
for ax, nm in zip(axs, ('σ₁', 'σ₂', 'σ₃', 'V')):
    col = COL[nm]; (A, B) = P[nm]
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.0); sp.set_visible(True)
    ax.axhline(0, color=LICK_COL, lw=0.8, ls='--', zorder=1); ax.axvline(0, color='0.9', lw=0.6, zorder=0)
    # the invariant set: where the flow must be tangent
    if nm in ('σ₁', 'V'): ax.axvspan(-0.045, 0.045, color=col, alpha=0.35, lw=0, zorder=1)
    if nm in ('σ₃', 'V'): ax.axhspan(-0.045, 0.045, color=col, alpha=0.35, lw=0, zorder=1)
    if nm == 'σ₂': ax.plot(0, 0, 'o', ms=7, mfc=col, mec='none', alpha=0.6, zorder=2)
    # the map from A to B
    if nm != 'σ₃':
        rad = {'σ₁': -0.35, 'σ₂': 0.0, 'V': -0.35}[nm]
        ax.add_patch(FancyArrowPatch(A, B, arrowstyle='<->', mutation_scale=9, color=col, lw=1.2, shrinkA=9, shrinkB=9, connectionstyle=f'arc3,rad={rad}', zorder=3))
    else:   # σ₃ maps each well onto itself: nothing ties the two distances from the origin
        for p_, lab in ((A, 'a'), (B, 'a′')):
            ax.add_patch(FancyArrowPatch((0, -0.42), (p_[0], -0.42), arrowstyle='<->', mutation_scale=7, color=col, lw=0.9, shrinkA=0, shrinkB=0, zorder=3))
            ax.text(p_[0] / 2, -0.52, lab, ha='center', va='top', fontsize=SMALL, color=col)
    for p_, lab, c_ in ((A, 'A', A_COL), (B, 'B', B_COL)):
        ax.plot(*p_, 'o', ms=11, mfc='none', mec=WELL, mew=2.0, zorder=5); ax.plot(*p_, 'o', ms=5.5, mfc=c_, mec='none', zorder=6)
        ax.text(p_[0], p_[1] + (0.3 if p_[1] >= 0 else -0.3), lab, ha='center', va='center', fontsize=SMALL * 1.1, color=c_, fontweight='bold', zorder=7)
    ax.set_title(TIT[nm], loc='left', fontsize=TITLE_FS, color=col)
    ax.text(0.5, -0.05, RULE[nm], transform=ax.transAxes, ha='center', va='top', fontsize=SMALL, color=col)
axs[0].text(1.42, 0.08, 'lick', fontsize=SMALL * 0.85, color=LICK_COL, ha='right', va='bottom')
fig.savefig('/home/leon/.claude/jobs/1758dd25/tmp/mock_schemes.png', dpi=250); print('saved')

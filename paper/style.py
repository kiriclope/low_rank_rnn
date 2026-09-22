"""House style of the mPFC dual-task paper (copied from /home/leon/dual/CLAUDE.md, overlaps/main_panels.py):
Nature-Neuroscience print typography, lowercase bold panel letters, per-seed colours (the model's analog of the
per-mouse tab10 colours), sample A = #332288 indigo, sample B = #44AA99 teal. Import BEFORE building axes."""
import seaborn as sns, matplotlib.pyplot as plt, numpy as np
PS = 1.30                          # print scale for a ~9.5 in canvas at 183 mm (canvas_in / 7.2 * 0.72 ≈ 0.95 → raised for legibility)
sns.set_context('notebook'); sns.set_style('ticks')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 400,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': PS*8, 'axes.titlesize': PS*8, 'xtick.labelsize': PS*7, 'ytick.labelsize': PS*7,
    'legend.fontsize': PS*6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
    'axes.linewidth': 0.7, 'lines.linewidth': 1.3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'mathtext.fontset': 'dejavusans',
})
TITLE_FS = PS*8; SMALL = PS*6.5; STAT_FS = PS*6.5
A_COL, B_COL = '#332288', '#44AA99'          # sample A / B (the paper's odor colours)
LICK_COL = '#C44E52'                          # the lick line / lick region accent (bright palette red)
SEED_COL = sns.color_palette('tab10', n_colors=10)   # per-seed colour, the analog of the per-mouse colour
_bright = sns.color_palette('bright')
COND_COL = {'DPA': _bright[3], 'Go': _bright[0], 'NoGo': _bright[2], 'Dual': _bright[1]}
TIE_LAB = {'free': 'free', 'pair': 'σ₁', 'inv': 'σ₂', 'test': 'σ₃', 'klein': 'V'}
def panel_letter(fig, ax, L, x=None, dy=0.012):
    p = ax.get_position(); fig.text(p.x0 - 0.035 if x is None else x, p.y1 + dy, L.lower(), fontsize=PS*10, fontweight='bold', va='top', ha='left')
def save(fig, stem, outdir='/home/leon/dual/figures/paper_share/modelling'):
    import os; os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'svg'): fig.savefig(f'{outdir}/{stem}.{ext}', bbox_inches='tight')
    print('saved', f'{outdir}/{stem}.png/.svg')

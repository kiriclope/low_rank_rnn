"""Render the Extended Data figures of the modelling section in the house style by running the source note's figure
scripts under the paper rcParams (Arial, NN sizes, thin rules), with seaborn's context calls neutralized and leading
panel letters ("A  …") lowercased and bolded, then composing the ED images. Run in a screen:
  LD_PRELOAD=... python render_ed.py            (all)   |   python render_ed.py ed14   (one)"""
import sys, os, runpy, subprocess, re, matplotlib; matplotlib.use('Agg')
sys.path.insert(0, '/home/leon/rnn/paper'); import style; from style import PS
import seaborn as sns, matplotlib.pyplot as plt, matplotlib.axes, matplotlib.figure
sns.set_context = lambda *a, **k: None; sns.set_style = lambda *a, **k: None
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'svg.fonttype': 'none', 'axes.linewidth': 0.7})
_axtext, _figtext = matplotlib.axes.Axes.text, matplotlib.figure.Figure.text
def _fix(s, kw):
    m = re.match(r'^([A-H])  (.*)', s if isinstance(s, str) else '')
    if m and kw.get('fontweight') == 'bold': return m.group(1).lower() + '  ' + m.group(2)
    return s
matplotlib.axes.Axes.text = lambda self, x, y, s, *a, **k: _axtext(self, x, y, _fix(s, k), *a, **k)
matplotlib.figure.Figure.text = lambda self, x, y, s, *a, **k: _figtext(self, x, y, _fix(s, k), *a, **k)
OUT = '/home/leon/dual/figures/paper_share/modelling'; TMP = '/home/leon/.claude/jobs/ec0810d6/tmp/ed'; os.makedirs(TMP, exist_ok=True); os.chdir('/home/leon/rnn')
S = '/home/leon/rnn/scratchpad'
JOBS = {  # ed number -> list of (script, argv, env) ; several parts are stacked vertically
 'ed11': [(f'{S}/theory_fig.py', [f'{TMP}/theory.png', f'{TMP}/theory_pred.png'], {})],
 'ed12': [],   # theory_pred.png from ed11
 'ed13': [(f'{S}/dpa_summary_fig.py', [f'{TMP}/sim_cd.png'], {'ROWS': 'dpa,mn', 'NOFREE': '1'}), (f'{S}/group_predictions_fig.py', [f'{TMP}/group_predictions.png'], {})],
 'ed14': [(f'{S}/gng_theory_fig.py', [f'{TMP}/gng_theory.png', f'{TMP}/gng_theory_supp.png'], {}), (f'{S}/gng_sim_fig.py', [f'{TMP}/gng_sim.png'], {})],
 'ed15': [(f'{S}/dual_theory_fig.py', [f'{TMP}/dual_theory.png', f'{TMP}/dual_theory_supp.png'], {}), (f'{S}/dpa_summary_fig.py', [f'{TMP}/released.png'], {'ROWS': 'expert'})],
 'ed16': [(f'{S}/rulesym_summary_fig.py', ['results/dual/sweep_lif_rulesym', f'{TMP}/rulesym.png'], {}), (f'{S}/afc2_summary_fig.py', ['results/dual/sweep_lif_rulesym', f'{TMP}/afc2.png'], {})],
}
PARTS = {'ed11': ['theory.png'], 'ed12': ['theory_pred.png'], 'ed13': ['sim_cd.png', 'group_predictions.png'], 'ed14': ['gng_theory.png', 'gng_sim.png'], 'ed15': ['dual_theory.png', 'released.png'], 'ed16': ['rulesym.png', 'afc2.png']}
want = sys.argv[1:] or list(JOBS)
for ed in want:
    for script, argv, env in JOBS[ed]:
        os.environ.update(env); sys.argv = [script] + argv
        print('running', os.path.basename(script), argv[-1], flush=True); runpy.run_path(script, run_name='__main__'); plt.close('all')
        for k in env: os.environ.pop(k, None)
    parts = [f'{TMP}/{p}' for p in PARTS[ed] if os.path.exists(f'{TMP}/{p}')]
    if parts:
        subprocess.run(['convert', *parts, '-resize', '2200x', '-append', '-quality', '92', f'{OUT}/{ed}.png'], check=True); print('composed', f'{OUT}/{ed}.png', flush=True)
print('ED_DONE')

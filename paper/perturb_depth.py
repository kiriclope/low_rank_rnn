"""Depth ↔ performance by perturbation (the model twin of Fig. 6): move the memory wells along the lick axis of a
trained Dual network WITHOUT touching the test-driven field, by an external drive along m1 applied to the units during
the delay only (the laser window), or — for comparison — by adding a mean to n1 (a parameter change that also moves the
test-driven field). For each perturbation strength: the delay-end state of A and B trials (two well locations per
strength), the DPA choice on paired / unpaired trials (DPA-only and dual trials), and the Go / NoGo responses.
Output: figures/paper_share/modelling/perturb_depth.{png,svg} + perturb_depth.json.  Run: LD_PRELOAD=... python perturb_depth.py"""
import sys, os, json, numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad'); sys.path.insert(0, '/home/leon/rnn/paper')
from style import *
from bifurcation_probe import load_run, run_dt_alpha
from src.tasks import make_timings, generate_dpa_trials, generate_dual_trials
RED, BLUE, GREEN = '#d62728', '#1f77b4', '#2ca02c'
os.chdir('/home/leon/rnn'); SW, ARM = 'results/dual/sweep_lif_recipe7_bfix', 'recipe7'; SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2,3,4,5,6,7').split(',')]
DELTAS = np.round(np.linspace(-0.6, 1.0, int(os.environ.get('NDELTA', 41))), 4); NTR = int(os.environ.get('NTR', 1024)); DELTAS_CTRL = np.round(np.linspace(-1.2, 1.2, 9), 4); DEV = os.environ.get('DEV', 'cuda:1')
def run(model, X, drive=None, window=None):
    """the model's own update ("both" integration) with an extra per-unit drive added to the input drive on the steps in window"""
    X = X.to(DEV); B, T, _ = X.shape; N = model.hidden_size; rec = torch.zeros(B, N, device=DEV); rates = torch.zeros(B, N, device=DEV); out = []
    with torch.no_grad():
        for t in range(T):
            x_t = X[:, t]; inp = model.Ai * model.wi(x_t)
            if drive is not None and window[0] <= t < window[1]: inp = inp + drive
            hidden = (rates @ model.n) @ (model.rec_scale * model.m).T / N
            rec = model.exp_alpha_rec * rec + (1.0 - model.exp_alpha_rec) * hidden
            phi = model.nonlinearity(model.gain * (inp + rec) + model.unit_bias)
            rates = model.exp_alpha * rates + (1.0 - model.exp_alpha) * phi
            out.append(rates @ model.n / N)
    return torch.stack(out, 1).cpu().numpy()
res = {}
JS = os.environ.get('OUT', '/home/leon/dual/figures/paper_share/modelling/perturb_depth.json')
if os.environ.get('PLOT_ONLY'):
    for k, v in json.load(open(JS)).items():
        s_, var, d = k.split('|'); res[(int(s_), var, float(d))] = v
    SEEDS = []
for s in SEEDS:
    m, cfg = load_run(SW, f's{s}_{ARM}', stage='expert', device=DEV); dt, alpha, _ = run_dt_alpha(cfg); eta = cfg['noise'] * float(np.sqrt(1 - np.exp(-alpha) ** 2)); half = int(round(0.5 / dt))
    Tp, Td = make_timings(dt)['dpa'], make_timings(dt)['dual']; torch.manual_seed(s)
    Xd, _ = generate_dpa_trials(NTR, Tp, cfg['input_size'], noise=eta, target_rank=2, windowed_targets=True, decay_to_zero=False, response_in_cue=cfg['response_in_cue'], prelick_free=True, hold_window=0.5, hold_anchor='sample')
    Xu, _, _, names = generate_dual_trials(NTR, Td, cfg['input_size'], noise=eta, target_rank=2, cue_on_go_input=cfg['cue_on_go_input'], cue_scale=cfg['cue_scale'], nogo_target=cfg['nogo_target'], go_target=cfg['go_target'], response_in_cue=cfg['response_in_cue'], windowed_targets=cfg['windowed_targets'], decay_to_zero=cfg['decay_to_zero'], gng_response=cfg['gng_response'])
    sA = Xd[:, int(Tp.n_stim_on[0]):int(Tp.n_stim_off[0]), 0].mean(1).numpy() > 0.5; tC = Xd[:, int(Tp.n_stim_on[1]):int(Tp.n_stim_off[1]), 2].mean(1).numpy() > 0.5; pair_d = (sA & tC) | (~sA & ~tC)
    samp = np.array([n[0] for n in names]); tst = np.array([n[-1] for n in names]); pair_u = ((samp == 'A') & (tst == 'C')) | ((samp == 'B') & (tst == 'D')); go = np.array(['_go_' in n for n in names]); ng = np.array(['_nogo_' in n for n in names])
    m1 = m.m.detach()[:, 1]; drive_dir = (m1 / m1.norm() * float(np.sqrt(m.hidden_size))).to(DEV)   # unit-rms pattern along m1
    DUALWIN = os.environ.get('DUALWIN', 'cue')   # where the drive stops on dual trials: 'cue' = at cue onset (the response stimulus, as the test is for DPA), 'odor' = at the Go/NoGo odor onset
    win_d = (int(Tp.n_stim_off[0]), int(Tp.n_stim_on[1])); win_u = (int(Td.n_stim_off[0]), int(Td.n_stim_on[2] if DUALWIN == 'cue' else Td.n_stim_on[1]))
    for variant in ('drive', 'nmean'):
        for d in (DELTAS if variant == 'drive' else DELTAS_CTRL):
            if variant == 'drive': kd = run(m, Xd, d * drive_dir, win_d); ku = run(m, Xu, d * drive_dir, win_u)
            else:
                n_backup = m.n.detach().clone()
                with torch.no_grad(): m.n[:, 1] += float(d) * 0.5
                kd = run(m, Xd); ku = run(m, Xu)
                with torch.no_grad(): m.n.copy_(n_backup)
            ric = cfg['response_in_cue']; tod = int(Tp.n_stim_off[1]); tou, co = int(Td.n_stim_off[3]), int(Td.n_stim_off[2])
            depth_A = kd[sA, int(Tp.n_stim_on[1]) - 1, 1].mean() / eta; depth_B = kd[~sA, int(Tp.n_stim_on[1]) - 1, 1].mean() / eta
            LICK = os.environ.get('LICK', 'any')   # 'any': κ₁ > 0 at any step of the response window (a lick event); 'mean': window mean > 0 (the sweep's criterion)
            L = (lambda seg: (seg > 0).any(1)) if LICK == 'any' else (lambda seg: seg.mean(1) > 0)
            lick_d = L(kd[:, tod - half:tod, 1] if ric else kd[:, tod:tod + half, 1]); lick_u = L(ku[:, tou - half:tou, 1] if ric else ku[:, tou:tou + half, 1]); cue = L(ku[:, co - half:co, 1] if ric else ku[:, co:co + half, 1])
            res[(s, variant, float(f'{d:.3f}'))] = dict(depth_A=float(depth_A), depth_B=float(depth_B), dpa_hit=float(lick_d[pair_d].mean()), dpa_fa=float(lick_d[~pair_d].mean()), dpa_acc=float(((lick_d == pair_d).mean())),
                                                dpa_acc_A=float((lick_d == pair_d)[sA].mean()), dpa_acc_B=float((lick_d == pair_d)[~sA].mean()), dual_acc=float((lick_u == pair_u).mean()), go=float(cue[go].mean()), nogo=float(1 - cue[ng].mean()),
                                                depthU_A=float(ku[samp == 'A', win_u[1] - 1, 1].mean() / eta), depthU_B=float(ku[samp == 'B', win_u[1] - 1, 1].mean() / eta),
                                                gng_acc_A=float(np.mean(np.r_[cue[go & (samp == 'A')], ~cue[ng & (samp == 'A')]])), gng_acc_B=float(np.mean(np.r_[cue[go & (samp == 'B')], ~cue[ng & (samp == 'B')]])),
                                                dual_acc_A=float((lick_u == pair_u)[samp == 'A'].mean()), dual_acc_B=float((lick_u == pair_u)[samp == 'B'].mean()))
        print(f'seed {s} {variant} done', flush=True)
if not os.environ.get('PLOT_ONLY'): json.dump({f'{k[0]}|{k[1]}|{k[2]:.3f}': v for k, v in res.items()}, open(JS, 'w'), indent=0)
if os.environ.get('OUT'): print('PERTURB_DONE'); sys.exit(0)   # a partial run: no figure (merge the parts, then PLOT_ONLY=1)
SEEDS = list(range(8))
# ── figure: accuracy vs well location, one curve per network (A filled, B open); row 1 the delay drive, row 2 the n1-mean control ──
fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2)); fig.subplots_adjust(hspace=0.75, wspace=0.4, left=0.07, right=0.985, top=0.9, bottom=0.09)
for r, variant in enumerate(('drive', 'nmean')):
    axs = axes[r]; DL = DELTAS if variant == 'drive' else DELTAS_CTRL
    for s in SEEDS:
        rows = [res[(s, variant, float(f'{d:.3f}'))] for d in DL]
        for lab in ('A', 'B'):
            axs[0].plot([rw['depth_' + lab] for rw in rows], [rw['dpa_acc_' + lab] for rw in rows], '-o', color=SEED_COL[s], ms=3, lw=0.8, alpha=0.8, mfc=(SEED_COL[s] if lab == 'A' else 'white'))
        depm = [(rw['depth_A'] + rw['depth_B']) / 2 for rw in rows]; depu = [(rw['depthU_A'] + rw['depthU_B']) / 2 for rw in rows]
        axs[1].plot(depm, [rw['dpa_hit'] for rw in rows], '-s', color=SEED_COL[s], ms=3, lw=0.8, alpha=0.6, mfc='white'); axs[1].plot(depm, [rw['dpa_fa'] for rw in rows], '-o', color=SEED_COL[s], ms=3, lw=0.8, alpha=0.8)
        axs[2].plot(depu, [rw['go'] for rw in rows], '-s', color=BLUE, ms=3, lw=0.8, alpha=0.6, mfc='white'); axs[2].plot(depu, [rw['nogo'] for rw in rows], '-o', color=GREEN, ms=3, lw=0.8, alpha=0.6)
    for ax in axs: ax.axvline(0, color=LICK_COL, lw=0.7, ls='--'); ax.set_ylim(-0.03, 1.03)
    axs[0].set_xlabel('well location when the test arrives, $\\kappa_1/\\eta$'); axs[1].set_xlabel('well location when the test arrives, $\\kappa_1/\\eta$'); axs[2].set_xlabel('well location when the cue arrives, $\\kappa_1/\\eta$')
    axs[0].set_ylabel('DPA accuracy'); axs[1].set_ylabel('P(lick at the test)'); axs[2].set_ylabel('Go / NoGo accuracy')
    axs[0].set_title('DPA accuracy per sample (A filled, B open)', loc='left', fontsize=TITLE_FS); axs[1].set_title('hits (open squares) and false alarms (filled)', loc='left', fontsize=TITLE_FS); axs[2].set_title('Go (blue, open) and NoGo (green, filled)', loc='left', fontsize=TITLE_FS)
    fig.text(0.07, 0.955 - r * 0.485, ('drive along $m_1$ from sample offset to the response stimulus (test / cue); lick = $\\kappa_1 > 0$ at any step of the response window' if variant == 'drive' else 'control: a mean added to $n_1$ moves the field and the readout together with the wells'), fontsize=PS*8, fontweight='bold', va='bottom')
for L, ax in zip('abcdef', axes.flat): panel_letter(fig, ax, L)
save(fig, 'perturb_depth'); print('PERTURB_DONE')

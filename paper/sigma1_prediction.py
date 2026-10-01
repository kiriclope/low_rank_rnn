"""The model's sigma_1 prediction for Fig. 6: a delay input that does not depend on the sample moves the A and B
memory states by the same amount along the choice axis; an input that favours one sample does not.

For each network that learned the task (sweep_lif_log s1-s6) at the naive and expert checkpoints: DPA trials at the
trained noise, a constant per-unit input added from sample offset to test onset (the laser window, as in
perturb_depth.py), and the change of the late-delay state (the last 1.5 s before the test, the data's bins 45-53)
on A and B trials, in units of eta. Inputs:
  sample-independent : the Fig. 5 drive along m1; a uniform input to every unit; random patterns with a random mean
                       (unit r.m.s., no relation to the sample)
  sample-specific    : a pattern along m0, the sample write vector (A-preferring units up, B-preferring down)
Output: /home/leon/dual/figures/paper_share/modelling/sigma1_prediction.json
Run:  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python sigma1_prediction.py
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad'); sys.path.insert(0, '/home/leon/rnn/paper')
from bifurcation_probe import load_run, run_dt_alpha
from src.tasks import make_timings, generate_dpa_trials

os.chdir('/home/leon/rnn')
SW, ARM = 'results/dual/sweep_lif_log', 'log'
SEEDS = [int(x) for x in os.environ.get('SEEDS', '1,2,3,4,5,6').split(',')]
DEV = os.environ.get('DEV', 'cuda:0'); NTR = int(os.environ.get('NTR', 1024)); NRAND = int(os.environ.get('NRAND', 8))
DELTAS = [-0.3, -0.15, 0.15, 0.3]
OUT = os.environ.get('OUT', '/home/leon/dual/figures/paper_share/modelling/sigma1_prediction.json')   # OUT: a separate file for extra seeds


def run(model, X, drive=None, window=None):
    """perturb_depth.run: the model's own update with an extra per-unit input on the steps in window; returns kappa(t)"""
    X = X.to(DEV); B, T, _ = X.shape; N = model.hidden_size
    rec = torch.zeros(B, N, device=DEV); rates = torch.zeros(B, N, device=DEV); out = []
    with torch.no_grad():
        for t in range(T):
            inp = model.Ai * model.wi(X[:, t])
            if drive is not None and window[0] <= t < window[1]:
                inp = inp + drive
            hidden = (rates @ model.n) @ (model.rec_scale * model.m).T / N
            rec = model.exp_alpha_rec * rec + (1.0 - model.exp_alpha_rec) * hidden
            rates = model.exp_alpha * rates + (1.0 - model.exp_alpha) * model.nonlinearity(model.gain * (inp + rec) + model.unit_bias)
            out.append(rates @ model.n / N)
    return torch.stack(out, 1).cpu().numpy()


def unit_rms(v):
    v = v.float()
    return v / v.pow(2).mean().sqrt()


res = []
for s in SEEDS:
    for stage in ('naive', 'expert'):
        m, cfg = load_run(SW, f's{s}_{ARM}', stage=stage, device=DEV)
        dt, alpha, _ = run_dt_alpha(cfg); eta = cfg['noise'] * float(np.sqrt(1 - np.exp(-alpha) ** 2))
        Tp = make_timings(dt)['dpa']; torch.manual_seed(100 + s)
        X, _ = generate_dpa_trials(NTR, Tp, cfg['input_size'], noise=eta, target_rank=2, windowed_targets=True, decay_to_zero=False,
                                   response_in_cue=cfg['response_in_cue'], prelick_free=True, hold_window=0.5, hold_anchor='sample')
        sA = X[:, int(Tp.n_stim_on[0]):int(Tp.n_stim_off[0]), 0].mean(1).numpy() > 0.5
        t_test = int(Tp.n_stim_on[1]); ld = slice(t_test - int(round(1.5 / dt)), t_test); win = (int(Tp.n_stim_off[0]), t_test)
        N = m.hidden_size
        k0 = run(m, X)
        base = {lab: k0[msk, ld].mean((0, 1)) for lab, msk in (('A', sA), ('B', ~sA))}
        g = torch.Generator().manual_seed(1000 + s)
        pats = [('m1', 'independent', unit_rms(m.m.detach()[:, 1].cpu())), ('uniform', 'independent', torch.ones(N))]
        for r in range(NRAND):
            xi = torch.randn(N, generator=g); mu = float(torch.randn(1, generator=g))
            pats.append((f'random{r}', 'independent', unit_rms(xi + mu)))
        pats.append(('m0', 'specific', unit_rms(m.m.detach()[:, 0].cpu())))
        for name, kind, p in pats:
            p = p.to(DEV)
            for d in DELTAS:
                k = run(m, X, d * p, win)
                dA = (k[sA, ld].mean((0, 1)) - base['A']) / eta; dB = (k[~sA, ld].mean((0, 1)) - base['B']) / eta
                res.append(dict(seed=s, stage=stage, pattern=name, kind=kind, delta=d, dA1=float(dA[1]), dB1=float(dB[1]),
                                dA0=float(dA[0]), dB0=float(dB[0])))
        print(f'seed {s} {stage}: {len(pats)} patterns done', flush=True)
json.dump(res, open(OUT, 'w'), indent=0)
R = res
for kind in ('independent', 'specific'):
    a = np.array([r['dA1'] for r in R if r['kind'] == kind]); b = np.array([r['dB1'] for r in R if r['kind'] == kind])
    print(f'{kind:12s}: choice axis corr(A,B) {np.corrcoef(a, b)[0, 1]:+.3f}, median |A-B|/(|A|+|B|) '
          f'{np.median(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-9)):.2f}, n = {len(a)}')
print('saved', OUT)

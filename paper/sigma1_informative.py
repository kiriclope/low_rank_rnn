"""sigma1_informative.py — does a delay input that CARRIES sample information still move the A and B memory states
together along the choice axis? (Leon 2026-09-30, on Fig. 6k–m: "we are considering the laser as an input that has no
sample info, but ACC has sample information, so removing ACC to mPFC terminals could be sample selective".)

σ₁ constrains how a manipulation that treats the two samples alike acts, not whether the input carries sample
information. Silencing a projection removes, on an A trial, what it sends on A trials and, on a B trial, what it sends on
B trials; if those are mirror images (A and B carried in equal measure), the two removals are σ₁ images of each other and
the choice-axis change is the same on A and B trials. Only an input that favors one sample separates them.

Same networks, trials, window and read-out as sigma1_prediction.py (sweep_lif_log s1–s6, naïve and expert checkpoints,
DPA trials, input from sample offset to test onset, late-delay state over the last 1.5 s before the test, units of η).
Input classes (unit-r.m.s. pattern m̂₀ = the sample write vector; δ < 0 = removing the input, as the laser does):
  info_sym   : +δ·m̂₀ on A trials, −δ·m̂₀ on B trials — each trial's own sample, both samples alike (ACC-like copy)
  info_mixed : a random common pattern plus the same sample-informative part (ξ ± m̂₀), both samples alike
  info_asym  : +δ·m̂₀ on A trials, −½δ·m̂₀ on B trials — sample information carried more strongly for A
  bias_m0    : +δ·m̂₀ on every trial — favors sample A (the "sample-specific" input of Fig. 6l)
Output: /home/leon/dual/figures/paper_share/modelling/sigma1_informative.json
Run:  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python sigma1_informative.py
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad'); sys.path.insert(0, '/home/leon/rnn/paper')
from bifurcation_probe import load_run, run_dt_alpha
from src.tasks import make_timings, generate_dpa_trials

os.chdir('/home/leon/rnn')
SW, ARM = 'results/dual/sweep_lif_log', 'log'
SEEDS = [int(x) for x in os.environ.get('SEEDS', '1,2,3,4,5,6').split(',')]
DEV = os.environ.get('DEV', 'cuda:0'); NTR = int(os.environ.get('NTR', 1024)); NRAND = int(os.environ.get('NRAND', 4))
DELTAS = [-0.3, -0.15, 0.15, 0.3]
OUT = os.environ.get('OUT', '/home/leon/dual/figures/paper_share/modelling/sigma1_informative.json')


def run(model, X, drive=None, window=None):
    """sigma1_prediction.run; `drive` may be (N,) or (B, N) (a trial-dependent input)"""
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
        m0 = unit_rms(m.m.detach()[:, 0].cpu()).to(DEV)
        sgn = torch.tensor(np.where(sA, 1.0, -1.0), dtype=torch.float32, device=DEV)[:, None]       # +1 on A trials, −1 on B
        asym = torch.tensor(np.where(sA, 1.0, -0.5), dtype=torch.float32, device=DEV)[:, None]
        g = torch.Generator().manual_seed(2000 + s)
        classes = [('info_sym', 'informative, alike', lambda d: d * sgn * m0[None, :]),
                   ('info_asym', 'informative, A-biased', lambda d: d * asym * m0[None, :]),
                   ('bias_m0', 'favors A', lambda d: d * m0)]
        for r in range(NRAND):
            xi = unit_rms(torch.randn(N, generator=g) + float(torch.randn(1, generator=g))).to(DEV)
            classes.append((f'info_mixed{r}', 'informative, alike', (lambda xi_: (lambda d: d * (xi_[None, :] + sgn * m0[None, :]) / np.sqrt(2)))(xi)))
        for name, kind, fn in classes:
            for d in DELTAS:
                k = run(m, X, fn(d), win)
                dA = (k[sA, ld].mean((0, 1)) - base['A']) / eta; dB = (k[~sA, ld].mean((0, 1)) - base['B']) / eta
                res.append(dict(seed=s, stage=stage, pattern=name, kind=kind, delta=d, dA1=float(dA[1]), dB1=float(dB[1]),
                                dA0=float(dA[0]), dB0=float(dB[0])))
        print(f'seed {s} {stage}: {len(classes)} input classes done', flush=True)
json.dump(res, open(OUT, 'w'), indent=0)
print('\nchoice axis (κ₁) and sample axis (κ₀), change of the late-delay state on A against B trials, in η:')
for kind in ('informative, alike', 'informative, A-biased', 'favors A'):
    R = [r for r in res if r['kind'] == kind]
    a1 = np.array([r['dA1'] for r in R]); b1 = np.array([r['dB1'] for r in R]); a0 = np.array([r['dA0'] for r in R]); b0 = np.array([r['dB0'] for r in R])
    print(f'  {kind:22s} n={len(R):3d} | κ₁: corr(A,B) {np.corrcoef(a1, b1)[0, 1]:+.2f}, mean |A−B| {np.mean(np.abs(a1 - b1)):.3f}, '
          f'mean |A+B|/2 {np.mean(np.abs(a1 + b1) / 2):.3f} | κ₀: corr(A,B) {np.corrcoef(a0, b0)[0, 1]:+.2f} '
          f'(−1 = mirror: memory strength; +1 = both shifted toward one sample)')
print('saved', OUT)

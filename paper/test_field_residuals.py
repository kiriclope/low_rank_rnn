"""The group's predictions for the TEST-driven choice states, scored per network: κ at the end of the test window for the four
trial types (A,C) (A,D) (B,C) (B,D) at the trained noise, and the three relations the elements impose on them:
  r1 = |D1 κ(A,C) − κ(B,D)|   (σ1: A↔B and C↔D, response kept)      r2 = |κ(A,C) + κ(B,C)|   (σ2: A↔B alone; C fixed ⇒ C field odd)
  r3 = |D3 κ(A,C) − κ(A,D)|   (σ3: C↔D alone, response flipped)
each divided by the mean |κ| of the four states. Usage: test_field_residuals.py <sweep>:<rid> ... [STAGES env, default dpa,expert]"""
import sys, os, json, numpy as np, torch; torch.set_num_threads(int(os.environ.get('NTHR', 8))); sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad')
from symtools import load_run, sig_of
from bifurcation_probe import run_dt_alpha
from src.tasks import make_timings, generate_dpa_trials
D1, D3 = np.diag([-1., 1.]), np.diag([1., -1.])
def four_states(m, cfg, noise=None):
    """κ at the end of the test, averaged over trials at the TRAINED input noise (the networks' operating point; noise-free
    they lose the sample). The (A,D), (B,C), (B,D) trials are the (A,C) trials with their input channels relabeled, noise
    included (swap A↔B = channels 0,1; C↔D = channels 2,3): valid trials of those types with common random numbers, so a
    tied network satisfies the orbit relations trial by trial (Lemma 12.1) and the predicted residuals are exact zeros."""
    dt, _, _ = run_dt_alpha(cfg); T = make_timings(dt)['dpa']; torch.manual_seed(0)
    X, _ = generate_dpa_trials(int(os.environ.get("NTR", 512)), T, cfg['input_size'], noise=sig_of(cfg) if noise is None else noise, target_rank=2,
                               windowed_targets=True, decay_to_zero=False, response_in_cue=cfg['response_in_cue'], prelick_free=True,
                               hold_window=cfg.get('dpa_hold_window', 0.5), hold_anchor=cfg.get('dpa_hold_anchor', 'sample'),
                               post_response_window=cfg.get('dpa_post_response_window'))
    s0, s1, t0, t1 = int(T.n_stim_on[0]), int(T.n_stim_off[0]), int(T.n_stim_on[1]), int(T.n_stim_off[1])
    isA = X[:, s0:s1, 0].mean(1) > 0.5; isC = X[:, t0:t1, 2].mean(1) > 0.5
    XAC = X[isA & isC]
    def swap(Z, i, j): Z = Z.clone(); Z[..., [i, j]] = Z[..., [j, i]]; return Z
    trials = {('A', 'C'): XAC, ('B', 'C'): swap(XAC, 0, 1), ('A', 'D'): swap(XAC, 2, 3), ('B', 'D'): swap(swap(XAC, 0, 1), 2, 3)}
    m.noise = 0.0; out = {}
    with torch.no_grad():
        for key, Z in trials.items(): out[key] = m(Z).numpy()[:, t1 - 1].mean(0)
    return out
if __name__ == '__main__':
    out = {}
    for spec in sys.argv[1:]:
        sw, rid = spec.split(':')
        for st in os.environ.get('STAGES', 'dpa,expert').split(','):
            m, cfg = load_run(sw, rid, stage=st, device='cpu'); S = four_states(m, cfg)
            sc = np.mean([np.linalg.norm(v) for v in S.values()])
            r1 = np.linalg.norm(D1 @ S['A', 'C'] - S['B', 'D']) / sc; r2 = np.linalg.norm(S['A', 'C'] + S['B', 'C']) / sc; r3 = np.linalg.norm(D3 @ S['A', 'C'] - S['A', 'D']) / sc
            out[f'{rid}|{st}'] = dict(r1=float(r1), r2=float(r2), r3=float(r3), states={f'{a}{c}': [float(x) for x in S[a, c]] for a, c in S})
            print(f'{rid:12s} {st:6s} r1 {r1:.3f}  r2 {r2:.3f}  r3 {r3:.3f} | AC ({S["A","C"][0]:+.2f},{S["A","C"][1]:+.2f}) AD ({S["A","D"][0]:+.2f},{S["A","D"][1]:+.2f}) BC ({S["B","C"][0]:+.2f},{S["B","C"][1]:+.2f}) BD ({S["B","D"][0]:+.2f},{S["B","D"][1]:+.2f})', flush=True)
    if os.environ.get('OUT'): json.dump(out, open(os.environ['OUT'], 'w'), indent=0)

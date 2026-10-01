"""Before dual training (the naive checkpoint, after GNG): the lick carried from a Go trial into the test, per learner —
false alarms at the test on unpaired Go trials, the sample's sign kept on κ₀ at test onset on Go trials, NoGo correct at
the cue. Same trial generator and seeds as draft_numbers_log.py section B. Run: LD_PRELOAD=... python naive_carryover16.py"""
import sys, os, json, numpy as np, torch; torch.set_num_threads(8)
sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad')
from symtools import load_run
from bifurcation_probe import run_dt_alpha
from src.tasks import make_timings, generate_dual_trials
os.chdir('/home/leon/rnn'); SW = 'results/dual/sweep_lif_log'
R = {json.loads(l)['run_id']: json.loads(l) for l in open(f'{SW}/results.jsonl')}
LEARN = [f's{s}_log' for s in range(16) if R[f's{s}_log']['accuracy']['after_dpa']['dpa'] >= 0.95]
rows = {}
for rid in LEARN:
    m, cfg = load_run(SW, rid, stage='naive', device='cpu'); dt, alpha, _ = run_dt_alpha(cfg); eta = cfg['noise'] * float(np.sqrt(1 - np.exp(-alpha) ** 2)); torch.manual_seed(1)
    Td = make_timings(dt)['dual']; half = int(round(0.5 / dt))
    X, y, _, names = generate_dual_trials(384, Td, cfg['input_size'], noise=eta, target_rank=2, cue_on_go_input=cfg['cue_on_go_input'], cue_scale=cfg['cue_scale'],
                                          nogo_target=cfg['nogo_target'], go_target=cfg['go_target'], response_in_cue=cfg['response_in_cue'],
                                          gng_response=cfg.get('gng_response', True), gng_memory=cfg.get('dual_gng_memory', False), attention_input=cfg.get('attention_input', False))
    with torch.no_grad(): k = m(X).numpy()
    k0, k1 = k[..., 0], k[..., 1]
    to, co = int(Td.n_stim_off[3]), int(Td.n_stim_off[2]); t_on = int(Td.n_stim_on[3]); ric = cfg['response_in_cue']
    L = lambda seg: (seg > 0).any(1)
    test_lick = L(k1[:, (to - half):to] if ric else k1[:, to:to + half]); cue_lick = L(k1[:, (co - half):co] if ric else k1[:, co:co + half])
    samp = np.array([n[0] for n in names]); tst = np.array([n[-1] for n in names]); pair = ((samp == 'A') & (tst == 'C')) | ((samp == 'B') & (tst == 'D'))
    go = np.array(['_go_' in n for n in names]); ng = np.array(['_nogo_' in n for n in names])
    s0 = k0[:, t_on - 1]; sA = np.sign(s0[samp == 'A'].mean()); kept = np.where(samp == 'A', np.sign(s0) == sA, np.sign(s0) == -sA)
    rows[rid] = dict(fa_unpaired_go=float(test_lick[go & ~pair].mean()), sign_kept_go=float(kept[go].mean()), nogo_cue_correct=float(1 - cue_lick[ng].mean()))
    print(f'{rid}: FA on unpaired Go trials {rows[rid]["fa_unpaired_go"]:.2f}, sample sign kept (Go) {rows[rid]["sign_kept_go"]:.2f}, NoGo correct at the cue {rows[rid]["nogo_cue_correct"]:.2f}', flush=True)
v = lambda k: np.array([r[k] for r in rows.values()])
print(f'learners (n = {len(rows)}): FA {v("fa_unpaired_go").min():.2f} to {v("fa_unpaired_go").max():.2f}, median {np.median(v("fa_unpaired_go")):.2f}; '
      f'sign kept {v("sign_kept_go").min():.2f} to {v("sign_kept_go").max():.2f}; NoGo at the cue median {np.median(v("nogo_cue_correct")):.2f}')
json.dump(rows, open('/home/leon/dual/figures/paper_share/modelling/naive_carryover16.json', 'w'), indent=1)

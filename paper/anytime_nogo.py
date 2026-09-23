import sys, json, numpy as np, torch; sys.path.insert(0, '/home/leon/rnn'); sys.path.insert(0, '/home/leon/rnn/scratchpad')
from bifurcation_probe import load_run, run_dt_alpha
from src.tasks import make_timings, generate_dual_trials
DEV = 'cuda:1'
def run(model, X):
    X = X.to(DEV); B, T, _ = X.shape; N = model.hidden_size; rec = torch.zeros(B, N, device=DEV); rates = torch.zeros(B, N, device=DEV); out = []
    with torch.no_grad():
        for t in range(T):
            inp = model.Ai * model.wi(X[:, t]); hidden = (rates @ model.n) @ (model.rec_scale * model.m).T / N
            rec = model.exp_alpha_rec * rec + (1.0 - model.exp_alpha_rec) * hidden; phi = model.nonlinearity(model.gain * (inp + rec) + model.unit_bias)
            rates = model.exp_alpha * rates + (1.0 - model.exp_alpha) * phi; out.append(rates @ model.n / N)
    return torch.stack(out, 1)[:, :, 1].cpu().numpy()
SETS = [('free (bfix, in-cue trained)', 'results/dual/sweep_lif_recipe7_bfix', 'recipe7', range(8)),
        ('σ₁ tied (bfix)', 'results/dual/sweep_lif_symdpa_bfix', 'symdpa_pair', range(4)), ('V tied (bfix)', 'results/dual/sweep_lif_symdpa_bfix', 'symdpa_klein', range(4)),
        ('postresp free', 'results/dual/sweep_lif_postresp', 'postresp_free', range(4)), ('postresp σ₁', 'results/dual/sweep_lif_postresp', 'postresp_pair', range(4)),
        ('postresp σ₂', 'results/dual/sweep_lif_postresp', 'postresp_inv', range(4)), ('postresp σ₃', 'results/dual/sweep_lif_postresp', 'postresp_test', range(4)), ('postresp V', 'results/dual/sweep_lif_postresp', 'postresp_klein', range(4))]
res = {}
print(f'{"set":28s} seed | NoGo in-cue: mean  any-step  any-100ms | NoGo post-cue: mean  any-step  any-100ms | Go in-cue any-100ms  post any-100ms')
for lab, sw, arm, seeds in SETS:
    for s in seeds:
        m, cfg = load_run(sw, f's{s}_{arm}', stage='expert', device=DEV); dt, alpha, _ = run_dt_alpha(cfg); eta = cfg['noise'] * float(np.sqrt(1 - np.exp(-alpha) ** 2)); half = int(round(0.5 / dt)); k5 = max(1, int(round(0.1 / dt)))
        Td = make_timings(dt)['dual']; torch.manual_seed(s)
        X, _, _, names = generate_dual_trials(1024, Td, cfg['input_size'], noise=eta, target_rank=2, cue_on_go_input=cfg['cue_on_go_input'], cue_scale=cfg['cue_scale'], nogo_target=cfg['nogo_target'], go_target=cfg['go_target'], response_in_cue=cfg['response_in_cue'], windowed_targets=cfg['windowed_targets'], decay_to_zero=cfg['decay_to_zero'], gng_response=cfg['gng_response'])
        k = run(m, X); ng = np.array(['_nogo_' in n for n in names]); go = np.array(['_go_' in n for n in names]); c0, c1 = int(Td.n_stim_on[2]), int(Td.n_stim_off[2])
        ks = np.stack([np.convolve(row, np.ones(k5) / k5, mode='same') for row in k])   # 100-ms running mean
        W = {'cue': (c1 - half, c1), 'post': (c1, c1 + half)}
        r = {}
        for wn, (a, b) in W.items():
            r[f'nogo_mean_{wn}'] = float((k[ng, a:b].mean(1) <= 0).mean()); r[f'nogo_any_{wn}'] = float(((k[ng, a:b] > 0).any(1) == False).mean()); r[f'nogo_any100_{wn}'] = float(((ks[ng, a:b] > 0).any(1) == False).mean())
            r[f'go_any100_{wn}'] = float((ks[go, a:b] > 0).any(1).mean())
        res[f'{arm}|{s}'] = r
        print(f'{lab:28s} s{s}   | {r["nogo_mean_cue"]:.2f}  {r["nogo_any_cue"]:.2f}  {r["nogo_any100_cue"]:.2f}          | {r["nogo_mean_post"]:.2f}  {r["nogo_any_post"]:.2f}  {r["nogo_any100_post"]:.2f}           | {r["go_any100_cue"]:.2f}  {r["go_any100_post"]:.2f}', flush=True)
json.dump(res, open('/home/leon/dual/figures/paper_share/modelling/anytime_nogo.json', 'w'), indent=0)

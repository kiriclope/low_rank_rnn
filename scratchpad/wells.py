"""Measure the held A/B sample-memory location (kappa0, kappa1) on 'none' (no go/nogo) dual trials,
read deep in the delay [7,8]s where only attention is on -> the autonomous memory well. Read-only."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/home/leon/rnn"); os.chdir("/home/leon/rnn")
from plot_sweep import _load_sweep_meta, _build_model, _load_ckpt, TIMINGS
from src.tasks import generate_dual_trials

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
dt = TIMINGS["dual"]
# readout n for kappa: model exposes kappa via forward? We read the readout channels directly.


@torch.no_grad()
def wells(meta, sweep_dir):
    model = _build_model(meta, DEV)
    if not _load_ckpt(model, sweep_dir, "expert", meta.run_id, DEV):
        return None
    X, y, _, names = generate_dual_trials(
        1024, timing=dt, input_size=meta.input_size, noise=meta.noise,
        target_rank=meta.rank, cue_on_go_input=meta.cue_on_go_input,
        cue_scale=meta.cue_scale, nogo_target=meta.nogo_target,
        input_scale=1.0, attention_input=meta.attention_input,
    )
    # kappa = readout projections; model(...) returns predicted readout per channel already = kappa
    out = model(X.to(DEV), y.to(DEV)).cpu()   # [B,T,rank]
    names = np.asarray(names).astype(str)
    is_none = np.array([("_go_" not in n and "_nogo_" not in n) for n in names])
    is_A = np.array([n.startswith("A") for n in names]) & is_none
    is_B = np.array([n.startswith("B") for n in names]) & is_none
    # deep-delay window [7,8]s on 'none' trials: only attention on, no cue/test yet
    w0, w1 = int(7.0/dt.dt), int(8.0/dt.dt)
    k0 = out[:, w0:w1, 0].mean(1)   # memory axis
    k1 = out[:, w0:w1, -1].mean(1)  # decision/lick axis
    return dict(
        A_k0=k0[is_A].mean().item(), A_k1=k1[is_A].mean().item(),
        B_k0=k0[is_B].mean().item(), B_k1=k1[is_B].mean().item(),
    )


for sweep in sys.argv[1:]:
    sd = f"results/dual/{sweep}"
    metas = _load_sweep_meta(sd)
    print(f"\n== {sweep} :: held A/B memory (kappa1<0 = no-lick) on 'none' trials, delay [7,8]s ==")
    k1s = []
    for m in sorted(metas, key=lambda x: int(x.run_id.split('_')[0][1:])):
        r = wells(m, sd)
        if r is None:
            print(f"  {m.run_id}: (no ckpt)"); continue
        both_lo = (r['A_k1'] < 0) and (r['B_k1'] < 0)
        k1s += [r['A_k1'], r['B_k1']]
        print(f"  {m.run_id:9s}  A=({r['A_k0']:+.2f},{r['A_k1']:+.2f})  B=({r['B_k0']:+.2f},{r['B_k1']:+.2f})   both k1<0: {both_lo}")
    if k1s:
        print(f"  --> mean held memory kappa1 = {np.mean(k1s):+.2f}   (negative = no-lick, the goal)")

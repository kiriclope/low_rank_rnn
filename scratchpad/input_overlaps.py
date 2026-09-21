"""Input-to-readout overlaps (1/N) n_j^T w_c for every channel, per checkpoint — the quantity the equivariance
condition P W_in = W_in S_sigma constrains (a sigma_3- or V-equivariant net has n1.w_go = n1.w_nogo = 0 exactly).
Usage: input_overlaps.py <sweep>:<rid>:<stage> ..."""
import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/leon/rnn")
import numpy as np
from bifurcation_probe import load_run
from src.flow_field import low_rank_numpy_params
CH = ["A", "B", "C", "D", "go", "nogo"]   # channel 4 = go (+cue), 5 = nogo (go_on_rwd_input False)
print("run\tstage\t" + "\t".join(f"n0.{c}" for c in CH) + "\t" + "\t".join(f"n1.{c}" for c in CH))
for spec in sys.argv[1:]:
    sw, rid, stage = spec.split(":")
    try: m, cfg = load_run(sw, rid, stage=stage, device="cpu")
    except Exception as e: print(f"{rid}\t{stage}\tSKIP"); continue
    P = low_rank_numpy_params(m); Nv = np.asarray(P["Nvec"]); Wi = np.asarray(P["Wi"]); N = Nv.shape[0]
    ov = (Nv.T @ Wi) / N            # (2, C)
    C = min(len(CH), Wi.shape[1])
    print(f"{rid}\t{stage}\t" + "\t".join(f"{ov[0,c]:+.3f}" for c in range(C)) + "\t" + "\t".join(f"{ov[1,c]:+.3f}" for c in range(C)), flush=True)

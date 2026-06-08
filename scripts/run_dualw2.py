import json, os
from sweep import RunConfig, run_single

out_dir = "/home/leon/rnn/results/dualw2"
os.makedirs(out_dir, exist_ok=True)

cfg = RunConfig(
    run_id="s0_struct_dualw2_cue", seed=0, init_style="structured",
    gain=2.0, cue_on_go_input=True, noise=0.5, model_noise=0.5,
    memory_lambda=0.8, decision_lambda=0.5,
    dual_loss="separated", dpa_weight=2.0, gng_weight=1.0,
    epochs_dpa=100, epochs_gng=100, epochs_dual=100,
    out_dir=out_dir,
)

result = run_single(cfg, "cuda:0")

with open(os.path.join(out_dir, "results.jsonl"), "a") as f:
    f.write(json.dumps(result) + "\n")

print("DONE", result["status"], result["accuracy"]["after_dual"])
print("components:", result["dual_loss_components"])

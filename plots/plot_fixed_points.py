"""
plot_fixed_points.py — fixed-point phase portraits in the κ-plane.

For each of {dpa, naive (post-GNG), expert (post-Dual)} checkpoints of a
single run, generate the appropriate task trials and call
plot_task_flow_fields() from dynamics.py.

Usage
-----
    python plot_fixed_points.py
    python plot_fixed_points.py --run_id s0_struct_ml0.8_dl0.5 \
                                --ckpt_dir /home/leon/results/dual/sweep1 \
                                --out_dir  /home/leon/results/dual/sweep1
    python plot_fixed_points.py --stage expert --task dual
"""

from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.tasks  import TaskTiming, generate_dpa_trials, generate_gng_trials, generate_dual_trials
from src.models import LowRankModel
from src.dynamics import plot_task_flow_fields

sns.set_context("notebook")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)


# ---------------------------------------------------------------------------
# Default config (matches RunConfig defaults in sweep.py)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    hidden_size   = 512,
    rank          = 2,
    gain          = 2.0,
    input_size    = 8,
    tau           = 0.3,
    dt_base       = 0.03,
    tau_rec_frac  = 0.75,
)

STAGE_TASK = {
    "dpa":    "dpa",
    "naive":  "gng",
    "expert": "dual",
}


def _load_run_config(ckpt_dir: str, run_id: str) -> dict | None:
    """
    Return the saved config dict for run_id from <ckpt_dir>/results.jsonl, or None.

    Note: config["input_size"] is already post-RunConfig.__post_init__ (i.e. it has
    been decremented when cue_on_go_input is True), so use it as-is — do NOT subtract
    again.
    """
    import json
    jsonl = os.path.join(ckpt_dir, "results.jsonl")
    if not os.path.exists(jsonl):
        return None
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("run_id") == run_id:
            return row.get("config", {})
    return None


def build_model(device: str, gain: float | None = None, input_size: int = 8) -> LowRankModel:
    DT        = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    alpha     = DT / DEFAULTS["tau"]
    alpha_rec = DT / (DEFAULTS["tau"] * DEFAULTS["tau_rec_frac"])
    return LowRankModel(
        input_size  = input_size,
        hidden_size = DEFAULTS["hidden_size"],
        output_size = 0,
        rank        = DEFAULTS["rank"],
        gain        = gain if gain is not None else DEFAULTS["gain"],
        alpha       = alpha,
        alpha_rec   = alpha_rec,
        noise       = 0.0,
        rwd         = True,
        device      = device,
    )


def make_timing():
    DT = DEFAULTS["dt_base"] * DEFAULTS["tau_rec_frac"]
    return {
        "dpa":  TaskTiming([2.0, 8.0],            [3.0, 9.0],            10.0, DT),
        "gng":  TaskTiming([2.0, 4.0],            [3.0, 5.0],             6.0, DT),
        "dual": TaskTiming([2.0, 4.0, 6.0, 8.0], [3.0, 5.0, 7.0, 9.0], 10.0, DT),
    }


def generate_trials(task: str, timing, n_batch: int = 256,
                    input_size: int = 8, cue_on_go_input: bool = False):
    if task == "dpa":
        inputs, targets = generate_dpa_trials(n_batch, timing, input_size=input_size)
        return inputs, targets, None
    if task == "gng":
        inputs, targets = generate_gng_trials(n_batch, timing, input_size=input_size,
                                               cue_on_go_input=cue_on_go_input)
        return inputs, targets, None
    if task == "dual":
        inputs, targets, _, condition_names = generate_dual_trials(n_batch, timing, input_size=input_size,
                                                                    cue_on_go_input=cue_on_go_input)
        return inputs, targets, condition_names
    raise ValueError(task)


def plot_stage(
    ckpt_path: str,
    stage: str,
    task: str,
    device: str,
    timing_map: dict,
    out_path: str | None = None,
    n_batch: int = 256,
    gain: float | None = None,
    input_size: int = 8,
    cue_on_go_input: bool = False,
):
    model = build_model(device, gain=gain, input_size=input_size)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    timing  = timing_map[task]
    inputs, targets, condition_names = generate_trials(task, timing, n_batch=n_batch,
                                                       input_size=input_size,
                                                       cue_on_go_input=cue_on_go_input)
    inputs  = torch.as_tensor(inputs,  dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32)

    fig, axes, data = plot_task_flow_fields(
        model, inputs, timing, task,
        targets          = targets,
        condition_names  = condition_names,
        n_fp_seeds       = 41,
        cue_on_go_input  = cue_on_go_input,
    )
    fig.suptitle(f"Fixed points — {stage} checkpoint ({task.upper()} task)", y=1.01)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved {out_path}")

    return fig, axes, data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id",   type=str, default="s0_struct_ml0.8_dl0.5")
    parser.add_argument("--ckpt_dir", type=str, default="/home/leon/results/dual/sweep1")
    parser.add_argument("--out_dir",  type=str, default=None)
    parser.add_argument("--stage",    type=str, default=None,
                        help="One of: dpa, naive, expert  (default: all three)")
    parser.add_argument("--task",     type=str, default=None,
                        help="Override task for the chosen stage (dpa/gng/dual)")
    parser.add_argument("--n_batch",  type=int,   default=256)
    parser.add_argument("--device",   type=str,   default=None)
    parser.add_argument("--gain",       type=float, default=None,
                        help="Override model gain (default: read from DEFAULTS or results.jsonl if --auto_gain)")
    parser.add_argument("--input_size",     type=int,   default=None,
                        help="Override model input_size (default: 8)")
    parser.add_argument("--cue_on_go_input", action="store_true",
                        help="Route GNG cue onto go input channel (reduces input_size by 1)")
    parser.add_argument("--auto_gain", action="store_true",
                        help="(Deprecated/no-op) gain, input_size and cue_on_go_input are "
                             "now read from results.jsonl automatically when present.")
    parser.add_argument("--show",     action="store_true")
    args = parser.parse_args()

    device  = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join(args.ckpt_dir, "figures", "individual", "flows")
    os.makedirs(out_dir, exist_ok=True)

    # Pull gain / input_size / cue_on_go_input from the saved config when available;
    # explicit CLI flags override.  config["input_size"] is already resized.
    cfg = _load_run_config(args.ckpt_dir, args.run_id)
    if cfg is not None:
        print(f"Read config for {args.run_id} from results.jsonl")

    gain = args.gain
    if gain is None and cfg is not None:
        gain = float(cfg["gain"])

    if args.cue_on_go_input:
        cue_on_go_input = True
    elif cfg is not None:
        cue_on_go_input = bool(cfg.get("cue_on_go_input", False))
    else:
        cue_on_go_input = False

    if args.input_size is not None:
        input_size = args.input_size                 # explicit override, used as-is
    elif cfg is not None:
        input_size = int(cfg["input_size"])          # already post-__post_init__
    else:
        input_size = DEFAULTS["input_size"]
        if cue_on_go_input:
            input_size -= 1

    timing_map = make_timing()

    stages = [args.stage] if args.stage else ["dpa", "naive", "expert"]

    for stage in stages:
        ckpt = os.path.join(args.ckpt_dir, "models", f"{stage}_{args.run_id}.pth")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found, skipping: {ckpt}")
            continue

        task     = args.task if args.task else STAGE_TASK[stage]
        out_path = os.path.join(out_dir, f"fp_{stage}_{args.run_id}.pdf")

        print(f"[{stage}] task={task}  gain={gain}  input_size={input_size}  cue_on_go_input={cue_on_go_input}  ckpt={os.path.basename(ckpt)}")
        plot_stage(ckpt, stage, task, device, timing_map,
                   out_path=out_path, n_batch=args.n_batch, gain=gain,
                   input_size=input_size, cue_on_go_input=cue_on_go_input)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

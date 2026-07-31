"""
Rerun only the Dual stage for all runs in a sweep directory.

Loads naive_<run_id>.pth from --source_dir (defaults to --sweep_dir),
trains the dual stage, saves expert_<run_id>.pth to --sweep_dir,
and writes results.jsonl to --sweep_dir.

Usage:
    # in-place rerun
    python rerun_dual.py --sweep_dir results/dual/sweep_nogo0_cue5 --n_gpus 2 --n_workers 10

    # new folder, load checkpoints from original sweep
    python rerun_dual.py \\
        --sweep_dir  results/dual/sweep_nogo0_cue5_nogo2 \\
        --source_dir results/dual/sweep_nogo0_cue5 \\
        --gng_nogo_weight 2.0 --epochs_dual 200 --no_scheduler --n_gpus 2 --n_workers 10
"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from sweep import (
    RunConfig,
    _dpa_accuracy_by_type, _gng_accuracy_by_type, _dual_accuracy,
    train_val_split,
)
from src.models import LowRankModel
from src.tasks import TaskTiming, make_timings, generate_dual_trials
from src.train import Optimization, MaskedMultiTargetLoss, MaskedMultiTargetDualLoss


def rerun_dual_single(config: RunConfig, device: str, out_dir: str, naive_dir: str,
                      epochs_dual: int | None = None, no_scheduler: bool = False,
                      ckpt_prefix: str = "naive") -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    rid       = config.run_id
    DT        = config.dt_base * config.tau_rec_frac
    alpha     = DT / config.tau
    alpha_rec = DT / (config.tau * config.tau_rec_frac)
    noise             = float(config.noise       * torch.sqrt(1.0 - torch.exp(torch.tensor(-alpha)) ** 2))
    model_noise_sigma = float(config.model_noise * torch.sqrt(1.0 - torch.exp(torch.tensor(-alpha)) ** 2))

    _t = make_timings(DT)
    dpa_timing, gng_timing, dual_timing = _t["dpa"], _t["gng"], _t["dual"]

    model = LowRankModel(
        input_size=config.input_size, hidden_size=config.hidden_size,
        output_size=0, rank=config.rank, gain=config.gain,
        alpha=alpha, alpha_rec=alpha_rec, noise=0.0,
        rwd=config.rwd, rwd_scale=config.rwd_scale,
        nonlinearity=config.nonlinearity,
        use_unit_bias=config.use_unit_bias,
        unit_bias_trainable=config.unit_bias_trainable,
        unit_bias_scale=config.unit_bias_scale,
        device=device,
    )

    naive_path = os.path.join(naive_dir, f"{ckpt_prefix}_{rid}.pth")
    model.load_state_dict(torch.load(naive_path, map_location=device))
    print(f"[{rid}]  loaded {ckpt_prefix} checkpoint: {naive_path}", flush=True)

    def _eval(label):
        model.noise = 0.0
        dpa = _dpa_accuracy_by_type(model, dpa_timing, config.input_size, noise=noise, device=device,
                                    target_rank=config.target_rank)
        gng = _gng_accuracy_by_type(model, gng_timing, config.input_size, noise=noise, device=device,
                                    target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
                                    cue_scale=config.cue_scale, nogo_target=config.nogo_target,
                                    go_on_rwd_input=config.go_on_rwd_input)
        print(f"[{rid}]   {label}: "
              f"dpa={dpa['overall']:.3f} (pair={dpa['pair']:.3f} unpair={dpa['unpair']:.3f})  "
              f"gng={gng['overall']:.3f} (go={gng['go']:.3f} nogo={gng['nogo']:.3f})", flush=True)
        return {"dpa": dpa["overall"], "gng": gng["overall"]}

    acc_after_gng = _eval("after GNG (loaded)")

    dual_freeze_input = list(range(config.input_size)) if "dual" in config.freeze_input_stages else []
    dual_freeze_rank0 = [0] if config.freeze_rank0_dual else None

    X, y, _, _ = generate_dual_trials(
        config.n_batch, dual_timing, config.input_size, noise=noise,
        target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
        cue_scale=config.cue_scale, nogo_target=config.nogo_target,
        go_target=config.go_target, go_on_rwd_input=config.go_on_rwd_input,
        input_scale=config.input_scale, attention_input=config.attention_input,
    )
    tl, vl = train_val_split(X.to(device), y.to(device), config.batch_size)

    opt   = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sched = None if no_scheduler else optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-5)

    if config.dual_loss == "separated":
        dual_criterion = MaskedMultiTargetDualLoss(
            timing=dual_timing,
            dpa_weight=config.dpa_weight, gng_weight=config.gng_weight,
            gng_go_weight=config.gng_go_weight, gng_nogo_weight=config.gng_nogo_weight,
            aux_weight=config.aux_weight, bl_weight=config.bl_weight,
            go_hinge_thresh=config.go_hinge_thresh,
            nogo_push_memory=config.nogo_push_memory,
            pin_decay_zeros=config.windowed_targets and config.decay_to_zero,
        )
        print(f"[{rid}]  loss=separated"
              f"  go_w={config.gng_go_weight}  nogo_w={config.gng_nogo_weight}"
              f"  go_hinge={config.go_hinge_thresh}", flush=True)
    else:
        dual_criterion = MaskedMultiTargetLoss(target_weight=1.0, zero_weight=1.0)

    n_epochs = epochs_dual if epochs_dual is not None else config.epochs_dual
    model.noise = model_noise_sigma
    t0 = time.time()
    trainer = Optimization(
        model, tl, vl, dual_criterion, opt, sched,
        config.grad_clip_norm, num_epochs=n_epochs,
        freeze_low_rank_cols=dual_freeze_rank0,
        freeze_input_dims=dual_freeze_input,
        stop_loss=config.stop_loss,
        verbose=True,
    )
    train_l, val_l, _ = trainer.fit()
    dual_loss_components = (dict(dual_criterion.last_components) if config.dual_loss == "separated" else None)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"expert_{rid}.pth"))

    acc_after_dual = _eval("after Dual")
    dual_dpa, dual_gng, dual_go, dual_nogo = _dual_accuracy(
        model, dual_timing, config.input_size, noise=noise, device=device,
        target_rank=config.target_rank, cue_on_go_input=config.cue_on_go_input,
        cue_scale=config.cue_scale, nogo_target=config.nogo_target,
        go_on_rwd_input=config.go_on_rwd_input, input_scale=config.input_scale,
    )
    elapsed = time.time() - t0
    print(f"[{rid}]  Dual done in {elapsed:.1f}s"
          f"  train={train_l[-1]:.4f}  val={val_l[-1]:.4f}"
          f"  dpa={acc_after_dual['dpa']:.3f}  gng={acc_after_dual['gng']:.3f}"
          f"  dual_dpa={dual_dpa:.3f}  dual_gng={dual_gng:.3f}", flush=True)

    return {
        "acc_after_gng":  acc_after_gng,
        "acc_after_dual": {**acc_after_dual, "dual_dpa": dual_dpa, "dual_gng": dual_gng},
        "train_l": train_l,
        "val_l":   val_l,
        "dual_loss_components": dual_loss_components,
    }


def _worker(worker_id: int, n_gpus: int, job_queue: mp.Queue, result_queue: mp.Queue,
            sweep_dir: str, source_dir: str,
            epochs_dual: int | None, no_scheduler: bool, ckpt_prefix: str = "naive"):
    device = f"cuda:{worker_id % n_gpus}" if torch.cuda.is_available() else "cpu"
    while True:
        item = job_queue.get()
        if item is None:
            break
        config, old_result = item
        out_dir   = os.path.join(sweep_dir, config.run_id)
        naive_dir = os.path.join(source_dir, config.run_id)
        log_path  = os.path.join(out_dir, "dual_rerun.log")
        os.makedirs(out_dir, exist_ok=True)
        with open(log_path, "w", buffering=1) as log_f:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = log_f
            try:
                out = rerun_dual_single(config, device, out_dir, naive_dir,
                                        epochs_dual=epochs_dual, no_scheduler=no_scheduler,
                                        ckpt_prefix=ckpt_prefix)
                new_result = dict(old_result)
                new_result["config"]                       = dataclasses.asdict(config)
                new_result["accuracy"]["after_gng"]        = out["acc_after_gng"]
                new_result["accuracy"]["after_dual"]       = out["acc_after_dual"]
                new_result["final_train_loss"]["dual"]      = out["train_l"][-1]
                new_result["final_val_loss"]["dual"]        = out["val_l"][-1]
                new_result["loss_curves"]["dual"]           = {"train": out["train_l"], "val": out["val_l"]}
                new_result["dual_loss_components"]          = out["dual_loss_components"]
                result_queue.put(("ok", config.run_id, new_result))
            except Exception:
                tb = traceback.format_exc()
                print(f"[{config.run_id}] ERROR:\n{tb}", flush=True)
                result_queue.put(("error", config.run_id, tb))
            finally:
                sys.stdout = old_out
                sys.stderr = old_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir",       required=True,
                        help="Output directory for results.jsonl and expert checkpoints")
    parser.add_argument("--source_dir",      default=None,
                        help="Directory with naive checkpoints (default: same as sweep_dir)")
    parser.add_argument("--n_gpus",          type=int,   default=2)
    parser.add_argument("--n_workers",       type=int,   default=None)
    parser.add_argument("--epochs_dual",     type=int,   default=None)
    parser.add_argument("--no_scheduler",    action="store_true")
    parser.add_argument("--gng_go_weight",   type=float, default=None)
    parser.add_argument("--gng_nogo_weight", type=float, default=None)
    parser.add_argument("--nogo_target",     type=float, default=None,
                        help="Override nogo_target for the Dual stage (e.g. 0.0 or -1.0).")
    parser.add_argument("--ckpt_prefix",     type=str,   default="naive",
                        help="Checkpoint prefix to load: 'naive' (after GNG) or 'expert' (after Dual)")
    args = parser.parse_args()

    sweep_dir   = args.sweep_dir
    source_dir  = args.source_dir or sweep_dir
    n_gpus      = min(args.n_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    n_workers   = args.n_workers if args.n_workers is not None else n_gpus
    epochs_dual = args.epochs_dual
    no_scheduler = args.no_scheduler
    ckpt_prefix  = args.ckpt_prefix

    # Load configs from source sweep
    source_results = os.path.join(source_dir, "results.jsonl")
    with open(source_results) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    ok_entries = [e for e in entries if e.get("status") == "ok"]

    # Apply weight overrides to each config
    for e in ok_entries:
        if args.gng_go_weight is not None:
            e["config"]["gng_go_weight"] = args.gng_go_weight
        if args.gng_nogo_weight is not None:
            e["config"]["gng_nogo_weight"] = args.gng_nogo_weight
        if args.nogo_target is not None:
            e["config"]["nogo_target"] = args.nogo_target

    ep_str    = str(epochs_dual) if epochs_dual is not None else "from config"
    sched_str = "none" if no_scheduler else "ReduceLROnPlateau"
    go_w   = args.gng_go_weight   if args.gng_go_weight   is not None else "from config"
    nogo_w = args.gng_nogo_weight if args.gng_nogo_weight is not None else "from config"
    print(f"Rerunning dual stage for {len(ok_entries)} runs | {n_gpus} GPU(s) | {n_workers} workers"
          f" | epochs={ep_str} | scheduler={sched_str}"
          f" | gng_go_w={go_w} | gng_nogo_w={nogo_w}")
    print(f"  source : {source_dir}")
    print(f"  output : {sweep_dir}")

    os.makedirs(sweep_dir, exist_ok=True)

    ctx          = mp.get_context("spawn")
    job_queue    = ctx.Queue()
    result_queue = ctx.Queue()

    for entry in ok_entries:
        cfg = RunConfig(**entry["config"])
        job_queue.put((cfg, entry))
    for _ in range(n_workers):
        job_queue.put(None)

    workers = [ctx.Process(target=_worker,
                           args=(i, n_gpus, job_queue, result_queue,
                                 sweep_dir, source_dir, epochs_dual, no_scheduler, ckpt_prefix))
               for i in range(n_workers)]
    for w in workers:
        w.start()

    results_path  = os.path.join(sweep_dir, "results.jsonl")
    results_by_id = {}
    n_done = 0
    while n_done < len(ok_entries):
        status, run_id, payload = result_queue.get()
        n_done += 1
        if status == "ok":
            results_by_id[run_id] = payload
            acc = payload["accuracy"]["after_dual"]
            print(f"  [{n_done}/{len(ok_entries)}] {run_id}: "
                  f"dpa={acc['dpa']:.3f}  gng={acc['gng']:.3f}  "
                  f"dual_dpa={acc['dual_dpa']:.3f}  dual_gng={acc['dual_gng']:.3f}")
        else:
            print(f"  [{n_done}/{len(ok_entries)}] {run_id}: ERROR — {payload[:200]}")
        with open(results_path, "w") as f:
            for e in results_by_id.values():
                f.write(json.dumps(e) + "\n")

    for w in workers:
        w.join()

    print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

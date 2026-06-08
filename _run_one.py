"""
Worker script for --per_run_screen mode.
Called by sweep.py; not meant to be run directly.

Usage:
    python _run_one.py <config.json> <device> <results.jsonl>
"""
import fcntl
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sweep import RunConfig, run_single

config_path  = sys.argv[1]
device       = sys.argv[2]
results_path = sys.argv[3]

with open(config_path) as f:
    cfg_dict = json.load(f)

config   = RunConfig(**cfg_dict)
run_dir  = os.path.join(config.out_dir, config.run_id)
os.makedirs(run_dir, exist_ok=True)

result = run_single(config, device, models_dir=run_dir)

with open(results_path, "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    f.write(json.dumps(result) + "\n")
    fcntl.flock(f, fcntl.LOCK_UN)

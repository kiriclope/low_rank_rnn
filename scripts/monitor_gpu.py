#!/usr/bin/env python3
"""
GPU monitor for sweep runs.
Usage: python monitor_gpu.py [--interval SECONDS] [--history N]
"""
import argparse
import subprocess
import time
import sys
from collections import deque
from datetime import datetime


def query_gpus():
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = []
    for line in out.strip().splitlines():
        idx, name, util, mem_used, mem_total, temp = [x.strip() for x in line.split(",")]
        gpus.append(dict(
            idx=int(idx), name=name.strip(),
            util=int(util), mem_used=int(mem_used), mem_total=int(mem_total),
            temp=int(temp),
        ))
    return gpus


def count_procs_per_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        uuid_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
            text=True,
        )
        uuid_to_idx = {}
        for line in uuid_out.strip().splitlines():
            parts = line.split(",")
            uuid_to_idx[parts[1].strip()] = int(parts[0].strip())

        counts = {}
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            uuid = parts[0].strip()
            idx = uuid_to_idx.get(uuid, -1)
            counts[idx] = counts.get(idx, 0) + 1
        return counts
    except Exception:
        return {}


def sparkline(values, width=10):
    bars = " ▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    lo, hi = 0, 100
    span = hi - lo or 1
    chars = []
    for v in list(values)[-width:]:
        idx = int((v - lo) / span * (len(bars) - 1))
        chars.append(bars[max(0, min(len(bars) - 1, idx))])
    return "".join(chars).rjust(width)


def mem_bar(used, total, width=20):
    frac = used / total if total else 0
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {used:5d}/{total:5d} MiB ({100*frac:.0f}%)"


def clear():
    print("\033[H\033[J", end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Refresh interval in seconds (default: 3)")
    parser.add_argument("--history",  type=int,   default=30,
                        help="Utilisation history length for sparkline (default: 30)")
    args = parser.parse_args()

    history = {}  # gpu_idx -> deque of util values

    try:
        while True:
            gpus   = query_gpus()
            procs  = count_procs_per_gpu()
            now    = datetime.now().strftime("%H:%M:%S")

            for g in gpus:
                if g["idx"] not in history:
                    history[g["idx"]] = deque(maxlen=args.history)
                history[g["idx"]].append(g["util"])

            clear()
            print(f"  GPU monitor — {now}   (interval {args.interval}s, Ctrl-C to quit)\n")

            for g in gpus:
                idx   = g["idx"]
                spark = sparkline(history[idx], width=args.history)
                n_proc = procs.get(idx, 0)
                print(f"  GPU {idx}  {g['name']}")
                print(f"    Util : {g['util']:3d}%  {spark}  (last {args.history} samples)")
                print(f"    Mem  : {mem_bar(g['mem_used'], g['mem_total'])}")
                print(f"    Temp : {g['temp']}°C   Processes: {n_proc}")
                print()

            sys.stdout.flush()
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

#!/bin/bash
# Queue mem_free behind onesided (never >8 concurrent runs: onesided 4 + mem_early 4 are running).
cd /home/leon/rnn
O=results/dual/sweep_lif_sub_onesided
while [ "$(grep -l 'RUN COMPLETE' $O/s*_onesided/train.log 2>/dev/null | wc -l)" -lt 4 ] && screen -ls | grep -q 'sweep_s[0-9]_onesided'; do sleep 60; done
echo "onesided done $(date '+%F %T'): $(grep -l 'RUN COMPLETE' $O/s*_onesided/train.log | wc -l)/4 complete"
python sweep.py --out_dir results/dual/sweep_lif_sub_mem_free --n_gpus 2 --per_run_screen --run_filter mem_free \
    2>&1 | tee results/dual/sweep_lif_sub_mem_free/launch.log
echo "MEM_FREE_LAUNCHED $(date '+%F %T')"

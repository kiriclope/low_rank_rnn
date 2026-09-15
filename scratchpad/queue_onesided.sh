#!/bin/bash
# Queue the onesided arm behind wave 2 of the tau x noise grid (never >8 concurrent runs).
cd /home/leon/rnn
G=results/dual/sweep_lif_sub_tau_noise
while [ "$(grep -l 'RUN COMPLETE' $G/s*_tau15_*/train.log 2>/dev/null | wc -l)" -lt 8 ] && screen -ls | grep -q 'sweep_s[0-9]_tau15'; do sleep 60; done
echo "wave2 done $(date '+%F %T'): $(grep -l 'RUN COMPLETE' $G/s*_tau15_*/train.log | wc -l)/8 complete"
python sweep.py --out_dir results/dual/sweep_lif_sub_onesided --n_gpus 2 --per_run_screen --run_filter onesided \
    2>&1 | tee results/dual/sweep_lif_sub_onesided/launch.log
echo "ONESIDED_LAUNCHED $(date '+%F %T')"

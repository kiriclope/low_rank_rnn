#!/bin/bash
# Wait for wave 2 (tau15) of the tau x noise grid, then plot its 8 runs and publish the whole grid.
cd /home/leon/rnn
G=results/dual/sweep_lif_sub_tau_noise
while [ "$(grep -l 'RUN COMPLETE' $G/s*_tau15_*/train.log 2>/dev/null | wc -l)" -lt 8 ] && screen -ls | grep -q 'sweep_s[0-9]_tau15'; do sleep 60; done
echo "wave2 done $(date '+%F %T')"
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python plot_sweep.py --sweep_dir $G --out_root results/figures --auto_xlim --device cuda:1 \
    --run_ids s0_tau15_n10 s1_tau15_n10 s2_tau15_n10 s3_tau15_n10 s0_tau15_n15 s1_tau15_n15 s2_tau15_n15 s3_tau15_n15
bash scratchpad/publish_gallery.sh sweep_lif_sub_tau_noise
echo "TAU15_PLOT_DONE $(date '+%F %T')"

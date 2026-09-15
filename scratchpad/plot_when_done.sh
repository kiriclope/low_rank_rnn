#!/bin/bash
# plot_when_done.sh <sweep_dir_name> <n_runs> <run_tag>  — wait for all runs of a per-seed-screen sweep
# to finish (or die), then plot_sweep + publish to the gallery. Always-plot rule.
cd /home/leon/rnn; sw=$1; n=$2; tag=$3; D=results/dual/$sw
done_n() { grep -l 'RUN COMPLETE' $D/s*_$tag/train.log 2>/dev/null | wc -l; }
while [ "$(done_n)" -lt "$n" ]; do
  grep -q "launched screen" $D/launch.log 2>/dev/null && ! screen -ls | grep -q "sweep_s[0-9]_$tag" && sleep 30 && [ "$(done_n)" -lt "$n" ] && { echo "runs died: $(done_n)/$n complete"; break; }
  sleep 60; done
echo "$sw training done $(date '+%F %T'): $(done_n)/$n complete"
LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python plot_sweep.py --sweep_dir $D --out_root results/figures --auto_xlim --device cuda:1
bash scratchpad/publish_gallery.sh $sw
echo "PLOT_DONE $sw $(date '+%F %T')"

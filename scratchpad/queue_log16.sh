#!/bin/bash
# queue_log16.sh — after the extra log seeds s8–s15 finish training (sweep_lif_log, launched 2026-09-30 16:44; Leon:
# "run more of these simulations (more seeds)", "go with 8 for now"): figures + noise-averaged flows + flow verdict +
# gallery, the Fig. 5f–h perturbation runs for the new seeds (separate file, the published perturb_depth.json untouched),
# σ₁ for the new learners (separate file), and the seed-by-seed comparison over all 16 networks.
# Markers (grep the log): LOG16 training done / plots done / perturbation done / sigma1 done / LOG16_DONE
# Run: screen -dmS queue_log16 bash -c "bash scratchpad/queue_log16.sh 2>&1 | tee results/dual/sweep_lif_log/queue_log16.log"
cd /home/leon/rnn
D=results/dual/sweep_lif_log
M=/home/leon/dual/figures/paper_share/modelling
LP=/home/leon/mambaforge/lib/libstdc++.so.6
RIDS="s8_log s9_log s10_log s11_log s12_log s13_log s14_log s15_log"

until python3 -c "import json,sys; d={json.loads(l)['run_id'] for l in open('$D/results.jsonl') if l.strip()}; sys.exit(0 if all(r in d for r in '$RIDS'.split()) else 1)"; do
  sleep 120
done
echo "LOG16 training done $(date)"

# per-run figures for the new seeds only; the sweep summary is redrawn over ALL runs (a --run_ids summary would drop s0–s7)
LD_PRELOAD=$LP python plot_sweep.py --sweep_dir $D --out_root results/figures --run_ids $RIDS --no_summary > $D/plot_wave2.log 2>&1
LD_PRELOAD=$LP python plot_sweep.py --sweep_dir $D --out_root results/figures --no_individual >> $D/plot_wave2.log 2>&1
LD_PRELOAD=$LP python plot_sweep.py --sweep_dir $D --out_root results/figures --run_ids $RIDS --no_summary --plots flow --field_input_noise > $D/plot_noiseavg_wave2.log 2>&1
LD_PRELOAD=$LP python -u flow_verdict.py --sweep_dir $D --stage expert --run_ids $RIDS > $D/verdict_wave2.log 2>&1
./scratchpad/publish_gallery.sh sweep_lif_log >> $D/plot_wave2.log 2>&1
echo "LOG16 plots done $(date)"

cd /home/leon/rnn/paper
SW=results/dual/sweep_lif_log ARM=log SEEDS=8,9,10,11,12,13,14,15 NDELTA=81 NTR=2048 DEV=cuda:1 OUT=$M/perturb_depth_log_s8_15.json \
  LD_PRELOAD=$LP python -u perturb_depth.py > /home/leon/rnn/$D/perturb_wave2.log 2>&1
echo "LOG16 perturbation done $(date)"

LEARN=$(python3 -c "
import json
rows = [json.loads(l) for l in open('/home/leon/rnn/$D/results.jsonl') if l.strip()]
print(','.join(str(int(r['run_id'].split('_')[0][1:])) for r in rows if r['run_id'] in '$RIDS'.split() and r['accuracy']['after_dpa']['dpa'] >= 0.95))")
echo "new learners (after_dpa DPA >= 0.95): ${LEARN:-none}"
if [ -n "$LEARN" ]; then
  SEEDS=$LEARN OUT=$M/sigma1_prediction_s8_15.json LD_PRELOAD=$LP python -u sigma1_prediction.py > /home/leon/rnn/$D/sigma1_wave2.log 2>&1
fi
echo "LOG16 sigma1 done $(date)"

python compare_log16.py > /home/leon/rnn/$D/compare_log16.log 2>&1
echo "LOG16_DONE $(date)"

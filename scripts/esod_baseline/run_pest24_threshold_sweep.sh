#!/usr/bin/env bash
# Threshold sweep (HESOD-Agri-Experiment-Plan.md SS8.1's grid, never run for
# Pest24 before now): for each of our four already-trained fusion arms,
# re-evaluate the SAME checkpoint at 7 different HeatMapParser thresholds.
# No retraining -- threshold is purely an inference-time parameter
# (models/common.py, HeatMapParser.threshold), so this only costs eval time.
#
# Answers the question raised in this session's TP-YOLO/A0 discussion:
# is our current fixed threshold=0.5 (hard-coded in every Pest24 yaml's
# HeatMapParser[256,8,0.5], never swept for Pest24's density) leaving free
# compute savings on the table, or is it already past the point where
# raising it would cost real recall? R0's own coverage-vs-miss diagnostic
# shows selector-dropped misses are still the dominant failure mode even in
# our best arm (concat+SABL, 49.5% of misses) at threshold=0.5 -- weak
# evidence AGAINST a "free lunch" from raising it further, but only the
# actual sweep settles this.
#
# Reports the full Pareto curve (accuracy AND cost per threshold), not a
# single cherry-picked point -- per SS8.1's explicit requirement.
#
# Usage: nohup bash run_pest24_threshold_sweep.sh > /root/pest24_sweep.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-1024}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_YAML="/root/autodl-tmp/Pest24_v1.yaml"
SWEEP_ROOT="$RUN_ROOT/sweep"

# HESOD-Agri-Experiment-Plan.md SS8.1's exact predeclared grid
THRESHOLDS=(0.10 0.20 0.30 0.40 0.50 0.60 0.70)

# The four fusion arms this session's TP-YOLO/A0 discussion is actually
# about -- R0 omitted (semantic-only baseline, not part of the "does the
# fusion mechanism's routing cost its keep" question this sweep targets).
ARMS=(
  "pest24_yolov5m_channel_pooled_concat"
  "pest24_yolov5m_channel_pooled_concat_sabl"
  "pest24_yolov5m_reliability_gate"
  "pest24_yolov5m_reliability_gate_sabl"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$ESOD_REPO"

for run_name in "${ARMS[@]}"; do
  ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"
  if [ ! -f "$ckpt" ]; then
    log "===== SKIP $run_name: no checkpoint at $ckpt ====="
    continue
  fi

  for thr in "${THRESHOLDS[@]}"; do
    out_dir="$SWEEP_ROOT/$run_name/thr_${thr}"
    mkdir -p "$out_dir"

    log "===== $run_name @ threshold=$thr : eval (AP/AP50/BPR/Occupy) ====="
    python test.py \
      --data "$DATA_YAML" --weights "$ckpt" --task test --hm-threshold "$thr" \
      --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" \
      --project "$SWEEP_ROOT/$run_name" --name "thr_${thr}" --exist-ok \
      2>&1 | tee "$out_dir/${run_name}_thr${thr}_test.log"

    log "===== $run_name @ threshold=$thr : measure (GFLOPs/FPS, batch=1) ====="
    python test.py \
      --data "$DATA_YAML" --weights "$ckpt" --task measure --hm-threshold "$thr" \
      --batch-size 1 --img-size "$IMG_SIZE" --device "$GPU" \
      --project "$SWEEP_ROOT/$run_name" --name "thr_${thr}_measure" --exist-ok \
      2>&1 | tee "$out_dir/${run_name}_thr${thr}_measure.log"
  done
done

log "===== SWEEP DONE ====="
log "Results under $SWEEP_ROOT/<arm>/thr_<threshold>/"
log "Quick aggregate: grep -H '^ *all' \$SWEEP_ROOT/*/thr_*/*_test.log"
log "                 grep -H 'GFLOPs:' \$SWEEP_ROOT/*/thr_*/*_measure.log"

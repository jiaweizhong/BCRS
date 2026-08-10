#!/usr/bin/env bash
# Top-K patch-budget ablation sweep on already-trained VisDrone_v2 checkpoints --
# no retraining. Forces HeatMapParser's inference-time patch selection to exactly
# K patches (test.py's --top-k flag), for K in {16,32,48,64}, across the three
# checkpoints trained so far (baseline / dual-evidence-concat / channel-pooled-concat).
#
# --top-k only exists in hesod/backends/hesod/ (the dev tree) -- hesod/backends/esod/
# (the pristine mirror baseline was trained through) never got this flag added. All
# three checkpoints are therefore evaluated through hesod/backends/hesod/'s test.py,
# including the baseline one, even though it was trained through the esod mirror.
# The baseline architecture (plain Segmenter) is identical in both trees, so the
# checkpoint should load fine, but this exact cross-tree load has not been exercised
# before this script -- if it fails on a class-resolution error, that's the first
# thing to look at.
#
# Usage: bash run_topk_sweep.sh [K ...]   (default: 16 32 48 64)

set -euo pipefail

HESOD_DEV="${HESOD_DEV:-$HOME/BCRS/hesod/backends/hesod}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-$HOME/BCRS/scripts/esod_baseline/audit_buckets.py}"
DATA="${DATA:-/root/autodl-tmp/VisDrone_v2.yaml}"
LABELS="${LABELS:-/root/autodl-tmp/VisDrone_v2/labels/val}"
IMAGES="${IMAGES:-/root/autodl-tmp/VisDrone_v2/images/val}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
IMG_SIZE="${IMG_SIZE:-1536}"
GPU="${GPU:-0}"

BASELINE_CKPT="${BASELINE_CKPT:-$RUN_ROOT/train/visdrone_yolov5m_official_data/weights/best.pt}"
DUAL_CKPT="${DUAL_CKPT:-$RUN_ROOT/train/visdrone_yolov5m_dual_evidence_concat/weights/best.pt}"
POOLED_CKPT="${POOLED_CKPT:-$RUN_ROOT/train/visdrone_yolov5m_channel_pooled_concat/weights/best.pt}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_one() {
  local ckpt="$1" name="$2" k="$3"
  local run_name="${name}_topk${k}"
  local out_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$out_dir"

  log "===== $run_name ====="
  cd "$HESOD_DEV"
  python test.py \
    --data "$DATA" --weights "$ckpt" \
    --batch-size 8 --img-size "$IMG_SIZE" --device "$GPU" --save-json \
    --top-k "$k" \
    --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
    2>&1 | tee "$out_dir/${run_name}_test.log"

  python "$AUDIT_SCRIPT" \
    --pred "$out_dir/best_predictions.json" \
    --labels "$LABELS" --images "$IMAGES" \
    2>&1 | tee "$out_dir/${run_name}_audit.log"
}

KS=("$@")
if [ ${#KS[@]} -eq 0 ]; then
  KS=(16 32 48 64)
fi

for K in "${KS[@]}"; do
  run_one "$BASELINE_CKPT" visdrone_yolov5m_baseline "$K"
  run_one "$DUAL_CKPT" visdrone_yolov5m_dual_evidence_concat "$K"
  run_one "$POOLED_CKPT" visdrone_yolov5m_channel_pooled_concat "$K"
done

log "All done. Results under $RUN_ROOT/test/<arm>_topk<K>/"

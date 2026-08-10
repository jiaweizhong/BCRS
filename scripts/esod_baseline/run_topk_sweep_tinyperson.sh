#!/usr/bin/env bash
# Top-K patch-budget sweep on the TinyPerson baseline + channel-pooled-concat
# checkpoints (scratch-hyp recipe) -- no retraining, mirrors
# run_topk_sweep.sh's VisDrone version. Motivation: channel-pooled-concat's
# free-threshold selector settled at Occupy=0.392 (~25/64 patches) vs.
# baseline's Occupy=0.128 (~8/64 patches) -- a real AP gain (see
# HESOD-Experiment-Plan.md SS4.3) but at +64% GFLOPs. This sweep answers two
# questions at once by testing BOTH checkpoints across the same K values:
#   1. Does forcing channel-pooled-concat down to a smaller, cheaper K
#      recover most of its AP gain at much lower GFLOPs? (the efficiency
#      question)
#   2. At a MATCHED K (matched compute), does channel-pooled-concat still
#      beat baseline? If yes, the dual-evidence fusion mechanism is adding
#      value beyond "just look at more area" (baseline could do that too by
#      loosening its own threshold) -- if baseline catches up at matched K,
#      the earlier free-threshold "win" was really just about area, not
#      fusion quality.
#
# K range brackets both checkpoints' natural operating points (~8 and ~25 out
# of 64 total coarse patches -- same ratio=8 HeatMapParser as VisDrone, so
# the 64-patch grid size is resolution-independent and the VisDrone K range
# transfers directly): 8, 16, 24, 32, 48, 64.
#
# Usage: bash run_topk_sweep_tinyperson.sh [K ...]   (default: 8 16 24 32 48 64)

set -euo pipefail

HESOD_DEV="${HESOD_DEV:-$HOME/BCRS/hesod/backends/hesod}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-$HOME/BCRS/scripts/esod_baseline/audit_buckets.py}"
DATA="${DATA:-/root/autodl-tmp/TinyPerson_v1.yaml}"
LABELS="${LABELS:-/root/autodl-tmp/TinyPerson_v1/labels/val}"
IMAGES="${IMAGES:-/root/autodl-tmp/TinyPerson_v1/images/val}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
IMG_SIZE="${IMG_SIZE:-2048}"
GPU="${GPU:-0}"

BASELINE_CKPT="${BASELINE_CKPT:-$RUN_ROOT/train/tinyperson_yolov5m_baseline/weights/best.pt}"
POOLED_CKPT="${POOLED_CKPT:-$RUN_ROOT/train/tinyperson_yolov5m_channel_pooled_concat/weights/best.pt}"

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
    --labels "$LABELS" --images "$IMAGES" --classes person \
    2>&1 | tee "$out_dir/${run_name}_audit.log"
}

KS=("$@")
if [ ${#KS[@]} -eq 0 ]; then
  KS=(8 16 24 32 48 64)
fi

for K in "${KS[@]}"; do
  run_one "$BASELINE_CKPT" tinyperson_yolov5m_baseline "$K"
  run_one "$POOLED_CKPT" tinyperson_yolov5m_channel_pooled_concat "$K"
done

log "All done. Results under $RUN_ROOT/test/<arm>_topk<K>/"

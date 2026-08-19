#!/usr/bin/env bash
# TinyPerson R0 (paper-comparable baseline) retrained with --disable-half
# (forces pure FP32 training, train.py's `half_precision = not opt.disable_half`
# defaults to AMP/FP16 mixed precision otherwise). Control run for the
# unresolved ~6pp APt50 / ~3pp APs50 reproduction gap (HESOD-Experiment-Plan.md
# SS2) -- the paper's own text says nothing about mixed precision (confirmed
# absent from reference/ESOD.pdf, same category as SyncBN/augmentation/NMS
# threshold, already-unspecified variables). Mixed precision plausibly hurts
# tiny-object localization specifically more than larger objects (box w/h in
# the single-digit-pixel range is more numerically sensitive than a 100px
# box), which would fit the APt50 > APs50 gap asymmetry -- untested until now.
# test.py/measure already default to FP32 (`--half` defaults False), so this
# is purely a training-time variable; no change needed on the eval side.
#
# This is a single targeted control, not part of the regular R0/concat/gate
# roster -- run it once, compare against the existing AMP-trained R0 result
# (55.26/71.23 APt50/APs50, HESOD-Experiment-Plan.md SS2) using the same
# official evaluator.
#
# Usage:
#   SMOKE=1 bash run_tinyperson_disable_half.sh       # ~minutes, validate first
#   nohup bash run_tinyperson_disable_half.sh > /root/tinyperson_fp32.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="/root/autodl-tmp/TinyPerson_v1"
DATA_YAML="/root/autodl-tmp/TinyPerson_v1.yaml"
HYP="data/hyps/hyp.tinyperson.yaml"
CLASSES="person"
TINY_SET_GT="/root/autodl-tmp/tiny_set/mini_annotations/tiny_set_test_all.json"

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  EPOCHS=1
  SUFFIX="_smoke"
  log_prefix="SMOKE"
else
  EPOCHS="${EPOCHS:-50}"
  SUFFIX=""
  log_prefix="REAL"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$log_prefix] $*"; }

log "epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML (--disable-half)"

run_name="tinyperson_yolov5m_baseline_fp32${SUFFIX}"
results_dir="$RUN_ROOT/test/$run_name"
mkdir -p "$results_dir"
cd "$ESOD_REPO"

ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"
if [ -f "$ckpt" ]; then
  log "===== $run_name already trained (found $ckpt), skipping training ====="
else
  log "===== Training $run_name (--disable-half, pure FP32) ====="
  python train.py \
    --data "$DATA_YAML" \
    --cfg models/cfg/esod/tinyperson_yolov5m.yaml \
    --weights weights/pretrained/yolov5m.pt \
    --hyp "$HYP" \
    --batch-size "$BATCH" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
    --disable-half \
    --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_train.log"
fi

if [ ! -f "$ckpt" ]; then
  log "FATAL: $run_name training finished but $ckpt does not exist -- aborting before eval"
  exit 1
fi

log "Evaluating $run_name"
python test.py \
  --data "$DATA_YAML" --weights "$ckpt" --task test \
  --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" --save-json --save-regions \
  --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
  2>&1 | tee "$results_dir/${run_name}_test.log"

log "Measuring $run_name (GFLOPs/FPS, batch=1)"
python test.py \
  --data "$DATA_YAML" --weights "$ckpt" \
  --batch-size 1 --img-size "$IMG_SIZE" --device "$GPU" --task measure \
  --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
  2>&1 | tee "$results_dir/${run_name}_measure.log"

if [ ! -f "$results_dir/best_predictions.json" ]; then
  log "WARNING: $run_name produced no predictions (expected at 1 epoch under SMOKE) -- skipping audit/vt_diagnose/official-eval"
else
  log "Auditing $run_name"
  python "$SCRIPT_DIR/audit_buckets.py" \
    --pred "$results_dir/best_predictions.json" \
    --labels "$DATA_ROOT/labels/val" --images "$DATA_ROOT/images/val" \
    --classes "$CLASSES" \
    2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed, continuing"

  log "vt_diagnose: $run_name"
  python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
    --labels-dir "$DATA_ROOT/labels/val" --images-dir "$DATA_ROOT/images/val" \
    --classes "$CLASSES" \
    2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed, continuing"

  if [ "$SMOKE" = "1" ]; then
    log "Skipping official TinyPerson evaluator under SMOKE (1-epoch predictions aren't meaningful for APt50/APs50)"
  else
    log "Official TinyPerson evaluator (APt50/APs50, paper-comparable protocol): $run_name"
    python "$SCRIPT_DIR/tinyperson_eval/eval_tinyperson_official.py" \
      --pred "$results_dir/best_predictions.json" \
      --gt "$TINY_SET_GT" \
      2>&1 | tee "$results_dir/${run_name}_official_eval.log" || log "WARNING: official evaluator failed, continuing"
  fi
fi

log "===== ALL DONE ====="
log "  FP32 R0: $RUN_ROOT/test/$run_name/"
log "  Compare ${run_name}_official_eval.log's APt50/APs50 against the AMP-trained"
log "  R0 result (55.26/71.23, HESOD-Experiment-Plan.md SS2)."

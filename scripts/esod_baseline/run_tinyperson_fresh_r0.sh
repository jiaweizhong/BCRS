#!/usr/bin/env bash
# Audited TinyPerson ESOD R0 reproduction runner.
#
# Despite the historical filename, "fresh" means a freshly prepared copy of
# the official no-dense benchmark, not the 794/816 with_dense dataset and not
# the Roboflow augmented export. Two deliberately distinct protocols exist:
#
#   PROTOCOL=paper    lr0=0.01, focal:dice=20:1, paper Eq.(4) hybrid masks
#   PROTOCOL=released lr0=0.005, weighted BCE, released hybrid-mask code
#
# Both use erased ignore/uncertain images, tiny_set_test_all.json, YOLOv5m
# initialization, ratio=8, 2048 input, 50 epochs, and global batch 8. Run them
# separately; their names and output roots cannot collide.
#
# Prepare a clean dataset first:
#   cd /root/BCRS/hesod/backends/hesod
#   python scripts/data_prepare.py \
#     --dataset /root/autodl-tmp/tiny_set_paper \
#     --tinyperson-mask-mode paper-hybrid
#   cd /root/BCRS
#   python scripts/esod_baseline/reorganize_tinyperson.py \
#     --raw-root /root/autodl-tmp/tiny_set_paper \
#     --out-root /root/autodl-tmp/TinyPerson_paper
#
# Usage:
#   SMOKE=1 PROTOCOL=paper bash scripts/esod_baseline/run_tinyperson_fresh_r0.sh
#   PROTOCOL=paper bash scripts/esod_baseline/run_tinyperson_fresh_r0.sh
#   PROTOCOL=released MASK_MODE=released-hybrid \
#     bash scripts/esod_baseline/run_tinyperson_fresh_r0.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-${GPU:-0}}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
EPOCHS="${EPOCHS:-50}"
PROTOCOL="${PROTOCOL:-paper}"
RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/tiny_set_${PROTOCOL}}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/TinyPerson_${PROTOCOL}}"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/TinyPerson_${PROTOCOL}.yaml}"
GT_JSON="${GT_JSON:-$RAW_ROOT/mini_annotations/tiny_set_test_all.json}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-0}"

case "$PROTOCOL" in
  paper)
    HYP="data/hyps/hyp.tinyperson.yaml"
    MASK_MODE="${MASK_MODE:-paper-hybrid}"
    TRAIN_FLAGS=(--selector-loss paper)
    ;;
  released)
    HYP="data/hyps/hyp.tinyperson.released.yaml"
    MASK_MODE="${MASK_MODE:-released-hybrid}"
    TRAIN_FLAGS=(--selector-loss upstream)
    ;;
  *)
    echo "FATAL: PROTOCOL must be 'paper' or 'released', got '$PROTOCOL'" >&2
    exit 2
    ;;
esac

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  EPOCHS=1
  RUN_KIND="smoke"
else
  RUN_KIND="real"
fi

RUN_ROOT="${RUN_ROOT:-$HOME/esod_tinyperson_${PROTOCOL}_${MASK_MODE}_${EPOCHS}ep}"
RUN_NAME="tinyperson_esod_${PROTOCOL}_r0_ratio8_${MASK_MODE}_${EPOCHS}ep_${RUN_KIND}"
RESULTS_DIR="$RUN_ROOT/test/$RUN_NAME"
CKPT="$RUN_ROOT/train/$RUN_NAME/weights/best.pt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$PROTOCOL/$RUN_KIND] $*"; }

if [ "$BATCH" -ne 8 ]; then
  log "WARNING: paper global batch is 8; requested BATCH=$BATCH"
fi
if [[ "$PROTOCOL" == "paper" && "$GPU" != *,* ]]; then
  log "WARNING: paper trained on two V100 GPUs; this run uses device specification '$GPU'"
fi

log "Auditing TinyPerson dataset before training"
python "$SCRIPT_DIR/audit_tinyperson_protocol.py" \
  --data-root "$DATA_ROOT" \
  --gt "$GT_JSON" \
  --expected-mask-mode "$MASK_MODE"

mkdir -p "$RESULTS_DIR"
cd "$ESOD_REPO"

log "protocol=$PROTOCOL mask=$MASK_MODE epochs=$EPOCHS img=$IMG_SIZE batch=$BATCH device=$GPU"
log "data=$DATA_YAML gt=$GT_JSON run=$RUN_NAME"

if [ -f "$CKPT" ]; then
  if [ "$REUSE_CHECKPOINTS" = "1" ]; then
    log "Reusing explicitly requested checkpoint: $CKPT"
  else
    log "FATAL: checkpoint already exists: $CKPT"
    log "Set REUSE_CHECKPOINTS=1 only after verifying its opt.yaml and protocol manifest."
    exit 1
  fi
else
  python train.py \
    --data "$DATA_YAML" \
    --cfg models/cfg/esod/tinyperson_yolov5m.yaml \
    --weights weights/pretrained/yolov5m.pt \
    --hyp "$HYP" \
    --batch-size "$BATCH" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
    "${TRAIN_FLAGS[@]}" \
    --project "$RUN_ROOT/train" --name "$RUN_NAME" --exist-ok \
    2>&1 | tee "$RESULTS_DIR/${RUN_NAME}_train.log"
fi

if [ ! -f "$CKPT" ]; then
  log "FATAL: training completed without $CKPT"
  exit 1
fi

python test.py \
  --data "$DATA_YAML" --weights "$CKPT" --task test \
  --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" --save-json --save-regions \
  --project "$RUN_ROOT/test" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$RESULTS_DIR/${RUN_NAME}_test.log"

python test.py \
  --data "$DATA_YAML" --weights "$CKPT" --task measure \
  --batch-size 1 --img-size "$IMG_SIZE" --device "$GPU" \
  --project "$RUN_ROOT/measure" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$RESULTS_DIR/${RUN_NAME}_measure.log"

PRED_JSON="$RESULTS_DIR/best_predictions.json"
if [ ! -f "$PRED_JSON" ]; then
  if [ "$SMOKE" = "1" ]; then
    log "SMOKE produced no predictions; official evaluation skipped"
    exit 0
  fi
  log "FATAL: missing predictions: $PRED_JSON"
  exit 1
fi

python "$SCRIPT_DIR/tinyperson_eval/eval_tinyperson_official.py" \
  --pred "$PRED_JSON" --gt "$GT_JSON" \
  2>&1 | tee "$RESULTS_DIR/${RUN_NAME}_official_eval.log"

python "$SCRIPT_DIR/audit_buckets.py" \
  --pred "$PRED_JSON" \
  --labels "$DATA_ROOT/labels/val" --images "$DATA_ROOT/images/val" \
  --classes person \
  2>&1 | tee "$RESULTS_DIR/${RUN_NAME}_audit.log"

log "DONE: $RESULTS_DIR"

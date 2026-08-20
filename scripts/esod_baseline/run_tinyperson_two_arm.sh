#!/usr/bin/env bash
# Train exactly two arms on either official TinyPerson or TinyPerson-Aug:
#   1) paper-loss ESOD R0
#   2) HESOD channel-pooled concat + ISPHead + coverage/SABL treatment
#
# DATASET=official uses the official no-dense/erased protocol and APt50/APs50.
# DATASET=aug uses the Roboflow splits and ordinary YOLO metrics only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ESOD_REPO="$REPO_ROOT/hesod/backends/hesod"

DATASET="${DATASET:-official}"
MASK_MODE="${MASK_MODE:-paper-hybrid}"
GPU="${GPU:-0}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
EPOCHS="${EPOCHS:-50}"
SMOKE="${SMOKE:-0}"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-0}"
HYP="data/hyps/hyp.tinyperson.yaml"
CLASSES="${CLASSES:-0}"

case "$DATASET" in
  official)
    RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/tinyperson_raw}"
    DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/TinyPerson_official_paper}"
    DATA_YAML="${DATA_YAML:-/root/autodl-tmp/TinyPerson_official_paper.yaml}"
    GT_JSON="${GT_JSON:-$RAW_ROOT/mini_annotations/tiny_set_test_all.json}"
    EVAL_SPLIT="val"
    DATA_TAG="official_paper"
    ;;
  aug)
    RAW_ROOT="${RAW_ROOT:-/root/autodl-tmp/tinyperson-aug}"
    DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/TinyPerson_aug_${MASK_MODE}}"
    DATA_YAML="${DATA_YAML:-/root/autodl-tmp/TinyPerson_aug_${MASK_MODE}.yaml}"
    GT_JSON=""
    # test.py --task test reads the YAML `test` entry, which is the 65-image
    # Roboflow test split. Keep the bucket audit on that exact same split.
    EVAL_SPLIT="test"
    DATA_TAG="aug_exploratory"
    ;;
  *)
    echo "DATASET must be official or aug, got: $DATASET" >&2
    exit 2
    ;;
esac

if [[ "$SMOKE" == "1" ]]; then
  EPOCHS=1
  RUN_KIND="smoke"
else
  RUN_KIND="full"
fi

RUN_ROOT="${RUN_ROOT:-$HOME/esod_tinyperson_twoarm_${DATA_TAG}_${MASK_MODE}_${EPOCHS}ep}"
RESULTS_ROOT="$REPO_ROOT/results/tinyperson_twoarm/$DATA_TAG"
mkdir -p "$RUN_ROOT" "$RESULTS_ROOT"

log() {
  printf '[tinyperson-two-arm] %s\n' "$*"
}

[[ -f "$DATA_YAML" ]] || { log "missing dataset YAML: $DATA_YAML"; exit 1; }

if [[ "$DATASET" == "official" ]]; then
  python "$SCRIPT_DIR/audit_tinyperson_protocol.py" \
    --data-root "$DATA_ROOT" \
    --gt "$GT_JSON" \
    --expected-mask-mode "$MASK_MODE"
else
  python "$SCRIPT_DIR/audit_tinyperson_aug.py" \
    --data-root "$DATA_ROOT" \
    --expected-mask-mode "$MASK_MODE"
fi

if [[ "$DATASET" == "official" && "$GPU" != *","* ]]; then
  log "WARNING: paper used two V100 GPUs; current device specification is '$GPU'"
fi
if [[ "$BATCH" != "8" ]]; then
  log "WARNING: audited protocol uses global batch 8; current batch is $BATCH"
fi

cd "$ESOD_REPO"

run_arm() {
  local run_name="$1"
  local model_cfg="$2"
  shift 2
  local extra_flags=("$@")
  local result_dir="$RESULTS_ROOT/$run_name"
  local checkpoint="$RUN_ROOT/train/$run_name/weights/best.pt"
  mkdir -p "$result_dir"

  if [[ -f "$checkpoint" && "$REUSE_CHECKPOINTS" != "1" ]]; then
    log "refusing existing checkpoint: $checkpoint"
    log "use a new RUN_ROOT, or explicitly set REUSE_CHECKPOINTS=1"
    exit 1
  fi

  if [[ ! -f "$checkpoint" ]]; then
    log "training $run_name"
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights weights/pretrained/yolov5m.pt \
      --hyp "$HYP" \
      --batch-size "$BATCH" \
      --img-size "$IMG_SIZE" \
      --epochs "$EPOCHS" \
      --device "$GPU" \
      "${extra_flags[@]}" \
      --project "$RUN_ROOT/train" \
      --name "$run_name" \
      --exist-ok 2>&1 | tee "$result_dir/${run_name}_train.log"
  else
    log "explicitly reusing $checkpoint"
  fi

  log "testing $run_name"
  python test.py \
    --data "$DATA_YAML" \
    --weights "$checkpoint" \
    --task test \
    --batch-size "$BATCH" \
    --img-size "$IMG_SIZE" \
    --device "$GPU" \
    --save-json \
    --save-regions \
    --project "$RUN_ROOT/test" \
    --name "$run_name" \
    --exist-ok 2>&1 | tee "$result_dir/${run_name}_test.log"

  local prediction_json="$RUN_ROOT/test/$run_name/best_predictions.json"
  [[ -f "$prediction_json" ]] || { log "missing predictions: $prediction_json"; exit 1; }
  cp "$prediction_json" "$result_dir/best_predictions.json"

  python test.py \
    --data "$DATA_YAML" \
    --weights "$checkpoint" \
    --batch-size 1 \
    --img-size "$IMG_SIZE" \
    --device "$GPU" \
    --task measure \
    --project "$RUN_ROOT/measure" \
    --name "$run_name" \
    --exist-ok 2>&1 | tee "$result_dir/${run_name}_measure.log"

  if [[ "$DATASET" == "official" ]]; then
    python "$SCRIPT_DIR/tinyperson_eval/eval_tinyperson_official.py" \
      --pred "$result_dir/best_predictions.json" \
      --gt "$GT_JSON" 2>&1 | tee "$result_dir/${run_name}_official_eval.log"
  fi

  python "$SCRIPT_DIR/audit_buckets.py" \
    --pred "$result_dir/best_predictions.json" \
    --labels "$DATA_ROOT/labels/$EVAL_SPLIT" \
    --images "$DATA_ROOT/images/$EVAL_SPLIT" \
    --classes "$CLASSES" 2>&1 | tee "$result_dir/${run_name}_audit.log"
}

SUFFIX="_ratio8_${MASK_MODE}_${EPOCHS}ep_${RUN_KIND}"

run_arm "tinyperson_${DATA_TAG}_r0${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m.yaml" \
  --selector-loss paper

run_arm "tinyperson_${DATA_TAG}_concat_isphead_coverage_sabl${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "complete: exactly two arms were run under $RUN_ROOT"

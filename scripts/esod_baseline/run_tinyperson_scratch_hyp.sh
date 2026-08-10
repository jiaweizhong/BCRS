#!/usr/bin/env bash
# Re-run TinyPerson with data/hyps/hyp.tinyperson.scratch.yaml instead of
# hyp.tinyperson.finetune.yaml (see HESOD-Experiment-Plan.md SS4.2 for why
# finetune was likely the wrong choice), as three separate single-variable
# arms so a selector-side change and a head-side change never land in the
# same run:
#
#   1. baseline               -- plain Segmenter, upstream selector loss,
#                                 upstream box loss. Via run_baseline.sh
#                                 (hesod/backends/esod, the pristine tree),
#                                 same as every other dataset's baseline.
#                                 Run name: tinyperson_yolov5m_baseline
#   2. channel-pooled concat  -- ChannelPooledConcatEvidenceSegmenter,
#                                 --selector-loss coverage (selector-side
#                                 only, upstream box loss). Same recipe as
#                                 the VisDrone roster arm. Via
#                                 hesod/backends/hesod (only tree with
#                                 --selector-loss / the pooled-concat cfg).
#                                 Run name: tinyperson_yolov5m_channel_pooled_concat
#   3. box-size-weighted      -- plain Segmenter, upstream selector loss,
#                                 --box-loss size_weighted (head-side only,
#                                 targets small-object box regression, the
#                                 #1-ranked VisDrone gap item). Via
#                                 hesod/backends/hesod (only tree with
#                                 --box-loss).
#                                 Run name: tinyperson_yolov5m_box_size_weighted
#
# All three use the same scratch hyp, same img-size 2048, same data. ~30min/
# arm on TinyPerson, so all three run sequentially on one GPU without much
# total wait.
#
# Usage: bash run_tinyperson_scratch_hyp.sh [gpu_index]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
HESOD_DEV="${HESOD_DEV:-$SCRIPT_DIR/../../hesod/backends/hesod}"
DATA="${DATA:-/root/autodl-tmp/TinyPerson_v1.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/TinyPerson_v1}"
IMG_SIZE=2048
BATCH=8
EPOCHS="${EPOCHS:-50}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_dev_arm() {
  local run_name="$1" model_cfg="$2"
  shift 2
  local extra_flags=("$@")

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$HESOD_DEV"

  log "Training $run_name -> $results_dir/${run_name}_train.log"
  python train.py \
    --data "$DATA" \
    --cfg "$model_cfg" \
    --weights weights/pretrained/yolov5m.pt \
    --hyp data/hyps/hyp.tinyperson.scratch.yaml \
    "${extra_flags[@]}" \
    --batch-size $BATCH --img-size $IMG_SIZE --epochs "$EPOCHS" --device "$GPU" \
    --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_train.log"

  local ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"

  log "Evaluating $run_name -> $results_dir/${run_name}_test.log"
  python test.py \
    --data "$DATA" --weights "$ckpt" \
    --batch-size $BATCH --img-size $IMG_SIZE --device "$GPU" --save-json \
    --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_test.log"

  log "Measuring $run_name (GFLOPs/FPS, batch=1) -> $results_dir/${run_name}_measure.log"
  python test.py \
    --data "$DATA" --weights "$ckpt" \
    --batch-size 1 --img-size $IMG_SIZE --device "$GPU" --task measure \
    --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_measure.log"

  log "Auditing $run_name -> $results_dir/${run_name}_audit.log"
  python "$SCRIPT_DIR/audit_buckets.py" \
    --pred "$results_dir/best_predictions.json" \
    --labels "$DATASET_ROOT/labels/val" --images "$DATASET_ROOT/images/val" \
    --classes person \
    2>&1 | tee "$results_dir/${run_name}_audit.log"
}

log "===== 1/3: baseline, scratch hyp (via run_baseline.sh, hesod/backends/esod) ====="
GPU="$GPU" RUN_ROOT="$RUN_ROOT" EPOCHS="$EPOCHS" bash "$SCRIPT_DIR/run_baseline.sh" tinyperson "$GPU"

log "===== 2/3: channel-pooled dual-evidence concat, scratch hyp (selector-side only) ====="
run_dev_arm tinyperson_yolov5m_channel_pooled_concat \
  models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat.yaml \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0

log "===== 3/3: box-regression size-weighting, scratch hyp (head-side only) ====="
run_dev_arm tinyperson_yolov5m_box_size_weighted \
  models/cfg/esod/tinyperson_yolov5m.yaml \
  --box-loss size_weighted --box-weight-ref-area 4.0 --box-weight-max 5.0

log "All done."
log "  baseline:          $RUN_ROOT/test/tinyperson_yolov5m_baseline/"
log "  channel-pooled:    $RUN_ROOT/test/tinyperson_yolov5m_channel_pooled_concat/"
log "  box-size-weighted: $RUN_ROOT/test/tinyperson_yolov5m_box_size_weighted/"

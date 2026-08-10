#!/usr/bin/env bash
# TinyPerson ratio=8 -> ratio=16 test, per the ESOD paper's own patch-size
# ablation text: "1/8 is a proper coefficient ... on the current VisDrone
# dataset. As for other datasets like TinyPerson ... 1/16 may be a more
# suitable choice." tinyperson_yolov5m.yaml copied VisDrone's ratio=8
# unchanged -- never adjusted for TinyPerson specifically until now.
#
# Two arms, both on the scratch-hyp recipe (HESOD-Experiment-Plan.md SS4.1):
#   1. baseline + ratio=16     -- isolates the ratio variable alone.
#      cfg: tinyperson_yolov5m_ratio16.yaml
#   2. full-spectral concat + ratio=16 -- ConcatEvidenceSegmenter (NOT
#      channel-pooled -- full MultiKernelSpectralFilter resolution), combined
#      with the ratio fix. TinyPerson's channel-pooled arm was already a
#      positive result (SS4.3) with compute headroom (baseline Occupy only
#      0.128), so this tests whether full spectral resolution does even
#      better once patch granularity is also fixed.
#      cfg: tinyperson_yolov5m_concat_ratio16.yaml
#
# Both need hesod/backends/hesod -- ConcatEvidenceSegmenter only exists in
# that tree, not the pristine esod/ / hesod/backends/esod/ mirrors.
#
# Usage: bash run_tinyperson_ratio16.sh [gpu_index]

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

run_arm() {
  local run_name="$1" model_cfg="$2"

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$HESOD_DEV"

  log "Training $run_name -> $results_dir/${run_name}_train.log"
  python train.py \
    --data "$DATA" \
    --cfg "$model_cfg" \
    --weights weights/pretrained/yolov5m.pt \
    --hyp data/hyps/hyp.tinyperson.scratch.yaml \
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

log "===== 1/2: baseline, ratio=16 ====="
run_arm tinyperson_yolov5m_ratio16 models/cfg/esod/tinyperson_yolov5m_ratio16.yaml

log "===== 2/2: full-spectral concat, ratio=16 ====="
run_arm tinyperson_yolov5m_concat_ratio16 models/cfg/esod/tinyperson_yolov5m_concat_ratio16.yaml

log "All done."
log "  ratio16 baseline: $RUN_ROOT/test/tinyperson_yolov5m_ratio16/"
log "  ratio16 concat:   $RUN_ROOT/test/tinyperson_yolov5m_concat_ratio16/"

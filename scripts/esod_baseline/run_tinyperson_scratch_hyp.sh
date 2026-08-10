#!/usr/bin/env bash
# Re-run TinyPerson with data/hyps/hyp.tinyperson.scratch.yaml instead of
# hyp.tinyperson.finetune.yaml (see HESOD-Experiment-Plan.md's TinyPerson
# section for why finetune was likely the wrong choice -- VisDrone/UAVDT both
# use scratch-style hyps despite also initializing from the pretrained COCO
# yolov5m checkpoint, the same precondition originally used to justify
# "finetune" for TinyPerson).
#
# Runs two arms back to back on the same GPU (sequential, not parallel --
# both need the whole card):
#   1. baseline (plain Segmenter, upstream loss) via run_baseline.sh, which
#      now defaults to the scratch hyp -- run name tinyperson_yolov5m_baseline
#      (overwrites the old finetune-hyp run; that run's numbers are already
#      recorded in HESOD-Experiment-Plan.md, nothing is lost).
#   2. channel-pooled dual-evidence concat, same recipe as the VisDrone roster
#      arm (--selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0), same
#      corrected scratch hyp, via hesod/backends/hesod (the only tree with
#      ChannelPooledConcatEvidenceSegmenter / --selector-loss) -- run name
#      tinyperson_yolov5m_channel_pooled_concat.
#
# Usage: bash run_tinyperson_scratch_hyp.sh [gpu_index]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
HESOD_DEV="${HESOD_DEV:-$SCRIPT_DIR/../../hesod/backends/hesod}"
DATA="${DATA:-/root/autodl-tmp/TinyPerson/tinyperson.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/TinyPerson}"
IMG_SIZE=2048
BATCH=8
EPOCHS="${EPOCHS:-50}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "===== 1/2: baseline, scratch hyp (via run_baseline.sh) ====="
GPU="$GPU" RUN_ROOT="$RUN_ROOT" EPOCHS="$EPOCHS" bash "$SCRIPT_DIR/run_baseline.sh" tinyperson "$GPU"

log "===== 2/2: channel-pooled dual-evidence concat, scratch hyp ====="
run_name="tinyperson_yolov5m_channel_pooled_concat"
results_dir="$RUN_ROOT/test/$run_name"
mkdir -p "$results_dir"

cd "$HESOD_DEV"

log "Training $run_name -> $results_dir/${run_name}_train.log"
python train.py \
  --data "$DATA" \
  --cfg models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat.yaml \
  --weights weights/pretrained/yolov5m.pt \
  --hyp data/hyps/hyp.tinyperson.scratch.yaml \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 \
  --batch-size $BATCH --img-size $IMG_SIZE --epochs "$EPOCHS" --device "$GPU" \
  --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
  2>&1 | tee "$results_dir/${run_name}_train.log"

ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"

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

log "All done. Baseline: $RUN_ROOT/test/tinyperson_yolov5m_baseline/  Channel-pooled: $results_dir/"

#!/usr/bin/env bash
# VisDrone baseline retrained with SAM-hybrid pseudo-masks instead of the
# Gaussian-only fallback every arm in this project has used so far. Isolates
# exactly one variable vs. the SS9 baseline (HESOD-Experiment-Plan.md SS1):
# same VisDrone_v2 data, same cfg/hyp/img-size/epochs, same hesod/backends/esod
# tree -- only mask generation changes.
#
# Prerequisite (run once, NOT done by this script):
#   cd hesod/backends/esod/third_party/segment-anything && pip install -e .
#   mkdir -p hesod/backends/esod/weights
#   cd hesod/backends/esod/weights
#   wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
#   cd scripts/esod_baseline
#   python gen_masks.py --esod-repo ../../hesod/backends/esod \
#     --dataset-root /root/autodl-tmp/VisDrone_v2 --splits train val \
#     --cls-ratio --overwrite
# data_prepare.py's SAM import is a bare try/except at module load time (see
# its own top-of-file code) -- once segment-anything is installed and the
# checkpoint is at hesod/backends/esod/weights/sam_vit_h_4b8939.pth,
# gen_mask() picks up SAM automatically, no flag needed. --overwrite is
# required since VisDrone_v2/masks/*.npy already exist (Gaussian-only, from
# the original SS9 run) and gen_masks.py skips existing files by default.
#
# Usage: bash run_visdrone_sam.sh [gpu_index]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
ESOD_REPO="${ESOD_REPO:-$SCRIPT_DIR/../../hesod/backends/esod}"
DATA="${DATA:-/root/autodl-tmp/VisDrone_v2.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/VisDrone_v2}"
IMG_SIZE=1536
BATCH=8
EPOCHS="${EPOCHS:-50}"
RUN_NAME="visdrone_yolov5m_sam_masks"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Sanity check: fail loudly if masks weren't actually regenerated with SAM --
# a silent Gaussian-only retrain would waste a 50-epoch run and look like a
# null result for the wrong reason.
if [ ! -f "$ESOD_REPO/weights/sam_vit_h_4b8939.pth" ]; then
  echo "ERROR: $ESOD_REPO/weights/sam_vit_h_4b8939.pth not found." >&2
  echo "Run the prerequisite steps in this script's header comment first." >&2
  exit 1
fi

results_dir="$RUN_ROOT/test/$RUN_NAME"
mkdir -p "$results_dir"
cd "$ESOD_REPO"

log "Training $RUN_NAME -> $results_dir/${RUN_NAME}_train.log"
python train.py \
  --data "$DATA" \
  --cfg models/cfg/esod/visdrone_yolov5m.yaml \
  --weights weights/pretrained/yolov5m.pt \
  --hyp data/hyps/hyp.visdrone.yaml \
  --batch-size $BATCH --img-size $IMG_SIZE --epochs "$EPOCHS" --device "$GPU" \
  --project "$RUN_ROOT/train" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$results_dir/${RUN_NAME}_train.log"

ckpt="$RUN_ROOT/train/$RUN_NAME/weights/best.pt"

log "Evaluating $RUN_NAME -> $results_dir/${RUN_NAME}_test.log"
python test.py \
  --data "$DATA" --weights "$ckpt" \
  --batch-size $BATCH --img-size $IMG_SIZE --device "$GPU" --save-json \
  --project "$RUN_ROOT/test" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$results_dir/${RUN_NAME}_test.log"

log "Measuring $RUN_NAME (GFLOPs/FPS, batch=1) -> $results_dir/${RUN_NAME}_measure.log"
python test.py \
  --data "$DATA" --weights "$ckpt" \
  --batch-size 1 --img-size $IMG_SIZE --device "$GPU" --task measure \
  --project "$RUN_ROOT/measure" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$results_dir/${RUN_NAME}_measure.log"

log "Auditing $RUN_NAME -> $results_dir/${RUN_NAME}_audit.log"
python "$SCRIPT_DIR/audit_buckets.py" \
  --pred "$results_dir/best_predictions.json" \
  --labels "$DATASET_ROOT/labels/val" --images "$DATASET_ROOT/images/val" \
  2>&1 | tee "$results_dir/${RUN_NAME}_audit.log"

log "Done. $results_dir/"

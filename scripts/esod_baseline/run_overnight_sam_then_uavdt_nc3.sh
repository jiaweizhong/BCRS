#!/usr/bin/env bash
# Overnight queue: VisDrone + SAM masks, then UAVDT nc=3 re-conversion + train.
# Single GPU, strictly sequential (&&-chained) -- a training job can't share a
# GPU with another training job, only with non-GPU work like list_examples.py.
#
# Prerequisite (should already be done, not re-checked here):
#   segment-anything importable + hesod/backends/esod/weights/sam_vit_h_4b8939.pth
#   present -- see ESOD-Baseline-Patches.md #8 if not.
#
# Usage: nohup bash run_overnight_sam_then_uavdt_nc3.sh > ~/overnight.log 2>&1 &
#        (or run inside tmux/screen -- either way, don't rely on the SSH
#        session staying connected for hours)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ============================================================
# Part 1: VisDrone + SAM masks (HESOD-Experiment-Plan.md, SAM
# experiment vs. SS9's Gaussian-only baseline)
# ============================================================
log "===== PART 1/2: VisDrone + SAM ====="

log "Regenerating VisDrone_v2 masks with SAM (overwrites Gaussian-only masks)"
cd "$SCRIPT_DIR"
python gen_masks.py \
  --esod-repo "$SCRIPT_DIR/../../hesod/backends/esod" \
  --dataset-root /root/autodl-tmp/VisDrone_v2 \
  --splits train val \
  --cls-ratio --overwrite

log "Training VisDrone SAM-masks baseline"
bash "$SCRIPT_DIR/run_visdrone_sam.sh" "$GPU"

# ============================================================
# Part 2: UAVDT nc=3 (car/truck/bus, not collapsed to class 0)
# HESOD-Experiment-Plan.md SS5 / ESOD-Baseline-Patches.md #6
# ============================================================
log "===== PART 2/2: UAVDT nc=3 ====="

ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/esod"

log "Re-converting UAVDT_raw with --keep-classes (car=0/truck=1/bus=2)"
cd "$ESOD_REPO"
python scripts/data_prepare.py --dataset /root/autodl-tmp/UAVDT_raw --keep-classes

log "Reorganizing into UAVDT_v3"
cd "$SCRIPT_DIR"
python reorganize_uavdt.py \
  --raw-root /root/autodl-tmp/UAVDT_raw \
  --out-root /root/autodl-tmp/UAVDT_v3

log "Writing UAVDT_v3.yaml"
cat > /root/autodl-tmp/UAVDT_v3.yaml << 'YAML'
train: /root/autodl-tmp/UAVDT_v3/split/train_ds.txt
val: /root/autodl-tmp/UAVDT_v3/images/test
test: /root/autodl-tmp/UAVDT_v3/images/test
nc: 3
names: ['car', 'truck', 'bus']
YAML

RUN_NAME="uavdt_yolov5m_nc3"
results_dir="$RUN_ROOT/test/$RUN_NAME"
mkdir -p "$results_dir"
cd "$ESOD_REPO"

log "Training $RUN_NAME"
python train.py \
  --data /root/autodl-tmp/UAVDT_v3.yaml \
  --cfg models/cfg/esod/uavdt_yolov5m_nc3.yaml \
  --weights weights/pretrained/yolov5m.pt \
  --hyp data/hyps/hyp.uavdt.yaml \
  --batch-size 8 --img-size 1280 --epochs 50 --device "$GPU" \
  --project "$RUN_ROOT/train" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$results_dir/${RUN_NAME}_train.log"

ckpt="$RUN_ROOT/train/$RUN_NAME/weights/best.pt"

log "Evaluating $RUN_NAME"
python test.py \
  --data /root/autodl-tmp/UAVDT_v3.yaml --weights "$ckpt" \
  --batch-size 8 --img-size 1280 --device "$GPU" --save-json \
  --project "$RUN_ROOT/test" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$results_dir/${RUN_NAME}_test.log"

log "Measuring $RUN_NAME (GFLOPs/FPS, batch=1)"
python test.py \
  --data /root/autodl-tmp/UAVDT_v3.yaml --weights "$ckpt" \
  --batch-size 1 --img-size 1280 --device "$GPU" --task measure \
  --project "$RUN_ROOT/measure" --name "$RUN_NAME" --exist-ok \
  2>&1 | tee "$results_dir/${RUN_NAME}_measure.log"

log "Auditing $RUN_NAME"
python "$SCRIPT_DIR/audit_buckets.py" \
  --pred "$results_dir/best_predictions.json" \
  --labels /root/autodl-tmp/UAVDT_v3/labels/test --images /root/autodl-tmp/UAVDT_v3/images/test \
  --classes car,truck,bus \
  2>&1 | tee "$results_dir/${RUN_NAME}_audit.log"

log "===== ALL DONE ====="
log "  VisDrone SAM: $RUN_ROOT/test/visdrone_yolov5m_sam_masks/"
log "  UAVDT nc=3:   $results_dir/"

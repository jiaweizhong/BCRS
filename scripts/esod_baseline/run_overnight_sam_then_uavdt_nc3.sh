#!/usr/bin/env bash
# Overnight queue: UAVDT nc=3 first, then VisDrone + SAM masks (redo).
# Single GPU, strictly sequential (&&-chained) -- a training job can't share a
# GPU with another training job, only with non-GPU work like list_examples.py.
#
# Reordered + hardened after the first attempt silently trained
# visdrone_yolov5m_sam_masks on 2-day-old Gaussian-only masks (gen_masks.py's
# --overwrite step never actually touched the .npy files -- confirmed via
# mtime: masks dated 2026-08-09 08:45, training completed 2026-08-11 19:06,
# metrics statistically identical to the Gaussian-only baseline). Root cause
# of *why* gen_masks.py didn't run was not pinned down (script logic itself
# looks correct on inspection) -- this version defends against a repeat by
# verifying mtimes actually changed before committing GPU time to training.
#
# Also newly handles a confound: SAM's `predictor` in data_prepare.py is a
# module-level global set once at import, so once SAM is installed, EVERY
# gen_mask() call in a process picks it up automatically -- including inside
# prepare_uavdt(), which would silently give UAVDT_v3 SAM-hybrid masks while
# the existing nc=1 baseline it's compared against used Gaussian-only. UAVDT
# nc=3 is only meant to test the class-count hypothesis (see
# ESOD-Baseline-Patches.md #6) -- keep it single-variable by temporarily
# renaming the SAM checkpoint during UAVDT's data-prep step only.
#
# Prerequisite (should already be done, not re-checked here):
#   segment-anything importable + hesod/backends/esod/weights/sam_vit_h_4b8939.pth
#   present -- see ESOD-Baseline-Patches.md #8 if not.
#
# Usage: nohup bash run_overnight_sam_then_uavdt_nc3.sh > /root/overnight2.log 2>&1 &
#        (or run inside tmux/screen -- either way, don't rely on the SSH
#        session staying connected for hours)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/esod"
SAM_CKPT="$ESOD_REPO/weights/sam_vit_h_4b8939.pth"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ============================================================
# Part 1: UAVDT nc=3 (car/truck/bus, not collapsed to class 0)
# HESOD-Experiment-Plan.md SS5 / ESOD-Baseline-Patches.md #6
# SAM kept OUT of this part -- see header note.
# ============================================================
log "===== PART 1/2: UAVDT nc=3 ====="

if [ -f "$SAM_CKPT" ]; then
  log "Temporarily disabling SAM checkpoint so prepare_uavdt() falls back to Gaussian-only (keeps nc=1-vs-nc=3 single-variable)"
  mv "$SAM_CKPT" "$SAM_CKPT.bak"
  RESTORE_SAM=1
else
  RESTORE_SAM=0
fi

log "Re-converting UAVDT_raw with --keep-classes (car=0/truck=1/bus=2)"
cd "$ESOD_REPO"
python scripts/data_prepare.py --dataset /root/autodl-tmp/UAVDT_raw --keep-classes

if [ "$RESTORE_SAM" = "1" ]; then
  log "Restoring SAM checkpoint"
  mv "$SAM_CKPT.bak" "$SAM_CKPT"
fi

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

# ============================================================
# Part 2: VisDrone + SAM masks, redone with verification this
# time (HESOD-Experiment-Plan.md SAM experiment vs. SS9's
# Gaussian-only baseline). Do NOT skip the mtime check below --
# that's exactly the step whose silent failure wasted the first
# attempt's GPU time.
# ============================================================
log "===== PART 2/2: VisDrone + SAM (redo) ====="

log "Regenerating VisDrone_v2 masks with SAM (overwrites Gaussian-only masks)"
cd "$SCRIPT_DIR"
python gen_masks.py \
  --esod-repo "$ESOD_REPO" \
  --dataset-root /root/autodl-tmp/VisDrone_v2 \
  --splits train val \
  --cls-ratio --overwrite

log "Verifying masks actually got touched just now (guards against a repeat of the silent-stale-mask failure)"
touched=$(find /root/autodl-tmp/VisDrone_v2/masks/train -name "*.npy" -newermt "10 minutes ago" | wc -l)
total=$(find /root/autodl-tmp/VisDrone_v2/masks/train -name "*.npy" | wc -l)
log "  $touched / $total train masks touched in the last 10 minutes"
if [ "$touched" -lt "$((total / 2))" ]; then
  log "FATAL: fewer than half the masks were regenerated -- aborting before wasting a training run on stale masks"
  exit 1
fi

log "Training VisDrone SAM-masks baseline"
bash "$SCRIPT_DIR/run_visdrone_sam.sh" "$GPU"

log "===== ALL DONE ====="
log "  UAVDT nc=3:   $results_dir/"
log "  VisDrone SAM: $RUN_ROOT/test/visdrone_yolov5m_sam_masks/"

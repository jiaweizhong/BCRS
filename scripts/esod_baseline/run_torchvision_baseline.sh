#!/usr/bin/env bash
# torchvision competitor baselines (Faster R-CNN, RetinaNet) for UAVDT +
# SeaPerson, via hesod/backends/baseline/. See that directory's README.md
# for protocol notes (resolution/metric/GFLOPs caveats vs the YOLOv5 arms
# in run_uavdt.sh/run_seaperson.sh -- this script mirrors their run_arm()
# resume pattern but does NOT reuse their scripts, since the model/backend
# is entirely different).
#
# Prerequisite: none beyond what run_uavdt.sh/run_seaperson.sh already
# need -- reads the SAME images/{split}/labels/{split} directories those
# scripts already produced via reorganize_uavdt.py / reorganize_seaperson.py.
#
# Usage:
#   bash run_torchvision_baseline.sh [gpu]
#   ARMS="uavdt_fasterrcnn" bash run_torchvision_baseline.sh 0
#   nohup bash run_torchvision_baseline.sh > /root/torchvision_baseline.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BASELINE_REPO="$SCRIPT_DIR/../../hesod/backends/baseline"
EPOCHS="${EPOCHS:-50}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [TORCHVISION] $*"; }

completed_epochs() {
  local results_file="$1/results.txt"
  if [ ! -f "$results_file" ]; then
    echo 0
    return
  fi
  awk 'NF { count++ } END { print count + 0 }' "$results_file"
}

# Same ARMS="name1,name2" allowlist convention as run_seaperson.sh.
ARMS="${ARMS:-}"

run_arm() {
  local run_name="$1" model="$2"
  local train_images="$3" train_labels="$4"
  local val_images="$5" val_labels="$6"
  local test_images="$7" test_labels="$8"
  local classes="$9" img_size="${10}" batch="${11}"

  if [ -n "$ARMS" ]; then
    case ",$ARMS," in
      *",$run_name,"*) ;;
      *) log "===== $run_name not in \$ARMS, skipping ====="; return 0 ;;
    esac
  fi

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$BASELINE_REPO"

  local train_dir="$RUN_ROOT/train/$run_name"
  local ckpt="$train_dir/weights/best.pt"
  local last_ckpt="$train_dir/weights/last.pt"
  local done_epochs
  done_epochs="$(completed_epochs "$train_dir")"
  local training_was_already_done=0

  if [ "$done_epochs" -ge "$EPOCHS" ] && [ -f "$ckpt" ]; then
    log "===== $run_name already completed $done_epochs/$EPOCHS epochs, skipping training ====="
    training_was_already_done=1
  elif [ -f "$last_ckpt" ]; then
    log "===== Resuming $run_name from $last_ckpt ($done_epochs/$EPOCHS epochs completed) ====="
    python train.py --resume "$last_ckpt" \
      2>&1 | tee -a "$results_dir/${run_name}_train.log"
  elif [ -f "$ckpt" ] || [ "$done_epochs" -gt 0 ]; then
    log "FATAL: $run_name is incomplete ($done_epochs/$EPOCHS epochs) but $last_ckpt is missing"
    exit 1
  else
    log "===== Training $run_name (model=$model, img-size=$img_size, batch=$batch) ====="
    python train.py \
      --model "$model" \
      --train-images-dir "$train_images" --train-labels-dir "$train_labels" \
      --val-images-dir "$val_images" --val-labels-dir "$val_labels" \
      --classes "$classes" --epochs "$EPOCHS" --batch-size "$batch" --img-size "$img_size" \
      --device "$GPU" --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_train.log"
  fi

  done_epochs="$(completed_epochs "$train_dir")"
  if [ "$done_epochs" -lt "$EPOCHS" ]; then
    log "FATAL: $run_name has only $done_epochs/$EPOCHS completed epochs after training/resume"
    exit 1
  fi
  if [ ! -f "$ckpt" ]; then
    log "FATAL: $run_name training finished but $ckpt does not exist -- aborting before eval"
    exit 1
  fi

  if [ "$training_was_already_done" = "1" ] \
    && [ -f "$results_dir/best_predictions.json" ] \
    && [ -f "$RUN_ROOT/measure/$run_name/measure.json" ]; then
    log "===== $run_name eval + measure already complete, skipping ====="
  else
    log "Evaluating $run_name"
    python test.py \
      --weights "$ckpt" --images-dir "$test_images" --labels-dir "$test_labels" \
      --task test --batch-size "$batch" --device "$GPU" --save-json \
      --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_test.log"

    log "Measuring $run_name (GFLOPs/FPS, batch=1)"
    python test.py \
      --weights "$ckpt" --images-dir "$test_images" --labels-dir "$test_labels" \
      --task measure --device "$GPU" \
      --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_measure.log"
  fi

  if [ ! -f "$results_dir/best_predictions.json" ]; then
    log "WARNING: $run_name produced no predictions -- skipping audit/vt_diagnose"
  else
    log "Auditing $run_name"
    python "$SCRIPT_DIR/audit_buckets.py" \
      --pred "$results_dir/best_predictions.json" \
      --labels "$test_labels" --images "$test_images" \
      --classes "$classes" \
      2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed, continuing"

    log "vt_diagnose: $run_name"
    python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
      --labels-dir "$test_labels" --images-dir "$test_images" \
      --classes "$classes" \
      2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed, continuing"
  fi
}

# ---- UAVDT ----
# No genuine held-out valid split (reorganize_uavdt.py only produces
# train/test) -- val during training reuses images/test, same as the
# existing YOLOv5 arms' own val: convention for this dataset (inherited,
# not introduced here; see backends/baseline/README.md).
#
# UAVDT_fresh, not UAVDT_v3: this project's active hesod/backends/hesod
# tree runs UAVDT against UAVDT_fresh (re-extracted from raw source
# archives, HESOD-Experiment-Plan.md SS7) -- UAVDT_v3 is the older
# frozen-tree run_baseline.sh naming and was never confirmed to have a
# reorganized labels/train on this box (confirmed missing entirely on
# 2026-08-23: `ls /root/autodl-tmp/UAVDT_v3/labels/` -> No such file or
# directory; only UAVDT_fresh/ exists under /root/autodl-tmp).
UAVDT_ROOT="${UAVDT_ROOT:-/root/autodl-tmp/UAVDT_fresh}"
run_arm "uavdt_fasterrcnn" fasterrcnn \
  "$UAVDT_ROOT/images/train" "$UAVDT_ROOT/labels/train" \
  "$UAVDT_ROOT/images/test" "$UAVDT_ROOT/labels/test" \
  "$UAVDT_ROOT/images/test" "$UAVDT_ROOT/labels/test" \
  "car,truck,bus" 1280 8

run_arm "uavdt_retinanet" retinanet \
  "$UAVDT_ROOT/images/train" "$UAVDT_ROOT/labels/train" \
  "$UAVDT_ROOT/images/test" "$UAVDT_ROOT/labels/test" \
  "$UAVDT_ROOT/images/test" "$UAVDT_ROOT/labels/test" \
  "car,truck,bus" 1280 8

# ---- SeaPerson (aka TinyPersonV2) ----
# Genuine 3-way split; valid is real held-out data, never touched by test.
# Batch=2, not 4: confirmed OOM'd at batch=4 on this exact GPU on
# 2026-08-23 (seaperson_fasterrcnn, mid-epoch-1, "CUDA out of memory. Tried
# to allocate 1.74 GiB... 29.89 GiB in use"). 2048px images through a
# two-stage/dense-anchor ResNet50-FPN detector is far more memory-hungry
# than YOLOv5m at the same resolution -- same class of issue, and same
# resolution (batch=2), as the spectral-only OOM saga (SS8). Raise only
# after confirming batch=2 comfortably fits with headroom to spare.
SEAPERSON_ROOT="${SEAPERSON_ROOT:-/root/autodl-tmp/seaperson_v2}"
run_arm "seaperson_fasterrcnn" fasterrcnn \
  "$SEAPERSON_ROOT/images/train" "$SEAPERSON_ROOT/labels/train" \
  "$SEAPERSON_ROOT/images/valid" "$SEAPERSON_ROOT/labels/valid" \
  "$SEAPERSON_ROOT/images/test" "$SEAPERSON_ROOT/labels/test" \
  "person" 2048 2

run_arm "seaperson_retinanet" retinanet \
  "$SEAPERSON_ROOT/images/train" "$SEAPERSON_ROOT/labels/train" \
  "$SEAPERSON_ROOT/images/valid" "$SEAPERSON_ROOT/labels/valid" \
  "$SEAPERSON_ROOT/images/test" "$SEAPERSON_ROOT/labels/test" \
  "person" 2048 2

log "===== ALL DONE ====="
log "  UAVDT Faster R-CNN:     $RUN_ROOT/test/uavdt_fasterrcnn/"
log "  UAVDT RetinaNet:        $RUN_ROOT/test/uavdt_retinanet/"
log "  SeaPerson Faster R-CNN: $RUN_ROOT/test/seaperson_fasterrcnn/"
log "  SeaPerson RetinaNet:    $RUN_ROOT/test/seaperson_retinanet/"

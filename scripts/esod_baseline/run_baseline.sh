#!/usr/bin/env bash
# Reproduce the official ESOD (YOLOv5m) baseline on VisDrone / UAVDT / TinyPerson
# using the pristine official repo (NOT the BCRS vendored/modified fork).
#
# Usage:
#   ./run_baseline.sh <visdrone|uavdt|tinyperson|all> [gpu_index]
#
# Config via env vars (all optional):
#   ESOD_REPO   path to the official alibaba/esod checkout (default: $HOME/BCRS/esod)
#   GPU         cuda device index (default: 0, overridden by $2)
#   EPOCHS      training epochs (default: 50, matches the paper)
#   RUN_ROOT    where logs/checkpoints go (default: $HOME/esod_baseline_runs)
#   SKIP_MASKS  set to 1 to skip mask generation/verification (dangerous: only
#               do this if you already know masks/<split>/*.npy exist for every
#               labeled image)
#
# Each dataset run: generate+verify masks -> download yolov5m.pt if missing ->
# train.py (50 epochs, paper hyperparameters) -> test.py (AP/AP50) ->
# test.py --task measure (GFLOPs/FPS). Every step's stdout+stderr is tee'd to
# $RUN_ROOT/logs/, and the checkpoint path is fixed via --exist-ok so reruns
# are idempotent.
#
# Run this under tmux/screen/nohup -- a dropped SSH session must not kill an
# overnight training run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ESOD_REPO="${ESOD_REPO:-$HOME/BCRS/esod}"
EPOCHS="${EPOCHS:-50}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
SKIP_MASKS="${SKIP_MASKS:-0}"
GPU="${GPU:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

require_file() {
  if [ ! -f "$1" ]; then
    log "ERROR: required file not found: $1"
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    log "ERROR: required directory not found: $1"
    exit 1
  fi
}

ensure_weights() {
  local weights_dir="$ESOD_REPO/weights/pretrained"
  local weights_file="$weights_dir/yolov5m.pt"
  mkdir -p "$weights_dir"
  if [ ! -f "$weights_file" ]; then
    log "yolov5m.pt not found, downloading pretrained COCO init ..."
    curl -fL -o "$weights_file" \
      "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt"
  fi
  require_file "$weights_file"
}

gen_and_verify_masks() {
  local dataset_root="$1"
  shift
  local extra_flags=("$@")
  log "Generating/verifying ESOD pseudo-masks under $dataset_root ..."
  python "$SCRIPT_DIR/gen_masks.py" \
    --esod-repo "$ESOD_REPO" \
    --dataset-root "$dataset_root" \
    --splits train val \
    "${extra_flags[@]}"
}

run_dataset() {
  local name="$1" data_yaml="$2" model_cfg="$3" hyp="$4" img_size="$5" batch="$6" dataset_root="$7"
  shift 7
  local mask_extra_flags=("$@")

  log "===== $name: baseline reproduction start (GPU=$GPU, epochs=$EPOCHS, img=$img_size, batch=$batch) ====="
  require_file "$data_yaml"
  require_file "$ESOD_REPO/$model_cfg"
  require_file "$ESOD_REPO/$hyp"
  require_dir "$dataset_root/images"
  require_dir "$dataset_root/labels"

  if [ "$SKIP_MASKS" != "1" ]; then
    gen_and_verify_masks "$dataset_root" "${mask_extra_flags[@]}"
  else
    log "SKIP_MASKS=1: trusting existing masks/*.npy without checking"
  fi

  ensure_weights

  mkdir -p "$RUN_ROOT/logs"
  local run_name="${name}_yolov5m_baseline"
  local train_log="$RUN_ROOT/logs/${run_name}_train.log"
  local test_log="$RUN_ROOT/logs/${run_name}_test.log"
  local measure_log="$RUN_ROOT/logs/${run_name}_measure.log"

  cd "$ESOD_REPO"

  log "Training $name -> $train_log"
  python train.py \
    --data "$data_yaml" \
    --cfg "$model_cfg" \
    --weights weights/pretrained/yolov5m.pt \
    --hyp "$hyp" \
    --batch-size "$batch" \
    --img-size "$img_size" \
    --epochs "$EPOCHS" \
    --device "$GPU" \
    --project "$RUN_ROOT/train" \
    --name "$run_name" \
    --exist-ok \
    2>&1 | tee "$train_log"

  local ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"
  require_file "$ckpt"

  log "Evaluating $name (AP/AP50, vanilla) -> $test_log"
  python test.py \
    --data "$data_yaml" \
    --weights "$ckpt" \
    --batch-size "$batch" \
    --img-size "$img_size" \
    --device "$GPU" \
    --project "$RUN_ROOT/test" \
    --name "$run_name" \
    --exist-ok \
    2>&1 | tee "$test_log"

  log "Measuring $name (GFLOPs/FPS, batch=1) -> $measure_log"
  python test.py \
    --data "$data_yaml" \
    --weights "$ckpt" \
    --batch-size 1 \
    --img-size "$img_size" \
    --device "$GPU" \
    --task measure \
    --project "$RUN_ROOT/measure" \
    --name "$run_name" \
    --exist-ok \
    2>&1 | tee "$measure_log"

  log "===== $name: done ====="
  log "---- $name test.py tail (AP/AP50) ----"
  tail -n 20 "$test_log"
  log "---- $name measure tail (GFLOPs/FPS) ----"
  tail -n 20 "$measure_log"
}

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
  echo "Usage: $0 <visdrone|uavdt|tinyperson|all> [gpu_index]"
  exit 1
fi
if [ -n "${2:-}" ]; then
  GPU="$2"
fi

case "$DATASET" in
  visdrone)
    run_dataset visdrone \
      /root/autodl-tmp/VisDrone.yaml \
      models/cfg/esod/visdrone_yolov5m.yaml \
      data/hyps/hyp.visdrone.yaml \
      1536 8 \
      /root/autodl-tmp/VisDrone \
      --cls-ratio
    ;;
  uavdt)
    run_dataset uavdt \
      /root/autodl-tmp/UAVDT_processed/uavdt.yaml \
      models/cfg/esod/uavdt_yolov5m.yaml \
      data/hyps/hyp.uavdt.yaml \
      1280 8 \
      /root/autodl-tmp/UAVDT_processed
    ;;
  tinyperson)
    # NOTE: the official repo ships hyp.tinyperson.finetune.yaml and
    # hyp.tinyperson.scratch.yaml, but no plain hyp.tinyperson.yaml, so
    # scripts/train.sh's `data/hyps/hyp.${DATASET}.yaml` default would 404 here.
    # We init from the pretrained COCO yolov5m checkpoint (same as the other two
    # datasets), so the "finetune" hyp set is used. This is an assumption, not
    # confirmed against the paper's supplementary code -- sanity-check the
    # resulting AP against Table II (APt50 61.3 / APs50 74.4) before trusting it.
    run_dataset tinyperson \
      /root/autodl-tmp/TinyPerson/tinyperson.yaml \
      models/cfg/esod/tinyperson_yolov5m.yaml \
      data/hyps/hyp.tinyperson.finetune.yaml \
      2048 8 \
      /root/autodl-tmp/TinyPerson
    ;;
  all)
    "$0" visdrone "$GPU"
    "$0" uavdt "$GPU"
    "$0" tinyperson "$GPU"
    ;;
  *)
    echo "Unknown dataset: $DATASET (expected visdrone|uavdt|tinyperson|all)"
    exit 1
    ;;
esac

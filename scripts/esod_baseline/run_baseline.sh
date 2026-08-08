#!/usr/bin/env bash
# Reproduce the official ESOD (YOLOv5m) baseline on VisDrone / UAVDT / TinyPerson
# using the pristine official repo (NOT the BCRS vendored/modified fork).
#
# Usage:
#   ./run_baseline.sh <visdrone|uavdt|tinyperson|all> [gpu_index]
#
# Config via env vars (all optional):
#   ESOD_REPO     path to the official alibaba/esod checkout (default: $HOME/BCRS/esod)
#   GPU           cuda device index (default: 0, overridden by $2)
#   EPOCHS        training epochs (default: 50, matches the paper)
#   RUN_ROOT      where logs/checkpoints go (default: $HOME/esod_baseline_runs)
#   SKIP_MASKS    set to 1 to skip mask generation/verification (dangerous: only
#                 do this if you already know masks/<split>/*.npy exist for every
#                 labeled image)
#   AUDIT_SCRIPT  path to the bucket-recall auditor (default: audit_buckets.py
#                 next to this script -- standalone, no BCRS package dependency)
#   SKIP_AUDIT    set to 1 to skip the per-bucket recall audit step
#
# Each dataset run: generate+verify masks -> download yolov5m.pt if missing ->
# train.py (50 epochs, paper hyperparameters) -> test.py (AP/AP50, --save-json)
# -> test.py --task measure (GFLOPs/FPS) -> audit_failure_cases.py (size/class
# bucket recall on the saved predictions). Every step's stdout+stderr is tee'd
# to $RUN_ROOT/logs/, and the checkpoint path is fixed via --exist-ok so
# reruns are idempotent.
#
# Run this under tmux/screen/nohup -- a dropped SSH session must not kill an
# overnight training run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ESOD_REPO="${ESOD_REPO:-$HOME/BCRS/esod}"
EPOCHS="${EPOCHS:-50}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
SKIP_MASKS="${SKIP_MASKS:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-$SCRIPT_DIR/audit_buckets.py}"
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

# Per-size-bin / per-class recall audit on the saved val predictions. Never
# allowed to abort the whole run: this is a diagnostic on top of an already
#-completed baseline, not a prerequisite for having reproduced it.
run_bucket_audit() {
  local name="$1" dataset_root="$2" pred_json="$3" classes_csv="$4" audit_log="$5"

  if [ "$SKIP_AUDIT" = "1" ]; then
    log "SKIP_AUDIT=1: skipping bucket recall audit for $name"
    return 0
  fi
  if [ ! -f "$AUDIT_SCRIPT" ]; then
    log "WARN: audit script not found at $AUDIT_SCRIPT, skipping bucket audit for $name"
    return 0
  fi
  if [ ! -f "$pred_json" ]; then
    log "WARN: $pred_json was not produced (test.py --save-json likely wrote 0 predictions), skipping bucket audit for $name"
    return 0
  fi

  local classes_args=()
  if [ -n "$classes_csv" ]; then
    classes_args=(--classes "$classes_csv")
  fi

  log "Auditing $name per-size/per-class recall -> $audit_log"
  set +e
  python "$AUDIT_SCRIPT" \
    --pred "$pred_json" \
    --labels "$dataset_root/labels/val" \
    --images "$dataset_root/images/val" \
    "${classes_args[@]}" \
    2>&1 | tee "$audit_log"
  local audit_status=$?
  set -e
  if [ "$audit_status" -ne 0 ]; then
    log "WARN: bucket audit for $name exited non-zero ($audit_status); see $audit_log"
  fi
}

run_dataset() {
  local name="$1" data_yaml="$2" model_cfg="$3" hyp="$4" img_size="$5" batch="$6" dataset_root="$7" classes_csv="$8"
  shift 8
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
  local audit_log="$RUN_ROOT/logs/${run_name}_audit.log"

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

  log "Evaluating $name (AP/AP50, vanilla, --save-json) -> $test_log"
  python test.py \
    --data "$data_yaml" \
    --weights "$ckpt" \
    --batch-size "$batch" \
    --img-size "$img_size" \
    --device "$GPU" \
    --save-json \
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

  # test.py names it "<weights-stem>_predictions.json" (best.pt -> best_predictions.json)
  local pred_json="$RUN_ROOT/test/$run_name/best_predictions.json"
  run_bucket_audit "$name" "$dataset_root" "$pred_json" "$classes_csv" "$audit_log"

  log "===== $name: done ====="
  log "---- $name test.py tail (AP/AP50) ----"
  tail -n 20 "$test_log"
  log "---- $name measure tail (GFLOPs/FPS) ----"
  tail -n 20 "$measure_log"
  if [ -f "$audit_log" ]; then
    log "---- $name bucket audit tail (size/class recall) ----"
    tail -n 40 "$audit_log"
  fi
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
    # classes_csv left empty -> audit_failure_cases.py uses its built-in 10-class
    # VisDrone default (pedestrian..motor), which matches the labels this pipeline
    # writes (see esod/scripts/data_prepare.py::prepare_visdrone(), cls - 1).
    run_dataset visdrone \
      /root/autodl-tmp/VisDrone.yaml \
      models/cfg/esod/visdrone_yolov5m.yaml \
      data/hyps/hyp.visdrone.yaml \
      1536 8 \
      /root/autodl-tmp/VisDrone \
      "" \
      --cls-ratio
    ;;
  uavdt)
    # NOTE on classes: the official esod/scripts/data_prepare.py::prepare_uavdt()
    # collapses ALL UAVDT categories to a single class 0 ("car") when it writes YOLO
    # labels -- despite the dataset nominally having 3 categories. We don't know
    # whether /root/autodl-tmp/UAVDT_processed/labels/*.txt was produced by that
    # official script (1 class) or by BCRS's own converter (3 classes: car/truck/bus,
    # per BCRS/configs/datasets/uavdt.yaml). "car,truck,bus" below assumes the
    # latter. If your labels only ever contain class id 0, change this to just "car"
    # (or the audit will still run correctly for size buckets -- only the per-class
    # table would be mislabeled).
    run_dataset uavdt \
      /root/autodl-tmp/UAVDT_processed/uavdt.yaml \
      models/cfg/esod/uavdt_yolov5m.yaml \
      data/hyps/hyp.uavdt.yaml \
      1280 8 \
      /root/autodl-tmp/UAVDT_processed \
      "car,truck,bus"
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
      /root/autodl-tmp/TinyPerson \
      "person"
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

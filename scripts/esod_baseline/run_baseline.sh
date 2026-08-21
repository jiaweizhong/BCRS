#!/usr/bin/env bash
# Reproduce the official ESOD (YOLOv5m) baseline on VisDrone / UAVDT / TinyPerson
# via hesod/backends/esod (the frozen ESOD reference copy inside the HESOD
# umbrella -- NOT the BCRS vendored/modified fork. This is the sole baseline
# copy since 2026-08-21; the standalone top-level esod/ checkout that used to
# mirror it was retired once confirmed byte-identical and unreferenced
# elsewhere, see ESOD-Baseline-Patches.md's provenance note.
#
# Usage:
#   ./run_baseline.sh <visdrone|uavdt|tinyperson|all> [gpu_index]
#
# Config via env vars (all optional):
#   ESOD_REPO     path to the ESOD checkout to run (default: hesod/backends/esod,
#                 resolved relative to this script's location)
#   GPU           cuda device index (default: 0, overridden by $2)
#   EPOCHS        training epochs (default: 50, matches the paper)
#   RUN_ROOT      where logs/checkpoints go (default: $HOME/esod_baseline_runs)
#   SKIP_MASKS    set to 1 to skip mask generation/verification (dangerous: only
#                 do this if you already know masks/<split>/*.npy exist for every
#                 labeled image)
#   MASK_MODE     gaussian (default) or released-hybrid. The selected mode is
#                 applied explicitly to all requested datasets.
#   AUDIT_SCRIPT  path to the bucket-recall auditor (default: audit_buckets.py
#                 next to this script -- standalone, no BCRS package dependency)
#   SKIP_AUDIT    set to 1 to skip detector-recall and selector-BPR audits
#
# Each dataset run: generate+verify masks -> download yolov5m.pt if missing ->
# train.py (50 epochs, paper hyperparameters) -> test.py (AP/AP50, --save-json)
# -> test.py --task measure (GFLOPs/FPS) -> detector recall audit -> exact-route
# patch dump -> paper BPRbox audit. Every step's stdout+stderr is tee'd
# to $RUN_ROOT/logs/, and the checkpoint path is fixed via --exist-ok so
# reruns are idempotent.
#
# Run this under tmux/screen/nohup -- a dropped SSH session must not kill an
# overnight training run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ESOD_REPO="${ESOD_REPO:-$SCRIPT_DIR/../../hesod/backends/esod}"
EPOCHS="${EPOCHS:-50}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
SKIP_MASKS="${SKIP_MASKS:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
MASK_MODE="${MASK_MODE:-gaussian}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-$SCRIPT_DIR/audit_buckets.py}"
PATCH_DUMP_SCRIPT="${PATCH_DUMP_SCRIPT:-$SCRIPT_DIR/dump_selected_patches.py}"
SELECTOR_AUDIT_SCRIPT="${SELECTOR_AUDIT_SCRIPT:-$SCRIPT_DIR/audit_selector_coverage.py}"
GPU="${GPU:-0}"

if [ "$MASK_MODE" != "gaussian" ] && [ "$MASK_MODE" != "released-hybrid" ]; then
  echo "ERROR: MASK_MODE must be gaussian or released-hybrid (got: $MASK_MODE)" >&2
  exit 1
fi

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
  local dataset_root="$1" val_split="$2"
  shift 2
  local extra_flags=("$@")
  log "Generating/verifying ESOD pseudo-masks under $dataset_root (splits: train $val_split) ..."
  python "$SCRIPT_DIR/gen_masks.py" \
    --esod-repo "$ESOD_REPO" \
    --dataset-root "$dataset_root" \
    --splits train "$val_split" \
    --mask-mode "$MASK_MODE" \
    --overwrite \
    "${extra_flags[@]}"
}

# Per-size-bin / per-class final-detector recall audit on saved predictions.
# A malformed or mismatched artifact is fatal: a silently wrong recall number
# is worse than no diagnostic at all. SKIP_AUDIT=1 is the explicit opt-out.
run_bucket_audit() {
  local name="$1" dataset_root="$2" val_split="$3" pred_json="$4" classes_csv="$5" audit_log="$6"

  if [ "$SKIP_AUDIT" = "1" ]; then
    log "SKIP_AUDIT=1: skipping bucket recall audit for $name"
    return 0
  fi
  if [ ! -f "$AUDIT_SCRIPT" ]; then
    log "ERROR: audit script not found at $AUDIT_SCRIPT"
    exit 1
  fi
  if [ ! -f "$pred_json" ]; then
    log "ERROR: $pred_json was not produced by test.py --save-json"
    exit 1
  fi

  local classes_args=()
  if [ -n "$classes_csv" ]; then
    classes_args=(--classes "$classes_csv")
  fi

  log "Auditing $name per-size/per-class recall -> $audit_log"
  python "$AUDIT_SCRIPT" \
    --pred "$pred_json" \
    --labels "$dataset_root/labels/$val_split" \
    --images "$dataset_root/images/$val_split" \
    "${classes_args[@]}" \
    2>&1 | tee "$audit_log"
}

run_selector_audit() {
  local name="$1" data_yaml="$2" ckpt="$3" img_size="$4" batch="$5"
  local dataset_root="$6" val_split="$7" pred_json="$8" classes_csv="$9" results_dir="${10}"

  if [ "$SKIP_AUDIT" = "1" ]; then
    return 0
  fi
  require_file "$PATCH_DUMP_SCRIPT"
  require_file "$SELECTOR_AUDIT_SCRIPT"
  require_file "$pred_json"
  local patches_json="$results_dir/${name}_selected_patches.json"
  local selector_log="$results_dir/${name}_selector_audit.log"
  local classes_args=()
  if [ -n "$classes_csv" ]; then
    classes_args=(--classes "$classes_csv")
  fi

  log "Dumping the exact threshold-routed patches for $name"
  python "$PATCH_DUMP_SCRIPT" \
    --esod-repo "$ESOD_REPO" --data "$data_yaml" --weights "$ckpt" \
    --img-size "$img_size" --batch-size "$batch" --device "$GPU" --task val \
    --out "$patches_json"
  log "Auditing $name paper BPRbox and detector recall -> $selector_log"
  python "$SELECTOR_AUDIT_SCRIPT" \
    --patches "$patches_json" --pred "$pred_json" \
    --labels "$dataset_root/labels/$val_split" --images "$dataset_root/images/$val_split" \
    "${classes_args[@]}" 2>&1 | tee "$selector_log"
}

run_dataset() {
  local name="$1" data_yaml="$2" model_cfg="$3" hyp="$4" img_size="$5" batch="$6" dataset_root="$7" classes_csv="$8" val_split="$9"
  shift 9
  local mask_extra_flags=("$@")

  log "===== $name: baseline reproduction start (GPU=$GPU, epochs=$EPOCHS, img=$img_size, batch=$batch, val_split=$val_split) ====="
  require_file "$data_yaml"
  require_file "$ESOD_REPO/$model_cfg"
  require_file "$ESOD_REPO/$hyp"
  require_dir "$dataset_root/images"
  require_dir "$dataset_root/labels"
  require_dir "$dataset_root/images/$val_split"
  require_dir "$dataset_root/labels/$val_split"

  if [ "$SKIP_MASKS" != "1" ]; then
    gen_and_verify_masks "$dataset_root" "$val_split" "${mask_extra_flags[@]}"
  else
    log "SKIP_MASKS=1: trusting existing masks/*.npy without checking"
  fi

  ensure_weights

  local run_name="${name}_yolov5m_baseline"
  if [ "$MASK_MODE" != "gaussian" ]; then
    run_name="${name}_yolov5m_${MASK_MODE//-/_}"
  fi
  # Co-located with everything test.py itself writes here (best_predictions.json,
  # buckets.json, PR/F1 curves, confusion matrix) so one copy of this directory
  # is a complete, self-contained result bundle -- no separate logs/ directory
  # to remember to also copy.
  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  local train_log="$results_dir/${run_name}_train.log"
  local test_log="$results_dir/${run_name}_test.log"
  local measure_log="$results_dir/${run_name}_measure.log"
  local audit_log="$results_dir/${run_name}_audit.log"

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

  local measured_buckets="$RUN_ROOT/measure/$run_name/buckets.json"
  require_file "$measured_buckets"
  cp "$measured_buckets" "$results_dir/buckets.json"

  # test.py names it "<weights-stem>_predictions.json" (best.pt -> best_predictions.json)
  local pred_json="$RUN_ROOT/test/$run_name/best_predictions.json"
  run_bucket_audit "$name" "$dataset_root" "$val_split" "$pred_json" "$classes_csv" "$audit_log"
  run_selector_audit "$run_name" "$data_yaml" "$ckpt" "$img_size" "$batch" \
    "$dataset_root" "$val_split" "$pred_json" "$classes_csv" "$results_dir"

  log "===== $name: done ====="
  log "---- $name test.py tail (AP/AP50) ----"
  tail -n 20 "$test_log"
  log "---- $name measure tail (GFLOPs/FPS) ----"
  tail -n 20 "$measure_log"
  if [ -f "$audit_log" ]; then
    log "---- $name bucket audit tail (size/class recall) ----"
    tail -n 40 "$audit_log"
  fi
  if [ -f "$results_dir/${run_name}_selector_audit.log" ]; then
    log "---- $name selector audit tail (paper BPRbox / detector recall) ----"
    tail -n 20 "$results_dir/${run_name}_selector_audit.log"
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
    # VisDrone_v2 is the canonical ESOD conversion with ignored/others regions
    # masked exactly once during data preparation.
    run_dataset visdrone \
      /root/autodl-tmp/VisDrone_v2.yaml \
      models/cfg/esod/visdrone_yolov5m.yaml \
      data/hyps/hyp.visdrone.yaml \
      1536 8 \
      /root/autodl-tmp/VisDrone_v2 \
      "" \
      val \
      --cls-ratio
    ;;
  uavdt)
    # UAVDT_v3 preserves car/truck/bus. The on-disk validation split is named
    # test because UAVDT has no separate val directory.
    run_dataset uavdt \
      /root/autodl-tmp/UAVDT_v3.yaml \
      models/cfg/esod/uavdt_yolov5m.yaml \
      data/hyps/hyp.uavdt.yaml \
      1280 8 \
      /root/autodl-tmp/UAVDT_v3 \
      "car,truck,bus" \
      test
    ;;
  tinyperson)
    # hyp.tinyperson.yaml is the only supported TinyPerson profile.
    run_dataset tinyperson \
      /root/autodl-tmp/TinyPerson_v1.yaml \
      models/cfg/esod/tinyperson_yolov5m.yaml \
      data/hyps/hyp.tinyperson.yaml \
      2048 8 \
      /root/autodl-tmp/TinyPerson_v1 \
      "person" \
      val
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

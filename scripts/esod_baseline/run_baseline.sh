#!/usr/bin/env bash
# Reproduce the official ESOD (YOLOv5m) baseline on VisDrone / UAVDT / TinyPerson
# via hesod/backends/esod (the frozen ESOD reference copy inside the HESOD
# umbrella -- NOT the BCRS vendored/modified fork, and, as of 2026-08-08, no
# longer the standalone top-level esod/ either; see ESOD-Baseline-Patches.md's
# "Dual-tree coverage" note. Point ESOD_REPO at esod/ if you specifically need
# the standalone copy for some reason.
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

ESOD_REPO="${ESOD_REPO:-$SCRIPT_DIR/../../hesod/backends/esod}"
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
  local dataset_root="$1" val_split="$2"
  shift 2
  local extra_flags=("$@")
  log "Generating/verifying ESOD pseudo-masks under $dataset_root (splits: train $val_split) ..."
  python "$SCRIPT_DIR/gen_masks.py" \
    --esod-repo "$ESOD_REPO" \
    --dataset-root "$dataset_root" \
    --splits train "$val_split" \
    "${extra_flags[@]}"
}

# Per-size-bin / per-class recall audit on the saved val predictions. Never
# allowed to abort the whole run: this is a diagnostic on top of an already
#-completed baseline, not a prerequisite for having reproduced it.
run_bucket_audit() {
  local name="$1" dataset_root="$2" val_split="$3" pred_json="$4" classes_csv="$5" audit_log="$6"

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
    --labels "$dataset_root/labels/$val_split" \
    --images "$dataset_root/images/$val_split" \
    "${classes_args[@]}" \
    2>&1 | tee "$audit_log"
  local audit_status=$?
  set -e
  if [ "$audit_status" -ne 0 ]; then
    log "WARN: bucket audit for $name exited non-zero ($audit_status); see $audit_log"
  fi
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

  # test.py names it "<weights-stem>_predictions.json" (best.pt -> best_predictions.json)
  local pred_json="$RUN_ROOT/test/$run_name/best_predictions.json"
  run_bucket_audit "$name" "$dataset_root" "$val_split" "$pred_json" "$classes_csv" "$audit_log"

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
    # /root/autodl-tmp/VisDrone(.yaml) is SS1's superseded Ultralytics-converted
    # data (HESOD-Experiment-Plan.md SS1.1) -- kept on disk for reference, not
    # deleted, but this case was still pointing at it until this fix. The
    # current reference dataset is VisDrone_v2 (official prepare_visdrone()
    # conversion, closes ~half the AP gap / ~3/4 the AP50 gap vs. SS1). classes_csv
    # left empty -> audit_failure_cases.py uses its built-in 10-class VisDrone
    # default (pedestrian..motor), which matches the labels this pipeline writes
    # (see esod/scripts/data_prepare.py::prepare_visdrone(), cls - 1).
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
    # /root/autodl-tmp/UAVDT_processed (third-party Kaggle repackaging, 3-class
    # car/truck/bus labels) is superseded: root-caused to a labeling defect
    # (oversized GT boxes over parking lots, HESOD-Experiment-Plan.md SS5) and
    # replaced with official-source data (UAV-benchmark-M.zip / M_attr.zip /
    # UAV-benchmark-MOTD_v1.0.zip from the official UAVDT release), converted
    # via the real scripts/data_prepare.py::prepare_uavdt() and
    # scripts/esod_baseline/reorganize_uavdt.py -- see that script's docstring
    # for the raw layout it expects. nc is 1 ("vehicle"), not 3: confirmed by
    # reading prepare_uavdt() itself, it hardcodes every GT box to class 0
    # regardless of raw category; data/uavdt.yaml and uavdt_yolov5m.yaml were
    # both updated to match (see their own local-patch comments).
    # val_split is "test", not "val": uavdt.yaml maps val -> images/test on disk
    # (UAVDT ships train/test, no separate val split). gen_masks.py/audit_buckets.py
    # need the real on-disk name or they silently skip the split entirely.
    run_dataset uavdt \
      /root/autodl-tmp/UAVDT_v2.yaml \
      models/cfg/esod/uavdt_yolov5m.yaml \
      data/hyps/hyp.uavdt.yaml \
      1280 8 \
      /root/autodl-tmp/UAVDT_v2 \
      "vehicle" \
      test
    ;;
  tinyperson)
    # NOTE: the official repo ships hyp.tinyperson.finetune.yaml and
    # hyp.tinyperson.scratch.yaml, but no plain hyp.tinyperson.yaml, so
    # scripts/train.sh's `data/hyps/hyp.${DATASET}.yaml` default would 404 here.
    # Originally picked "finetune" on the reasoning "we init from the pretrained
    # COCO yolov5m checkpoint, so finetune applies" -- but hyp.visdrone.yaml and
    # hyp.uavdt.yaml BOTH also init from that same checkpoint and are both
    # scratch-style (header comment "COCO training from scratch", lr0=0.01);
    # hyp.tinyperson.finetune.yaml's own header is unmodified upstream
    # ultralytics/yolov5 boilerplate ("VOC finetuning", never adapted to
    # TinyPerson), unlike hyp.tinyperson.scratch.yaml (has the ESOD-specific
    # `pixl` mask-loss gain, same family as the other two datasets' hyps). Using
    # "scratch" now for consistency with VisDrone/UAVDT -- see
    # HESOD-Experiment-Plan.md's TinyPerson section for the reasoning and the
    # finetune-hyp run's numbers it's being compared against.
    run_dataset tinyperson \
      /root/autodl-tmp/TinyPerson_v1.yaml \
      models/cfg/esod/tinyperson_yolov5m.yaml \
      data/hyps/hyp.tinyperson.scratch.yaml \
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

#!/usr/bin/env bash
# Frozen-selector fine-tune probe for SeaPerson (2026-09-06), NOT part of
# the 8-arm roster (run_seaperson.sh). Standalone script, does not touch
# run_seaperson.sh, mirrors run_uavdt_frozen_selector.sh's structure.
#
# Motivation: the user decided to unify the flagship recipe's fusion rule
# to Max on BOTH datasets and drop SABL entirely, for a cleaner story (one
# fusion rule, one efficient head, no dataset-specific component list).
# HESOD-Experiment-Plan.md SS3.7/SS4.5 already established two things that
# matter here:
#   1. On UAVDT, jointly training Max+ISPPHead from scratch (arm 12)
#      collapsed hard (-4.37pp Total Recall / -2.5pp BPR vs pure-max arm 9)
#      -- but a *staged* fine-tune (freeze the converged Max-fusion
#      selector, warm-start from arm 9, only train neck+head) recovered to
#      arm 9's own ceiling (arm 14: 0.394 mAP@.5 vs arm 9's 0.395).
#   2. On SeaPerson, arm 9 (Max+SABL+ISPPHead, jointly trained) landed at
#      0.763, below Max-only arm 10's own 0.766 -- the same joint-training
#      interaction cost UAVDT saw, never isolated from "does Max+ISPPHead
#      alone (no SABL) actually reach arm 10's ceiling" here.
# Going straight to staged training (skipping a doomed joint-training
# attempt) tests whether "Max + ISPPHead, no SABL" can reach SeaPerson's
# own Max-fusion ceiling (arm 10's 0.766) while keeping ISPPHead's
# efficiency gain -- the SeaPerson counterpart to UAVDT's arm 14.
#
# IMPORTANT: this does NOT test whether Max can match Concat on SeaPerson
# (arm 5, 0.772/arm 8, 0.774) -- the frozen-selector ceiling is bounded by
# whichever checkpoint's routing gets frozen (SS3.7's own structural-
# ceiling argument), so this run's ceiling is arm 10's 0.766, still below
# Concat. That gap is the accepted, honest cost of unifying on Max for a
# cleaner cross-dataset story, not something this run is meant to close.
#
# Prerequisite: arm 10 (seaperson_yolov5m_channel_pooled_max) must already
# be trained (it is -- part of run_seaperson.sh's roster, HESOD-Experiment-
# Plan.md SS4.2/SS4.3).
#
# Usage:
#   ARMS=seaperson_yolov5m_channel_pooled_max_isphead_frozen nohup bash run_seaperson_frozen_selector.sh > /root/seaperson_frozen.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/seaperson_v2}"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/seaperson.yaml}"
HYP="${HYP:-data/hyps/hyp.seaperson.yaml}"
CLASSES="person"
VAL_SPLIT="test"
EPOCHS="${EPOCHS:-20}"
log_prefix="FROZEN"

# Warm-start checkpoints -- must already exist for whichever arm below is
# actually run (checked inside run_arm(), only when a fresh warm-start
# training is about to happen, not up front, since a single invocation may
# target only one of these two arms via $ARMS).
# Arm (10): pooled Max fusion, no SABL/ISPPHead, coupled head.
ARM10_CKPT="${ARM10_CKPT:-$RUN_ROOT/train/seaperson_yolov5m_channel_pooled_max/weights/best.pt}"
# seaperson_yolov5m_max (2026-09-06, channel-pooling-isolation probe):
# same Max fusion rule, full-width (non-pooled) SpectralBranch instead.
ARM_MAX_CKPT="${ARM_MAX_CKPT:-$RUN_ROOT/train/seaperson_yolov5m_max/weights/best.pt}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$log_prefix] $*"; }

completed_epochs() {
  local results_file="$1/results.txt"
  if [ ! -f "$results_file" ]; then
    echo 0
    return
  fi
  awk 'NF { count++ } END { print count + 0 }' "$results_file"
}

log "epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML arm10_ckpt=$ARM10_CKPT arm_max_ckpt=$ARM_MAX_CKPT"

ARMS="${ARMS:-}"

run_arm() {
  local run_name="$1" model_cfg="$2" warm_start_ckpt="$3"
  shift 3
  local extra_flags=("$@")

  if [ -n "$ARMS" ]; then
    case ",$ARMS," in
      *",$run_name,"*) ;;
      *) log "===== $run_name not in \$ARMS, skipping ====="; return 0 ;;
    esac
  fi

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$ESOD_REPO"

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
    if [ ! -f "$warm_start_ckpt" ]; then
      log "FATAL: $run_name's warm-start checkpoint not found at $warm_start_ckpt"
      exit 1
    fi
    log "===== Training $run_name (warm-started from $warm_start_ckpt, selector frozen) ====="
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights "$warm_start_ckpt" \
      --freeze \
      --hyp "$HYP" \
      --batch-size "$BATCH" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
      "${extra_flags[@]}" \
      --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
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

  if [ "$training_was_already_done" = "1" ] && [ -f "$results_dir/best_predictions.json" ]; then
    log "===== $run_name eval already complete, skipping ====="
  else
    log "Evaluating $run_name"
    python test.py \
      --data "$DATA_YAML" --weights "$ckpt" --task "$VAL_SPLIT" \
      --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" --save-json --save-regions \
      --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_test.log"
  fi

  if [ "$training_was_already_done" = "1" ] && [ -f "$RUN_ROOT/measure/$run_name/buckets.json" ]; then
    log "===== $run_name measure already complete, skipping ====="
  else
    log "Measuring $run_name (GFLOPs/FPS, batch=1)"
    python test.py \
      --data "$DATA_YAML" --weights "$ckpt" \
      --batch-size 1 --img-size "$IMG_SIZE" --device "$GPU" --task measure \
      --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_measure.log"
  fi

  if [ ! -f "$results_dir/best_predictions.json" ]; then
    log "WARNING: $run_name produced no predictions -- skipping audit/vt_diagnose"
  else
    log "Auditing $run_name"
    python "$SCRIPT_DIR/audit_buckets.py" \
      --pred "$results_dir/best_predictions.json" \
      --labels "$DATA_ROOT/labels/$VAL_SPLIT" --images "$DATA_ROOT/images/$VAL_SPLIT" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed, continuing"

    log "vt_diagnose: $run_name"
    python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
      --labels-dir "$DATA_ROOT/labels/$VAL_SPLIT" --images-dir "$DATA_ROOT/images/$VAL_SPLIT" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed, continuing"
  fi
}

# max fusion (pooled) + ISPPHead, no SABL, selector frozen at arm (10)'s
# weights -- the SeaPerson counterpart to UAVDT's arm 14. Tests whether
# dropping SABL and unifying on Max reaches arm (10)'s own 0.766 ceiling
# while keeping ISPPHead's parameter/GFLOPs savings, as the new candidate
# flagship recipe (replacing SABL+ISPPHead entirely, on both datasets).
run_arm "seaperson_yolov5m_channel_pooled_max_isphead_frozen" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_max_isphead.yaml" "$ARM10_CKPT" \
  --selector-loss coverage --box-loss upstream

# max fusion (full-width, non-pooled spectral branch) + ISPPHead, no SABL,
# selector frozen at seaperson_yolov5m_max's own weights -- isolates
# channel pooling's own effect on the flagship "Max + staged ISPPHead"
# recipe specifically (2026-09-06), the missing half of the pooled-vs-
# non-pooled comparison run_seaperson.sh's own seaperson_yolov5m_max arm
# started at the fusion-only level. Batch size should match whatever
# seaperson_yolov5m_max's own selector training used (BATCH=2 override at
# invocation, e.g. `BATCH=2 ARMS=seaperson_yolov5m_max_isphead_frozen ...`)
# unless/until this staged fine-tune (frozen trunk, no gradient through the
# full-width spectral branch) is confirmed to tolerate the shared batch=8.
run_arm "seaperson_yolov5m_max_isphead_frozen" \
  "models/cfg/esod/seaperson_yolov5m_max_isphead.yaml" "$ARM_MAX_CKPT" \
  --selector-loss coverage --box-loss upstream

log "===== ALL DONE ====="
log "  Max+ISPPHead, pooled (frozen selector, no SABL):     $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_max_isphead_frozen/"
log "  Max+ISPPHead, non-pooled (frozen selector, no SABL): $RUN_ROOT/test/seaperson_yolov5m_max_isphead_frozen/"

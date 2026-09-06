#!/usr/bin/env bash
# Paper-direction judgment experiment (2026-09-06): does Dual-Evidence
# (semantic+spectral, whatever fusion rule) earn its keep over a much
# simpler single-evidence Spectral-only recipe, once both are given the
# same staged-ISPPHead treatment? Standalone script, touches neither
# run_uavdt.sh/run_seaperson.sh nor either dataset's own
# run_*_frozen_selector.sh (those stay scoped to the Max-fusion factorial).
#
# Motivation: across both datasets' full rosters, channel-pooled
# Spectral-only (arm 4) is one of the only components that is cleanly,
# stably positive EVERYWHERE -- UAVDT: +1.1pp mAP@.5 / +3.66pp Total Recall
# over R0; SeaPerson: +2.0pp mAP@.5 / +3.35pp recall over R0
# (HESOD-Experiment-Plan.md SS3.4 point 1 / SS4.4.2). Every dual-evidence
# fusion choice tested so far (Concat vs Max, pooled vs non-pooled) has
# gone in OPPOSITE directions on the two datasets, and SABL has been net
# negative-to-flat on both. This raises the real possibility that the
# semantic branch and its fusion machinery are not earning their keep at
# all, and that HESOD's actual contribution is spectral-saliency routing,
# not "dual evidence."
#
# This script builds the missing half of that comparison: Spectral-only
# (pooled), staged with ISPPHead (warm-start from arm 4's own converged
# checkpoint, freeze the trunk -- model.0-12, same freeze range as every
# other staged arm this project has run -- fine-tune ISPPHead only). Same
# recipe already proven out for the Max-fusion factorial (UAVDT arm 14,
# SeaPerson's own pooled-Max+ISPPHead-frozen arm): warm-starting into a
# reshaped head only needs intersect_dicts()'s direct-match-first fix
# (2026-09-05) to load the shared trunk correctly, already in place and
# dataset-agnostic.
#
# Decision rule (user-specified, 2026-09-06): compare this arm's mAP@.5 on
# each dataset against that dataset's best staged Dual-evidence+ISPPHead
# arm (UAVDT: arm 14, Max+ISPPHead-frozen, 0.394; SeaPerson: whichever of
# the pooled/non-pooled Max+ISPPHead-frozen arms wins).
#   - Dual stably >=1pp mAP better, or a clearly better efficiency/recall
#     Pareto point -> keep the dual-evidence story.
#   - Dual only wins by 0-0.5pp (or loses) -> drop semantic/fusion, pivot
#     the paper to a spectral-routing story.
#   - Split (SeaPerson favors Dual, UAVDT favors Spectral, or vice versa)
#     -> don't ship two dataset-specific HESOD variants; standardize on
#     whichever single recipe (Dual or Spectral-only) is simpler, i.e.
#     Spectral-only, since a dataset-specific fusion switch is already the
#     thing this decision is trying to avoid needing.
#
# Prerequisite: both datasets' own arm 4
# (uavdt_yolov5m_channel_pooled_spectral_only /
# seaperson_yolov5m_channel_pooled_spectral_only) must already be trained
# -- they are, part of each dataset's existing roster.
#
# Usage:
#   ARMS=uavdt_yolov5m_channel_pooled_spectral_only_isphead_frozen nohup bash run_spectral_isphead_staged.sh > /root/spectral_isphead_staged.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
EPOCHS="${EPOCHS:-20}"
log_prefix="SPECTRAL_ISPP_STAGED"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$log_prefix] $*"; }

completed_epochs() {
  local results_file="$1/results.txt"
  if [ ! -f "$results_file" ]; then
    echo 0
    return
  fi
  awk 'NF { count++ } END { print count + 0 }' "$results_file"
}

ARMS="${ARMS:-}"

run_arm() {
  local run_name="$1" model_cfg="$2" warm_start_ckpt="$3" data_yaml="$4" hyp="$5" \
    img_size="$6" batch="$7" classes="$8" val_split="$9"
  shift 9
  local extra_flags=("$@")

  if [ -n "$ARMS" ]; then
    case ",$ARMS," in
      *",$run_name,"*) ;;
      *) log "===== $run_name not in \$ARMS, skipping ====="; return 0 ;;
    esac
  fi

  local data_root
  data_root="$(dirname "$data_yaml")"

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
      log "FATAL: $run_name's warm-start checkpoint not found at $warm_start_ckpt -- that dataset's own arm 4 (Spectral-only, pooled) must be trained first"
      exit 1
    fi
    log "===== Training $run_name (warm-started from $warm_start_ckpt, selector frozen) ====="
    python train.py \
      --data "$data_yaml" \
      --cfg "$model_cfg" \
      --weights "$warm_start_ckpt" \
      --freeze \
      --hyp "$hyp" \
      --batch-size "$batch" --img-size "$img_size" --epochs "$EPOCHS" --device "$GPU" \
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
      --data "$data_yaml" --weights "$ckpt" --task "$val_split" \
      --batch-size "$batch" --img-size "$img_size" --device "$GPU" --save-json --save-regions \
      --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_test.log"
  fi

  if [ "$training_was_already_done" = "1" ] && [ -f "$RUN_ROOT/measure/$run_name/buckets.json" ]; then
    log "===== $run_name measure already complete, skipping ====="
  else
    log "Measuring $run_name (GFLOPs/FPS, batch=1)"
    python test.py \
      --data "$data_yaml" --weights "$ckpt" \
      --batch-size 1 --img-size "$img_size" --device "$GPU" --task measure \
      --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_measure.log"
  fi

  if [ ! -f "$results_dir/best_predictions.json" ]; then
    log "WARNING: $run_name produced no predictions -- skipping audit/vt_diagnose"
  else
    log "Auditing $run_name"
    python "$SCRIPT_DIR/audit_buckets.py" \
      --pred "$results_dir/best_predictions.json" \
      --labels "$data_root/labels/$val_split" --images "$data_root/images/$val_split" \
      --classes "$classes" \
      2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed, continuing"

    log "vt_diagnose: $run_name"
    python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
      --labels-dir "$data_root/labels/$val_split" --images-dir "$data_root/images/$val_split" \
      --classes "$classes" \
      2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed, continuing"
  fi
}

# UAVDT: Spectral-only (pooled) + ISPPHead, staged from arm 4's own
# converged checkpoint. --workers 2: same worker-leak precaution as every
# other channel-pooled-evidence arm on UAVDT.
run_arm "uavdt_yolov5m_channel_pooled_spectral_only_isphead_frozen" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_spectral_only_isphead.yaml" \
  "${UAVDT_ARM4_CKPT:-$RUN_ROOT/train/uavdt_yolov5m_channel_pooled_spectral_only/weights/best.pt}" \
  "${UAVDT_DATA_YAML:-/root/autodl-tmp/UAVDT_fresh.yaml}" \
  "data/hyps/hyp.uavdt.yaml" 1280 "${UAVDT_BATCH:-8}" "car,truck,bus" "test" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream --workers 2

# SeaPerson: Spectral-only (pooled) + ISPPHead, staged from arm 4's own
# converged checkpoint.
run_arm "seaperson_yolov5m_channel_pooled_spectral_only_isphead_frozen" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_spectral_only_isphead.yaml" \
  "${SEAPERSON_ARM4_CKPT:-$RUN_ROOT/train/seaperson_yolov5m_channel_pooled_spectral_only/weights/best.pt}" \
  "${SEAPERSON_DATA_YAML:-/root/autodl-tmp/seaperson.yaml}" \
  "data/hyps/hyp.seaperson.yaml" 2048 "${SEAPERSON_BATCH:-8}" "person" "test" \
  --selector-loss coverage --box-loss upstream

log "===== ALL DONE ====="
log "  UAVDT Spectral+ISPPHead (staged):     $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_spectral_only_isphead_frozen/"
log "  SeaPerson Spectral+ISPPHead (staged): $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_spectral_only_isphead_frozen/"

#!/usr/bin/env bash
# Frozen-selector fine-tune probe (2026-09-03), NOT part of the 8-arm
# roster or the max-fusion 2x2 factorial (run_uavdt.sh's own arms 9-12,
# HESOD-Experiment-Plan.md SS3.6). Standalone script, does not touch
# run_uavdt.sh, to avoid any risk to the shared roster's resume/skip logic.
#
# Motivation: SS3.6's completed 2x2 factorial shows arm (12) (max fusion +
# ISPPHead, no SABL, trained jointly from scratch) costs -4.37pp Total
# Recall / -2.5pp BPR relative to arm (9) (pure max fusion) -- but a joint
# from-scratch retrain can't separate "ISPPHead's own reduced head capacity
# structurally can't preserve as many correct candidates" from "training
# ISPPHead's weights jointly with the selector perturbs what the selector
# itself learns to route" (max fusion's own winner-take-all zero-gradient
# risk, SS3.4 point 5, means the selector may already be more sensitive to
# downstream training noise than concat's smooth-gradient combiner is).
#
# This script tests the second hypothesis directly: warm-start from arm
# (9)'s own converged checkpoint (--weights), freeze everything upstream of
# the FPN neck/Detect head (backbone + evidence branches + fusion Segmenter
# + HeatMapParser -- model.0 through model.12, train.py's --freeze, edited
# 2026-09-03 to freeze this range instead of its previous unused/inverted
# default), and fine-tune ONLY the neck+head under SABL or ISPPHead. If
# recall/BPR stay close to arm (9)'s own 0.940 BPR / 90.36% Total Recall
# under this frozen-selector regime -- unlike arms (11)/(12)'s joint-
# training results -- that supports the training-interaction hypothesis
# over a pure head-capacity story. If recall/BPR still drop by a similar
# margin even with the selector frozen and unable to be perturbed, the cost
# is intrinsic to the head's own reduced capacity, not a training artifact.
#
# Purely diagnostic/exploratory -- not required to report a paper number
# (SS3.6 already has real, audited joint-training numbers for arms 10-12
# and a documented recommendation). Shorter epoch budget than the main
# roster (EPOCHS, default 20 here vs. 50 there) since only the neck+head
# is being trained on top of already-converged, frozen upstream features --
# expected to converge faster than a from-scratch 50-epoch run; adjust via
# the EPOCHS env var if 20 proves insufficient (watch results.txt's own
# loss-plateau, not just epoch count, before trusting the final numbers).
#
# Usage:
#   ARMS=uavdt_yolov5m_channel_pooled_max_sabl_frozen nohup bash run_uavdt_frozen_selector.sh > /root/uavdt_frozen.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-1280}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UAVDT_fresh}"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/UAVDT_fresh.yaml}"
# Root cause resolved (2026-09-05, HESOD-Experiment-Plan.md SS3.7): every
# collapse this script produced before today (lr0=0.01, lr0=0.001 x
# {warmup=1, warmup=0, warmup=25, different --seed}) was NOT an LR/warmup/
# data-order issue -- an external audit found `intersect_dicts()`
# (utils/torch_utils.py) misapplied a plain-backbone layer-offset adapter
# to an already-ESOD-shaped source checkpoint, silently loading only
# ~29% of it (173/601 keys) and leaving the "frozen" trunk at random init
# the whole time. Fixed 2026-09-05 (direct-match-first, falls back to the
# offset adapter only when direct matching covers too little of the target
# model). A cheap identity-control rerun (Max ckpt -> Max cfg, CIoU,
# --freeze, hyp.uavdt.yaml's own short warmup=3/lr0=0.01, 8 epochs)
# confirmed the fix: `Transferred 601/601 items`, BPR stayed healthy
# through and past the epoch-3 warmup-end transition that used to
# collapse (0.988 during warmup -> 0.941 just after, matching arm 9's own
# 0.940 eval BPR almost exactly -- a normal recalibration, not a crash).
# Back to the standard, well-tested hyp.uavdt.yaml -- the lr0/warmup
# variants in hyp.uavdt_frozen.yaml were band-aids for a bug that's now
# actually fixed, not a real tuning need.
HYP="data/hyps/hyp.uavdt.yaml"
CLASSES="car,truck,bus"
VAL_SPLIT="test"
EPOCHS="${EPOCHS:-20}"
log_prefix="FROZEN"

# Arm (9)'s own converged checkpoint -- the warm-start source for both
# fine-tunes below. Must already exist (arm 9 is part of run_uavdt.sh's
# own roster and has been trained/audited, HESOD-Experiment-Plan.md SS3.5).
ARM9_CKPT="${ARM9_CKPT:-$RUN_ROOT/train/uavdt_yolov5m_channel_pooled_max/weights/best.pt}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$log_prefix] $*"; }

completed_epochs() {
  local results_file="$1/results.txt"
  if [ ! -f "$results_file" ]; then
    echo 0
    return
  fi
  awk 'NF { count++ } END { print count + 0 }' "$results_file"
}

if [ ! -f "$ARM9_CKPT" ]; then
  log "FATAL: arm (9) checkpoint not found at $ARM9_CKPT -- set ARM9_CKPT explicitly if it lives elsewhere"
  exit 1
fi

log "epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML arm9_ckpt=$ARM9_CKPT"

ARMS="${ARMS:-}"

run_arm() {
  local run_name="$1" model_cfg="$2"
  shift 2
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
    log "===== Training $run_name (warm-started from arm 9, selector frozen) ====="
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights "$ARM9_CKPT" \
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
      --data "$DATA_YAML" --weights "$ckpt" --task val \
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

# max fusion + SABL, selector frozen at arm (9)'s weights -- counterpart to
# arm (11) (jointly-trained max+SABL), isolates whether SABL's own -2.8pp
# mAP@.5 cost (SS3.6) persists when the selector can't be perturbed by
# SABL's training.
run_arm "uavdt_yolov5m_channel_pooled_max_sabl_frozen" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_max.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl --workers 2

# max fusion + ISPPHead, selector frozen at arm (9)'s weights -- counterpart
# to arm (12) (jointly-trained max+ISPPHead), isolates whether ISPPHead's
# own -4.37pp Total Recall / -2.5pp BPR cost (SS3.6, the larger of the two
# factors) persists when the selector can't be perturbed by training a
# lower-capacity head. ISPPHead's own neck+Detect weights don't exist in
# arm (9)'s checkpoint (different head shape) -- intersect_dicts() (train.py)
# loads only the matching frozen trunk, ISPPHead's own layers start random
# and are exactly what gets fine-tuned here.
run_arm "uavdt_yolov5m_channel_pooled_max_isphead_frozen" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_max_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream --workers 2

# max fusion + SABL + ISPPHead together, selector frozen at arm (9)'s
# weights -- the staged-training counterpart to arm (10) (HESOD Full v2,
# jointly trained from scratch). Same cfg/flags as arm (10) itself, only
# --weights/--freeze are added: warm-starts from arm (9)'s converged
# checkpoint instead of training everything together. This is the arm the
# "leverage Dual Max's own high recall" staged-training proposal is
# actually about -- if it recovers recall/BPR closer to arm (9)'s own
# 90.36%/0.940 than arm (10)'s end-to-end 86.21%/0.920 (or its noisier
# rerun, 90.29%/0.943 -- SS3.7 flags arm (10) itself as not very
# reproducible, so this arm's own result should ideally get a confirmation
# rerun too before being treated as final) while keeping ISPPHead's
# compute savings, it's a stronger flagship-recipe candidate than arm (10).
run_arm "uavdt_yolov5m_channel_pooled_max_sabl_isphead_frozen" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_max_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl --workers 2

# Confirmation rerun (2026-09-06) -- arm (15)'s own mAP@.5 (0.392) landed
# 0.2-0.3pp below both arms (13)/(14) alone (0.395/0.394), HESOD-Experiment-
# Plan.md SS3.7 own noise-floor discipline flags anything under ~2-4pp as
# indistinguishable from noise, and none of arms 13/14/15 have an
# independent rerun yet (unlike R0/arm3/arm5/arm8/arm10 earlier). Same
# config/flags as arm 15, only --seed differs (default init_seeds(2+rank)
# is identical every run since rank is always -1 on single-GPU -- without
# an explicit different seed this would just deterministically replay the
# same data order/augmentation, not a genuine independent sample). Seed is
# drawn fresh at launch time ($RANDOM, bash's own PRNG) rather than a
# literal so this rerun doesn't depend on a manually hand-picked constant.
RERUN_SEED=$RANDOM
log "arm15 rerun seed: $RERUN_SEED"
run_arm "uavdt_yolov5m_channel_pooled_max_sabl_isphead_frozen_run2" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_max_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl --workers 2 --seed "$RERUN_SEED"

log "===== ALL DONE ====="
log "  Max+SABL (frozen selector):          $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_max_sabl_frozen/"
log "  Max+ISPPHead (frozen selector):      $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_max_isphead_frozen/"
log "  Max+SABL+ISPPHead (frozen selector): $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_max_sabl_isphead_frozen/"

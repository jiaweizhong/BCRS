#!/usr/bin/env bash
# UAVDT R0 reproduction, re-run through the ACTIVE hesod/backends/hesod tree
# (the same pipeline TinyPerson/VisDrone/SeaDronesSeeV2 use), not the frozen
# hesod/backends/esod reference copy that scripts/esod_baseline/run_baseline.sh
# targets. The existing UAVDT accepted-evidence number (AP/AP50 20.1/37.0 vs
# paper's 22.5/40.7, HESOD-Experiment-Plan.md SS2) was produced by
# run_baseline.sh -- meaning it predates every active-tree fix made this
# session (fvcore uncalled-modules-warning suppression in test.py; the
# vt_diagnose.py non-numeric-image_id fix, though that one is dataset-
# agnostic and already applied regardless of which tree trained the
# checkpoint) and its provenance relative to this session's GPU host
# re-provisioning (which affected VisDrone/TinyPerson) is unconfirmed. This
# script re-establishes a clean, current-pipeline UAVDT R0 baseline before
# deciding whether to build concat/concat+SABL/ISPPHead arms for it, mirroring
# run_tinyperson.sh's exact structure for consistency.
#
# Config verified byte-identical between hesod/backends/esod and
# hesod/backends/hesod for uavdt_yolov5m.yaml and hyp.uavdt.yaml (diff empty,
# 2026-08-19) -- no config drift to account for, only the code tree differs.
#
# UAVDT has no separate val directory -- its data yaml's val: key already
# points at the on-disk "test" split (run_baseline.sh's own convention:
# val_split=test), so --task val (test.py's default) is correct here, unlike
# TinyPerson/VisDrone/SeaDronesSeeV2's --task test.
#
# No COCOeval/small-medium-large AP wiring yet for UAVDT (unlike the
# TinyPerson/SeaDronesSeeV2 format_*() functions in test.py) -- UAVDT's GT is
# MOT-style txt (UAV-benchmark-MOTD_v1.0/GT/*.txt), not COCO JSON, so that
# would need a GT-to-COCO-JSON conversion step first. Not attempted here;
# flag if wanted later.
#
# Usage:
#   SMOKE=1 bash run_uavdt.sh       # ~minutes, validate the pipeline first
#   nohup bash run_uavdt.sh > /root/uavdt.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-1280}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
# UAVDT_v3 does not exist on the remote box (confirmed 2026-08-23: `ls
# /root/autodl-tmp/UAVDT_v3/labels/` -> No such file or directory) --
# UAVDT_fresh is the only prepared copy and what every actual run this
# session (including the R0 rerun) has used, via explicit env var override.
# Defaulting to it here so that override stops being required every time.
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UAVDT_fresh}"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/UAVDT_fresh.yaml}"
HYP="data/hyps/hyp.uavdt.yaml"
CLASSES="car,truck,bus"
VAL_SPLIT="test"

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  EPOCHS=1
  SUFFIX="_smoke"
  log_prefix="SMOKE"
else
  EPOCHS="${EPOCHS:-50}"
  SUFFIX=""
  log_prefix="REAL"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$log_prefix] $*"; }

completed_epochs() {
  local results_file="$1/results.txt"
  if [ ! -f "$results_file" ]; then
    echo 0
    return
  fi
  awk 'NF { count++ } END { print count + 0 }' "$results_file"
}

log "epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML"

# Optional: comma-separated list of exact run_name(s) to run, matching
# run_seaperson.sh's own ARMS convention (added there first, ported here
# 2026-08-24 to selectively add the concat-only/concat+SABL isolation arms
# without re-running R0/concat+SABL+ISPPHead's eval+measure+audit, which
# this script always redoes even when training was already complete --
# unlike run_seaperson.sh, that skip-if-already-evaluated fix was never
# ported here; ARMS filtering sidesteps needing it for this use).
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
    log "===== Training $run_name ====="
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights weights/pretrained/yolov5m.pt \
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

  # Eval and measure are gated independently (2026-08-25, split from one
  # combined check) -- eval (--task val, a few minutes) recomputes BPR/mAP/
  # recall from a live forward pass (BPR specifically comes from
  # cluster_recall() on the selector's chosen regions, not from saved
  # predictions -- it cannot be backfilled from an existing
  # best_predictions.json). Measure (--task measure, batch=1 over the full
  # ~16.6k-image test set) is the expensive ~50min GFLOPs/FPS profiling step
  # and rarely needs redoing once captured. Deleting only
  # best_predictions.json (leaving buckets.json alone) now reruns eval alone
  # without also burning ~50min re-measuring GFLOPs/FPS that hasn't changed.
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
    log "WARNING: $run_name produced no predictions (expected at 1 epoch under SMOKE) -- skipping audit/vt_diagnose"
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

  log "TFR skipped for UAVDT -- not attempted here"
}

# --- 8-arm roster (2026-08-25), originally matching run_seaperson.sh's arm
# set/order exactly for direct arm-for-arm comparability. Arm (5) swapped
# 2026-08-27: gated-fusion -> channel-pooled-spectral-only (see that arm's
# own run_arm() call below for why). Same val=test protocol as the
# just-completed R0 rerun (kept deliberately -- see HESOD-Experiment-Plan.md's
# UAVDT open-item note; not changed here). ---

# R0: Segmenter, upstream loss, CIoU box (original baseline)
run_arm "uavdt_yolov5m_baseline${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m.yaml"

# semantic-only: SAME architecture as R0, coverage loss instead -- isolates
# the loss-function effect from the selector-architecture effect. No new
# cfg needed, reuses uavdt_yolov5m.yaml.
run_arm "uavdt_yolov5m_semantic_coverage${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0

# spectral-only: SpectralOnlySegmenter, coverage loss -- no semantic head
# contributes to the routing logit. --batch-size 2 here overrides the
# shared $BATCH for TRAINING only (argparse takes the last --batch-size
# occurrence; this one lands after "--batch-size $BATCH" in run_arm()'s own
# invocation) -- ported precaution from SeaPerson's spectral-only OOM
# (full-width unpooled branch, HESOD-Experiment-Plan.md SS8); UAVDT's
# img-size 1280 has ~0.39x SeaPerson's 2048 pixel count so this is likely
# unnecessary here, but untested, and an OOM would abort the whole roster
# under `set -euo pipefail`. Eval/measure stay at the shared batch --
# inference-only memory pressure is much lower than training's.
run_arm "uavdt_yolov5m_spectral_only${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_spectral_only.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --batch-size 2

# concat-only: channel-pooled concat evidence, CIoU box (no SABL, no
# ISPPHead) -- the missing isolation control HESOD-Experiment-Plan.md SS9.2
# flagged (only R0 and the fully-bundled concat+SABL+ISPPHead existed
# before). Added 2026-08-24 alongside concat+SABL below specifically to
# test whether SeaPerson's own SABL finding (no clean accuracy win, SS8.1)
# also holds on UAVDT.
#
# --workers 2 (2026-08-26): first fresh (non-resumed) launch of this arm
# showed TWO generations of DataLoader worker processes alive simultaneously
# by epoch 8 (16 processes instead of the expected 8, ~840MB-1.5GB RSS each,
# via `ps aux --sort=-%mem`) -- val's testloader iterator's workers from the
# previous epoch were not being torn down before the next epoch's iterator
# spawned a fresh set, growing host memory roughly per-epoch until the
# platform's memory quota (not visible via `free`/`dmesg` inside this
# container -- see the platform dashboard) was hit and the process was
# killed. R0/semantic-only (plain Segmenter, same --workers 8) completed all
# 50 epochs without this; concat-only's ChannelPooledConcatEvidenceSegmenter
# (and by extension gated-fusion/concat+SABL/concat+ISPPHead below, same
# family) reproduces it much faster -- root cause not isolated further, this
# is a mitigation (fewer workers per leaked generation), not a fix.
run_arm "uavdt_yolov5m_channel_pooled_concat${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream --workers 2

# channel-pooled-spectral-only (2026-08-27, replaces gated-fusion): isolates
# the confound in spectral-only's own strong result (90.38% Total Recall,
# +5.21pp over R0) -- spectral-only is simultaneously the only "spectral
# evidence alone" arm AND the only unpooled/full-capacity arm, so its
# accuracy could come from either the evidence itself or the extra channel
# capacity. This arm pools the same spectral evidence the same way
# concat-only does, isolating that variable directly (mirrors
# seaperson_yolov5m_channel_pooled_spectral_only.yaml's own confound-check,
# which resolved in favor of "evidence itself is sufficient" on SeaPerson).
# gated-fusion dropped from this roster: already has a clear negative result
# on SeaPerson (HESOD-Experiment-Plan.md SS5, worst of all coverage-loss
# arms there) -- not worth re-confirming on UAVDT, this confound-check is
# higher value for the same GPU budget. --workers 2: same worker-leak
# precaution as concat-only above (same evidence-branch family); pooling
# should also let it train stably, unlike spectral-only's forced batch=2.
run_arm "uavdt_yolov5m_channel_pooled_spectral_only${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_spectral_only.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream --workers 2

# concat+SABL: same concat architecture, SABL box regression, no ISPPHead --
# paired with concat-only above to isolate SABL's contribution cleanly,
# same comparison structure as SeaPerson's concat-only/concat+SABL pair.
# --workers 2: same precaution as concat-only above.
run_arm "uavdt_yolov5m_channel_pooled_concat_sabl${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl --workers 2

# Concat+SABL+ISPPHead (HESOD Full): the project's best-known lightweight
# recipe. First time this arm is actually retrained on UAVDT this session --
# its prior checkpoint (2026-08-21) predates all of it. Preserves UAVDT's
# own SPP layer + threshold=0.3 architecture deltas exactly (see the
# config's own header comment). --workers 2: same worker-leak precaution as
# every other channel-pooled arm (concat-only/gated-fusion-replacement/
# concat+SABL/concat+ISPPHead all needed it; this one was never run before
# so it never got the chance to crash, but there's no reason to think it's
# exempt).
run_arm "uavdt_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl --workers 2

# concat+ISPPHead (no SABL): same cfg as concat+SABL+ISPPHead, upstream/CIoU
# box loss -- isolates ISPPHead's saving from SABL's mixed/inconclusive
# accuracy effect, same comparison structure as SeaPerson's own pair.
# --workers 2: same worker-leak precaution as concat-only above.
run_arm "uavdt_yolov5m_channel_pooled_concat_isphead${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream --workers 2

# --- Exploratory probe (2026-08-30), NOT part of the 8-arm roster ---
#
# concat-only-posw1: same architecture/cfg as arm (5) Concat-only, only
# --pos-weight lowered 2.0 -> 1.0 (mask BCE positive-class weight,
# selector_loss=coverage only). Motivated by a --hm-threshold/--top-k
# inference-time sweep on arm (5)'s checkpoint (sweep_uavdt_concat_thresholds.sh)
# that found mAP@.5 completely flat (0.371 at every threshold from 0.3 to
# 0.6, recall dropping monotonically instead) -- ruling out "too many raw
# candidates diluting precision" as the mechanism behind concat-only trailing
# every single-evidence arm on mAP. pos_weight=2.0 doubles the loss cost of
# a missed-positive cell relative to a false-positive one, directly biasing
# the selector toward permissive/imprecise routing -- this experiment tests
# whether that specific asymmetry (not a fusion-architecture problem) is the
# root cause by removing it (pos_weight=1.0, no extra positive bias) and
# retraining from scratch. --workers 2: same precaution as arm (5).
run_arm "uavdt_yolov5m_channel_pooled_concat_posw1${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 1.0 --box-loss upstream --workers 2

# concat-only-maxfusion: same evidence branches as arm (5) Concat-only, but
# ChannelPooledMaxEvidenceSegmenter (torch.max in logit space) replaces the
# learned 1x1 combiner. Tests the fusion RULE itself as the root cause,
# independent of the coverage-loss pos_weight probe above -- max fusion
# structurally cannot produce a combined score below both single-evidence
# branches, which is concat-only's observed failure mode (SS3.4 point 2).
# Queued after the pos_weight probe, not run concurrently with it.
run_arm "uavdt_yolov5m_channel_pooled_max${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_max.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream --workers 2

log "===== ALL DONE ====="
log "  R0:                    $RUN_ROOT/test/uavdt_yolov5m_baseline${SUFFIX}/"
log "  Semantic-only:         $RUN_ROOT/test/uavdt_yolov5m_semantic_coverage${SUFFIX}/"
log "  Spectral-only:         $RUN_ROOT/test/uavdt_yolov5m_spectral_only${SUFFIX}/"
log "  Concat-only:           $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat${SUFFIX}/"
log "  Channel-Pooled Spectral-only: $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_spectral_only${SUFFIX}/"
log "  Concat+SABL:           $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat_sabl${SUFFIX}/"
log "  Concat+SABL+ISPPHead:  $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}/"
log "  Concat+ISPPHead:       $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat_isphead${SUFFIX}/"
log "  Compare R0's AP/AP50 against the just-completed R0 rerun (0.385/0.215) and the paper (0.407/0.225) --"
log "  this run reuses uavdt_yolov5m_baseline's existing checkpoint (already >= EPOCHS), so R0 itself will be skipped, not retrained."

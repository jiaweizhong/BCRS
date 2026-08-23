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
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UAVDT_v3}"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/UAVDT_v3.yaml}"
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

  if [ "$done_epochs" -ge "$EPOCHS" ] && [ -f "$ckpt" ]; then
    log "===== $run_name already completed $done_epochs/$EPOCHS epochs, skipping training ====="
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

  log "Evaluating $run_name"
  python test.py \
    --data "$DATA_YAML" --weights "$ckpt" --task val \
    --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" --save-json --save-regions \
    --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_test.log"

  log "Measuring $run_name (GFLOPs/FPS, batch=1)"
  python test.py \
    --data "$DATA_YAML" --weights "$ckpt" \
    --batch-size 1 --img-size "$IMG_SIZE" --device "$GPU" --task measure \
    --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_measure.log"

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

# R0: semantic-only selector, CIoU box loss (baseline) -- re-establish on the
# active tree before deciding on concat/SABL/ISPPHead arms
run_arm "uavdt_yolov5m_baseline${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m.yaml"

# concat-only: channel-pooled concat evidence, CIoU box (no SABL, no
# ISPPHead) -- the missing isolation control HESOD-Experiment-Plan.md SS9.2
# flagged (only R0 and the fully-bundled concat+SABL+ISPPHead existed
# before). Added 2026-08-24 alongside concat+SABL below specifically to
# test whether SeaPerson's own SABL finding (no clean accuracy win, SS8.1)
# also holds on UAVDT.
run_arm "uavdt_yolov5m_channel_pooled_concat${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss upstream

# concat+SABL: same concat architecture, SABL box regression, no ISPPHead --
# paired with concat-only above to isolate SABL's contribution cleanly,
# same comparison structure as SeaPerson's concat-only/concat+SABL pair.
run_arm "uavdt_yolov5m_channel_pooled_concat_sabl${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

# Concat+SABL+ISPPHead: the project's best-known lightweight recipe, first
# time run on UAVDT. Preserves UAVDT's own SPP layer + threshold=0.3
# architecture deltas exactly (see the config's own header comment).
run_arm "uavdt_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  R0:                    $RUN_ROOT/test/uavdt_yolov5m_baseline${SUFFIX}/"
log "  Concat-only:           $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat${SUFFIX}/"
log "  Concat+SABL:           $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat_sabl${SUFFIX}/"
log "  Concat+SABL+ISPPHead:  $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}/"
log "  Compare R0's AP/AP50 against the existing accepted-evidence value (20.1/37.0 audited, HESOD-Experiment-Plan.md SS2) --"
log "  a meaningfully different number here would indicate that value's provenance (frozen-tree run_baseline.sh, possibly pre-re-provisioning) matters."

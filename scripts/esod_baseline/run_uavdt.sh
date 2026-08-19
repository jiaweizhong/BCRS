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

log "epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML"

run_arm() {
  local run_name="$1" model_cfg="$2"
  shift 2
  local extra_flags=("$@")

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$ESOD_REPO"

  local ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"
  if [ -f "$ckpt" ]; then
    log "===== $run_name already trained (found $ckpt), skipping training ====="
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

# Concat+SABL+ISPPHead: the project's best-known lightweight recipe, first
# time run on UAVDT (no plain concat+SABL arm -- same skip-the-middle-arm
# reasoning as SeaDronesSeeV2/fresh-TinyPerson). Preserves UAVDT's own SPP
# layer + threshold=0.3 architecture deltas exactly (see the config's own
# header comment).
run_arm "uavdt_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}" \
  "models/cfg/esod/uavdt_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  R0:                    $RUN_ROOT/test/uavdt_yolov5m_baseline${SUFFIX}/"
log "  Concat+SABL+ISPPHead:  $RUN_ROOT/test/uavdt_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}/"
log "  Compare R0's AP/AP50 against the existing accepted-evidence value (20.1/37.0 audited, HESOD-Experiment-Plan.md SS2) --"
log "  a meaningfully different number here would indicate that value's provenance (frozen-tree run_baseline.sh, possibly pre-re-provisioning) matters."

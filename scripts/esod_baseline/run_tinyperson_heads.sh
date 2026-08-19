#!/usr/bin/env bash
# TinyPerson lightweight-Detect-head experiment (HESOD-Lightweight-Detector-
# Review-and-Roadmap.md SS5.2). Swaps only the Detect head on top of our best
# TinyPerson arm so far (channel_pooled_concat + SABL,
# tinyperson_yolov5m_channel_pooled_concat_sabl -- HESOD-Experiment-Plan.md):
#   - SharedDWHead ("H1a"): one shared DW+PW trunk feeding cls/reg/obj,
#     c_mid=min(c1,128) -- ~99% Detect-head param reduction vs YOLOv6Head.
#   - ISPPHead ("H1b"): shared inverted-residual/partial-conv stem, but keeps
#     YOLOv6Head's own full-c1-width predictors -- smaller reduction, reuses
#     proven predictor capacity.
# Everything else (selector, loss flags, hyp, img-size, epochs) is identical
# to the already-validated tinyperson_concat_sabl run, so any accuracy/GFLOPs
# delta is attributable to the head swap alone.
#
# head_type is now yaml-selectable (models/yolo.py parse_model/Detect wired
# this session): Detect, [nc, anchors, 'SharedDWHead'|'ISPPHead'] in the
# model cfg's final layer. Old configs (2-element Detect args) are unaffected
# -- default head_type stays 'YOLOv6Head'.
#
# Usage:
#   SMOKE=1 bash run_tinyperson_heads.sh       # ~minutes, validate first
#   nohup bash run_tinyperson_heads.sh > /root/tinyperson_heads.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="/root/autodl-tmp/TinyPerson_v1"
DATA_YAML="/root/autodl-tmp/TinyPerson_v1.yaml"
HYP="data/hyps/hyp.tinyperson.yaml"
CLASSES="person"

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
    --data "$DATA_YAML" --weights "$ckpt" --task test \
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
      --labels "$DATA_ROOT/labels/val" --images "$DATA_ROOT/images/val" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed for $run_name, continuing"

    log "vt_diagnose: $run_name"
    python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
      --labels-dir "$DATA_ROOT/labels/val" --images-dir "$DATA_ROOT/images/val" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed for $run_name, continuing"
  fi

  log "TFR skipped for TinyPerson (variable native resolution) -- not run"
}

# SharedDWHead ("H1a") on top of concat+SABL
run_arm "tinyperson_yolov5m_channel_pooled_concat_sabl_shareddw${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat_shareddw.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

# ISPPHead ("H1b") on top of concat+SABL
run_arm "tinyperson_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  SharedDWHead: $RUN_ROOT/test/tinyperson_yolov5m_channel_pooled_concat_sabl_shareddw${SUFFIX}/"
log "  ISPPHead:     $RUN_ROOT/test/tinyperson_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}/"

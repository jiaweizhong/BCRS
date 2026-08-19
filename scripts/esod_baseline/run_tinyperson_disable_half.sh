#!/usr/bin/env bash
# TinyPerson R0 / concat+SABL / concat+SABL+ISPPHead, all retrained with
# --disable-half (forces pure FP32 training, train.py's
# `half_precision = not opt.disable_half` defaults to AMP/FP16 mixed
# precision otherwise). Control run for the unresolved ~6pp APt50 / ~3pp
# APs50 reproduction gap (HESOD-Experiment-Plan.md SS2) -- the paper's own
# text says nothing about mixed precision (confirmed absent from
# reference/ESOD.pdf, same category as SyncBN/augmentation/NMS threshold,
# already-unspecified variables). Mixed precision plausibly hurts tiny-object
# localization specifically more than larger objects (box w/h in the
# single-digit-pixel range is more numerically sensitive than a 100px box),
# which would fit the APt50 > APs50 gap asymmetry -- untested until now.
# test.py/measure already default to FP32 (`--half` defaults False), so this
# is purely a training-time variable; no change needed on the eval side.
#
# All three arms run at the same --disable-half so we can see whether the
# precision effect (if any) is uniform across R0/concat/concat+ISPPHead or
# concentrated in one. Compare each against its AMP-trained counterpart:
#   R0:                    55.26/71.23 APt50/APs50 (HESOD-Experiment-Plan.md SS2)
#   concat+SABL:           0.627 mAP50 / 0.231 mAP50:95 / 76.72% Very Tiny recall
#   concat+SABL+ISPPHead:  0.625 mAP50 / 0.231 mAP50:95 / 76.75% Very Tiny recall
# (last two from HESOD-Lightweight-Detector-Review-and-Roadmap.md SS5.2's
# "H1a/H1b 实测结果")
#
# FP32 (no AMP) needs meaningfully more activation memory than the default
# AMP path -- R0 alone OOM'd at BATCH=8/img-size=2048 on a 32GB card
# (allocation failed by ~96MiB, right at the boundary). Pass a smaller BATCH
# via env var if any arm OOMs; batch size doesn't affect what's being tested
# here (precision), so it's a safe knob, though it does introduce a small
# secondary confound (BatchNorm statistics/gradient noise shift somewhat with
# batch size) worth remembering if results come back ambiguous.
#
# Usage:
#   SMOKE=1 bash run_tinyperson_disable_half.sh              # validate first
#   BATCH=6 SMOKE=1 bash run_tinyperson_disable_half.sh       # if OOM at default batch
#   nohup bash run_tinyperson_disable_half.sh > /root/tinyperson_fp32.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
MASK_MODE="${MASK_MODE:-paper-hybrid}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="/root/autodl-tmp/TinyPerson_v1"
DATA_YAML="/root/autodl-tmp/TinyPerson_v1.yaml"
HYP="data/hyps/hyp.tinyperson.yaml"
CLASSES="person"
TINY_SET_GT="/root/autodl-tmp/tiny_set/mini_annotations/tiny_set_test_all.json"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-0}"

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

python "$SCRIPT_DIR/audit_tinyperson_protocol.py" \
  --data-root "$DATA_ROOT" --gt "$TINY_SET_GT" --expected-mask-mode "$MASK_MODE"

log "protocol=paper-text mask=$MASK_MODE epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML (--disable-half)"

run_arm() {
  local run_name="$1" model_cfg="$2"
  shift 2
  local extra_flags=("$@")

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$ESOD_REPO"

  local ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"
  if [ -f "$ckpt" ]; then
    if [ "$REUSE_CHECKPOINTS" = "1" ]; then
      log "===== explicitly reusing $ckpt ====="
    else
      log "FATAL: $ckpt already exists; refusing silent checkpoint reuse"
      exit 1
    fi
  else
    log "===== Training $run_name (--disable-half, pure FP32) ====="
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights weights/pretrained/yolov5m.pt \
      --hyp "$HYP" \
      --batch-size "$BATCH" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
      --disable-half \
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
    log "WARNING: $run_name produced no predictions (expected at 1 epoch under SMOKE) -- skipping audit/vt_diagnose/official-eval"
  else
    log "Auditing $run_name"
    python "$SCRIPT_DIR/audit_buckets.py" \
      --pred "$results_dir/best_predictions.json" \
      --labels "$DATA_ROOT/labels/val" --images "$DATA_ROOT/images/val" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed, continuing"

    log "vt_diagnose: $run_name"
    python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
      --labels-dir "$DATA_ROOT/labels/val" --images-dir "$DATA_ROOT/images/val" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed, continuing"

    if [ "$SMOKE" = "1" ]; then
      log "Skipping official TinyPerson evaluator under SMOKE (1-epoch predictions aren't meaningful for APt50/APs50)"
    else
      log "Official TinyPerson evaluator (APt50/APs50, paper-comparable protocol): $run_name"
      python "$SCRIPT_DIR/tinyperson_eval/eval_tinyperson_official.py" \
        --pred "$results_dir/best_predictions.json" \
        --gt "$TINY_SET_GT" \
        2>&1 | tee "$results_dir/${run_name}_official_eval.log" || log "WARNING: official evaluator failed, continuing"
    fi
  fi
}

# R0: paper-comparable baseline
run_arm "tinyperson_paper_r0_ratio8_${MASK_MODE}_fp32${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m.yaml" \
  --selector-loss paper

# Concat+SABL: current best TinyPerson arm
run_arm "tinyperson_yolov5m_channel_pooled_concat_sabl_fp32${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

# Concat+SABL+ISPPHead: current leading lightweight-head candidate (HESOD-
# Lightweight-Detector-Review-and-Roadmap.md SS5.2)
run_arm "tinyperson_yolov5m_channel_pooled_concat_sabl_isphead_fp32${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  R0 FP32:                   $RUN_ROOT/test/tinyperson_yolov5m_baseline_fp32${SUFFIX}/"
log "  Concat+SABL FP32:          $RUN_ROOT/test/tinyperson_yolov5m_channel_pooled_concat_sabl_fp32${SUFFIX}/"
log "  Concat+SABL+ISPPHead FP32: $RUN_ROOT/test/tinyperson_yolov5m_channel_pooled_concat_sabl_isphead_fp32${SUFFIX}/"
log "  Compare each *_official_eval.log's APt50/APs50 against the AMP-trained counterparts noted in this script's header."

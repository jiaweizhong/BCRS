#!/usr/bin/env bash
# SeaDronesSeeV2 (CompressedVersion): R0 baseline + concat+SABL +
# concat+SABL+ISPPHead, per the user's plan -- a new dataset chosen for high
# native resolution (mixed, up to 3840x2160) and small objects, though the
# diagnostic (2026-08-xx) found only ~5.5% of train boxes are "Very Tiny"
# (<256px^2) vs VisDrone's ~31% / TinyPerson's ~53% -- so unlike those two,
# the "routing should win here" premise is only partially supported going in;
# this is an exploratory run, not a confirmatory one. No YOLOv5 baseline
# exists in the original SeaDronesSeeV2 paper (different class taxonomy,
# non-YOLO baselines) -- purely internal comparison, no external
# paper-accuracy target. The third arm (ISPPHead, HESOD-Lightweight-
# Detector-Review-and-Roadmap.md SS5.2) tests whether TinyPerson's
# accuracy-neutral ~21% GFLOPs reduction (SS5.2's "H1a/H1b 实测结果")
# holds on a third, structurally different dataset.
#
# img-size 1536: reference/HighResolution/*.pdf split 640x640 (EUAVDet.pdf,
# SeaLSOD-YOLO.pdf, and "Maritime Small Object Detection.pdf"'s SAHI tile
# size) vs YOLOv7-sea.pdf's much larger side-length-2400 (explicitly argued
# "for small targets, the larger the input image scale, the better the
# detection performance"). 640 is the same order of magnitude as Pest24's
# 800x600, where this project already found routing does NOT pay off
# (A0 dense beat every routed arm) -- so 640 risks repeating that outcome
# without testing anything new. 1536 is picked as a deliberately-higher
# middle ground with an in-project precedent (VisDrone's own protocol,
# same 8x8-patch-grid architecture) rather than either literature extreme;
# treat this first run as a trial, not a validated choice -- revisit against
# 2400 if this looks inconclusive.
#
# Data must already be converted: run once first --
#   python scripts/data_prepare.py --dataset /root/autodl-tmp/CompressedVersion
# (uses data_prepare.py::prepare_seadronesseev2(), dispatch matches on
# "seadrones" in the --dataset path)
#
# SMOKE TEST MODE: never run before -- always smoke first.
#
# Usage:
#   SMOKE=1 bash run_seadronesseev2.sh       # ~minutes, validate pipeline first
#   nohup bash run_seadronesseev2.sh > /root/seadronesseev2.log 2>&1 &   # real run
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-1536}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="/root/autodl-tmp/CompressedVersion"
DATA_YAML="data/seadronesseev2.yaml"
HYP="data/hyps/hyp.seadronesseev2.yaml"
CLASSES="swimmer,boat,jetski,life_saving_appliances,buoy"

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

  log "TFR skipped for SeaDronesSeeV2 (variable native resolution) -- not run"
}

# R0: semantic-only selector, CIoU box loss (baseline)
run_arm "seadronesseev2_yolov5m_baseline${SUFFIX}" \
  "models/cfg/esod/seadronesseev2_yolov5m.yaml"

# Concat+SABL: our best Pest24/VisDrone/TinyPerson Very-Tiny-recall arm
run_arm "seadronesseev2_yolov5m_channel_pooled_concat_sabl${SUFFIX}" \
  "models/cfg/esod/seadronesseev2_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

# Concat+SABL+ISPPHead: H1b lightweight Detect head (HESOD-Lightweight-
# Detector-Review-and-Roadmap.md SS5.2) on top of the concat+SABL arm above.
# On TinyPerson this held accuracy flat while cutting GFLOPs ~21% -- testing
# whether that holds on a third, structurally different dataset.
run_arm "seadronesseev2_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}" \
  "models/cfg/esod/seadronesseev2_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  R0:                   $RUN_ROOT/test/seadronesseev2_yolov5m_baseline${SUFFIX}/"
log "  Concat+SABL:           $RUN_ROOT/test/seadronesseev2_yolov5m_channel_pooled_concat_sabl${SUFFIX}/"
log "  Concat+SABL+ISPPHead:  $RUN_ROOT/test/seadronesseev2_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}/"

#!/usr/bin/env bash
# VisDrone: R0 baseline + our two best Pest24 arms (concat+SABL, gate+SABL) --
# same rationale as run_tinyperson.sh (HESOD-Agri-Experiment-Plan.md
# SS11.2.9's Pest24 A0-vs-routed finding motivates re-testing the routing
# thesis, and the gate-vs-concat question, on a high-native-resolution
# dataset). R0 (visdrone_yolov5m.yaml) and concat
# (visdrone_yolov5m_channel_pooled_concat.yaml) are pre-existing, previously-
# validated configs (this project's prior VisDrone baseline: 78.00% Very
# Tiny recall). visdrone_yolov5m_reliability_gate.yaml is newly built this
# session (same Segmenter-swap pattern as tinyperson_yolov5m_reliability_gate.yaml)
# and has never been run -- hence SMOKE mode below, same as TinyPerson.
#
# Config source: HESOD-Experiment-Plan.md's established VisDrone protocol
# (visdrone_yolov5m.yaml, hyp.visdrone.yaml, img-size 1536, 50 epochs, batch 8).
#
# VisDrone's native image resolution is NOT uniform (confirmed by direct
# sampling: 960x540/1360x765/1920x1080 mixed in a 200-image sample) -- same
# situation as TinyPerson. vt_diagnose.py is called with --images-dir so it
# reads each image's real size via PIL instead of assuming one global size.
# TFR (tfr_diagnose.py) is deliberately NOT run here yet, for the same
# reason as TinyPerson (see run_tinyperson.sh's header) -- needs test.py's
# --save-regions to store each image's actual achieved tensor shape first.
#
# SMOKE TEST MODE: run `SMOKE=1 bash run_visdrone.sh` first -- trains 1
# epoch per arm under a "_smoke" run_name, runs the full eval/audit/vt
# pipeline anyway. A near-zero-predictions result for the gate/concat arms
# specifically at 1 epoch is EXPECTED (train.py's use_gt warmup is active
# for epoch 0 regardless of total epoch count -- real selector routing is
# still essentially untrained) -- audit/vt_diagnose are skipped gracefully
# for any arm with no predictions, not treated as a failure. What actually
# matters at this stage: no crashes, finite (non-NaN) losses for all three
# arms, especially the never-before-run gate config.
#
# Usage:
#   SMOKE=1 bash run_visdrone.sh       # ~minutes, validate the pipeline first
#   nohup bash run_visdrone.sh > /root/visdrone.log 2>&1 &   # real run, overnight
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-1536}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="/root/autodl-tmp/VisDrone_v2"
DATA_YAML="/root/autodl-tmp/VisDrone_v2.yaml"
HYP="data/hyps/hyp.visdrone.yaml"
CLASSES="pedestrian,people,bicycle,car,van,truck,tricycle,awning-tricycle,bus,motor"

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
    log "WARNING: $run_name produced no predictions (expected at 1 epoch under SMOKE -- "
    log "  real selector routing is still essentially untrained) -- skipping audit/vt_diagnose for this arm"
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

  log "TFR skipped for VisDrone (variable native resolution, see script header) -- not run"
}

# R0: semantic-only selector, CIoU box loss (baseline, pre-existing config)
run_arm "visdrone_yolov5m_baseline${SUFFIX}" \
  "models/cfg/esod/visdrone_yolov5m.yaml"

# Concat+SABL: our best Pest24 Very-Tiny-recall arm (R3-equivalent),
# pre-existing base config (visdrone_yolov5m_channel_pooled_concat.yaml)
run_arm "visdrone_yolov5m_channel_pooled_concat_sabl${SUFFIX}" \
  "models/cfg/esod/visdrone_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

# Gate+SABL: newly built config this session -- re-tests whether the
# reliability gate beats concat here, since it did not on Pest24
# (HESOD-Agri-Experiment-Plan.md SS11.2.6/SS11.2.8)
run_arm "visdrone_yolov5m_reliability_gate_sabl${SUFFIX}" \
  "models/cfg/esod/visdrone_yolov5m_reliability_gate.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  R0:              $RUN_ROOT/test/visdrone_yolov5m_baseline${SUFFIX}/"
log "  Concat+SABL:     $RUN_ROOT/test/visdrone_yolov5m_channel_pooled_concat_sabl${SUFFIX}/"
log "  Gate+SABL:       $RUN_ROOT/test/visdrone_yolov5m_reliability_gate_sabl${SUFFIX}/"

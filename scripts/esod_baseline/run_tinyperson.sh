#!/usr/bin/env bash
# TinyPerson: R0 baseline + our two best Pest24 arms (concat+SABL, gate+SABL),
# re-tested here because TinyPerson is a high-native-resolution, small-and-
# sparse-target dataset -- the regime HESOD-Agri-Experiment-Plan.md SS11.2.9
# suggests routing needs to pay off, unlike Pest24 (low native resolution +
# extreme density, where A0/dense beat every routed arm). Also re-tests
# whether the reliability gate (F5/F6) can beat concat here, since Pest24's
# gate line closed without a win (SS11.2.6/SS11.2.8).
#
# Config source: HESOD-Experiment-Plan.md's established TinyPerson protocol
# (tinyperson_yolov5m.yaml, hyp.tinyperson.yaml, img-size 2048, 50 epochs,
# batch 8 -- same training protocol as Pest24, just a different resolution).
# concat/gate model yamls (tinyperson_yolov5m_channel_pooled_concat.yaml,
# tinyperson_yolov5m_reliability_gate.yaml) are newly built this session,
# mirroring pest24_yolov5m_reliability_gate.yaml's Segmenter-swap pattern --
# never run before, hence the SMOKE mode below.
#
# TinyPerson's native image resolution is NOT uniform (confirmed by direct
# sampling: 1920x1080/1280x720/plus several outlier sizes in a 200-image
# sample) -- unlike Pest24's confirmed-uniform 800x600. vt_diagnose.py is
# called with --images-dir so it reads each image's real size via PIL
# instead of assuming one global size.
#
# TFR (tfr_diagnose.py) is deliberately NOT run here yet: it assumes a
# single global native size when interpreting best_selected_regions.json's
# P3-grid-space coordinates, which is wrong for TinyPerson's variable
# resolution under rect=True batching (different aspect-ratio batches likely
# get different actual tensor shapes). Needs test.py's --save-regions to
# also store each image's actual achieved tensor shape before TFR can be
# computed correctly here -- not attempted in this script.
#
# SMOKE TEST MODE: run configs and real data have never been exercised
# together before. Run `SMOKE=1 bash run_tinyperson.sh` first -- this trains
# 1 epoch per arm under a "_smoke" run_name (so it can never collide with or
# be mistaken for the real run's checkpoint) and runs the full eval/audit/vt
# pipeline anyway (TinyPerson's val split is small, so this stays fast).
# Watch for: no crashes, finite (non-NaN) losses, and a plausible (not
# necessarily good) AP number. Only after that looks healthy, run without
# SMOKE for the real 50-epoch overnight pass.
#
# Usage:
#   SMOKE=1 bash run_tinyperson.sh       # ~minutes, validate the pipeline first
#   nohup bash run_tinyperson.sh > /root/tinyperson.log 2>&1 &   # real run, overnight
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

  # SMOKE mode with only 1 epoch: use_gt warmup (train.py's `use_gt = epoch <
  # epochs*0.6`) is active for epoch 0 regardless of total epoch count, so
  # in-training validation (GT-routed) can show nonzero signal while this
  # standalone eval (real selector routing, still essentially untrained after
  # 1 epoch) legitimately produces zero predictions -- best_predictions.json
  # then never gets written (test.py only writes it when len(jdict) > 0).
  # That is an expected SMOKE-mode outcome, not a failure -- don't let it
  # abort the whole script via pipefail and block the remaining arms.
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

  log "TFR skipped for TinyPerson (variable native resolution, see script header) -- not run"
}

# R0: semantic-only selector, CIoU box loss (baseline)
run_arm "tinyperson_yolov5m_baseline${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m.yaml"

# Concat+SABL: our best Pest24 Very-Tiny-recall arm (R3-equivalent)
run_arm "tinyperson_yolov5m_channel_pooled_concat_sabl${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

# Gate+SABL: re-tests whether the reliability gate beats concat here,
# since it did not beat concat on Pest24 (HESOD-Agri-Experiment-Plan.md
# SS11.2.6/SS11.2.8)
run_arm "tinyperson_yolov5m_reliability_gate_sabl${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_reliability_gate.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  R0:              $RUN_ROOT/test/tinyperson_yolov5m_baseline${SUFFIX}/"
log "  Concat+SABL:     $RUN_ROOT/test/tinyperson_yolov5m_channel_pooled_concat_sabl${SUFFIX}/"
log "  Gate+SABL:       $RUN_ROOT/test/tinyperson_yolov5m_reliability_gate_sabl${SUFFIX}/"

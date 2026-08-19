#!/usr/bin/env bash
# TinyPerson on the fresh, paper-comparable dataset (HESOD-Experiment-Plan.md
# SS6): both train and test now use the "with_dense" sets (794 train / 816
# test, exact match to the paper's stated counts) -- our original
# mini_annotations pipeline silently excluded the dense crowd-scene images
# (745/786 instead). Two arms: R0 (does training-set completeness alone close
# any of the ~6pp APt50 gap?) and concat+SABL+ISPPHead (does the project's
# best-known recipe, including the lightweight head, do the same or better?).
# Plain concat+SABL (no head swap) is skipped here, same reasoning as the
# SeaDronesSeeV2/UAVDT rosters -- go straight to the more informative
# comparison rather than the intermediate arm.
#
# Requires the fresh dataset already converted:
#   cd /root/BCRS/hesod/backends/hesod
#   python scripts/data_prepare.py --dataset /root/autodl-tmp/TinyPerson_fresh
# and a data yaml at /root/autodl-tmp/TinyPerson_fresh.yaml.
#
# Runs at EPOCHS=50 (paper's own protocol, clean comparison) by default;
# pass EPOCHS=100 for the extended-training variant. RUN_ROOT auto-suffixes
# by epoch count so 50 and 100 never collide/skip each other via the
# checkpoint-exists guard.
#
# Usage:
#   SMOKE=1 bash run_tinyperson_fresh_r0.sh
#   EPOCHS=50  nohup bash run_tinyperson_fresh_r0.sh > /root/tinyperson_fresh_50.log 2>&1 & disown
#   EPOCHS=100 nohup bash run_tinyperson_fresh_r0.sh > /root/tinyperson_fresh_100.log 2>&1 & disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="/root/autodl-tmp/TinyPerson_fresh"
DATA_YAML="/root/autodl-tmp/TinyPerson_fresh.yaml"
HYP="data/hyps/hyp.tinyperson.yaml"
CLASSES="person"

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  EPOCHS=1
  SUFFIX="_smoke"
  log_prefix="SMOKE"
  RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs_freshtp_smoke}"
else
  EPOCHS="${EPOCHS:-50}"
  SUFFIX=""
  log_prefix="REAL"
  RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs_freshtp_${EPOCHS}ep}"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$log_prefix] $*"; }

log "epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML run_root=$RUN_ROOT"

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
    log "WARNING: $run_name produced no predictions (expected at 1 epoch under SMOKE) -- skipping audit/vt_diagnose/official-eval"
  else
    log "Auditing $run_name"
    python "$SCRIPT_DIR/audit_buckets.py" \
      --pred "$results_dir/best_predictions.json" \
      --labels "$DATA_ROOT/test" --images "$DATA_ROOT/test" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed, continuing"

    log "vt_diagnose: $run_name"
    python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
      --labels-dir "$DATA_ROOT/test" --images-dir "$DATA_ROOT/test" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed, continuing"

    if [ "$SMOKE" = "1" ]; then
      log "Skipping official TinyPerson evaluator under SMOKE"
    else
      log "Official TinyPerson evaluator (APt50/APs50, paper-comparable protocol): $run_name"
      # Uses tiny_set_test_with_dense.json (816 images) to match what's
      # actually being tested on now, not the old 786-image tiny_set_test_all.json.
      python "$SCRIPT_DIR/tinyperson_eval/eval_tinyperson_official.py" \
        --pred "$results_dir/best_predictions.json" \
        --gt "$DATA_ROOT/annotations/tiny_set_test_with_dense.json" \
        2>&1 | tee "$results_dir/${run_name}_official_eval.log" || log "WARNING: official evaluator failed, continuing"
    fi
  fi
}

# R0: does training-set completeness alone close any of the ~6pp APt50 gap?
run_arm "tinyperson_yolov5m_baseline_fresh${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m.yaml"

# Concat+SABL+ISPPHead: the project's best-known TinyPerson recipe (concat
# selector + SABL box loss + ISPPHead lightweight Detect head), re-run on
# the fresh, complete dataset. Reuses the existing config -- head_type
# selection doesn't depend on which dataset instance is used.
run_arm "tinyperson_yolov5m_channel_pooled_concat_sabl_isphead_fresh${SUFFIX}" \
  "models/cfg/esod/tinyperson_yolov5m_channel_pooled_concat_isphead.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  R0 fresh ($EPOCHS epochs):                   $RUN_ROOT/test/tinyperson_yolov5m_baseline_fresh${SUFFIX}/"
log "  Concat+SABL+ISPPHead fresh ($EPOCHS epochs):  $RUN_ROOT/test/tinyperson_yolov5m_channel_pooled_concat_sabl_isphead_fresh${SUFFIX}/"
log "  Compare against the original mini_annotations R0 (55.26/71.23 APt50/APs50, HESOD-Experiment-Plan.md SS2)"
log "  and concat+SABL+ISPPHead's original mAP50:95/Very-Tiny-recall (0.231/76.75%, HESOD-Lightweight-Detector-Review-and-Roadmap.md SS5.2)."

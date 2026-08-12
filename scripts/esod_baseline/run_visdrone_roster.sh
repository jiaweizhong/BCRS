#!/usr/bin/env bash
# Train and evaluate the locked VisDrone selector roster on one protocol.
#
# Primary comparison:
#   - identical VisDrone_v2 data, YOLOv5m backbone/head, hyp, image size,
#     pretrained initialization, epochs, and batch size;
#   - E1.0 uses released-code weighted BCE;
#   - E2.* arms use the same HESOD coverage loss;
#   - every checkpoint is evaluated both with upstream threshold routing and
#     at the same exact Top-K budget.  The Top-K run enables SparseHead by
#     default, as required by HESOD-Proposal.md's locked roster.
#
# The optional paper-loss arm isolates the publication/code discrepancy:
# paper text says focal:dice=20:1, while released ESOD code executes weighted
# BCE and leaves focal/dice commented out.
#
# Usage:
#   bash scripts/esod_baseline/run_visdrone_roster.sh [gpu]
#
# Useful environment overrides:
#   TOP_K=32 EPOCHS=50 BATCH=8 REUSE_CHECKPOINTS=1 INCLUDE_PAPER=1
#   SPARSE_HEAD=1 RUN_ROOT=/root/esod_roster_runs SKIP_AUDIT=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
HESOD_REPO="${HESOD_REPO:-$SCRIPT_DIR/../../hesod/backends/hesod}"
DATA="${DATA:-/root/autodl-tmp/VisDrone_v2.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/VisDrone_v2}"
RUN_ROOT="${RUN_ROOT:-$HOME/hesod_roster_runs}"
WEIGHTS="${WEIGHTS:-weights/pretrained/yolov5m.pt}"
HYP="${HYP:-data/hyps/hyp.visdrone.yaml}"
IMG_SIZE="${IMG_SIZE:-1536}"
BATCH="${BATCH:-8}"
EPOCHS="${EPOCHS:-50}"
TOP_K="${TOP_K:-32}"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-1}"
INCLUDE_PAPER="${INCLUDE_PAPER:-1}"
SPARSE_HEAD="${SPARSE_HEAD:-1}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
require_file() { [ -f "$1" ] || { echo "ERROR: required file not found: $1" >&2; exit 1; }; }
require_dir() { [ -d "$1" ] || { echo "ERROR: required directory not found: $1" >&2; exit 1; }; }

ensure_weights() {
  local weights_file
  if [[ "$WEIGHTS" = /* ]]; then
    weights_file="$WEIGHTS"
  else
    weights_file="$HESOD_REPO/$WEIGHTS"
  fi

  if [ ! -f "$weights_file" ]; then
    if [ "$WEIGHTS" != "weights/pretrained/yolov5m.pt" ]; then
      echo "ERROR: custom pretrained weights not found: $weights_file" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$weights_file")"
    log "yolov5m.pt not found, downloading pretrained COCO initialization"
    curl -fL -o "$weights_file" \
      "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt"
  fi
  require_file "$weights_file"
}

audit_eval() {
  local eval_name="$1" ckpt="$2" eval_dir="$3"
  shift 3
  local routing_args=("$@")
  if [ "$SKIP_AUDIT" = "1" ]; then
    log "SKIP_AUDIT=1: skipping detector recall and selector BPRbox for $eval_name"
    return 0
  fi
  local pred_json="$eval_dir/best_predictions.json"
  local patches_json="$eval_dir/${eval_name}_selected_patches.json"
  require_file "$pred_json"
  python "$SCRIPT_DIR/audit_buckets.py" \
    --pred "$pred_json" \
    --labels "$DATASET_ROOT/labels/val" --images "$DATASET_ROOT/images/val" \
    2>&1 | tee "$eval_dir/${eval_name}_detector_recall.log"
  python "$SCRIPT_DIR/dump_selected_patches.py" \
    --esod-repo "$HESOD_REPO" --data "$DATA" --weights "$ckpt" \
    --img-size "$IMG_SIZE" --batch-size "$BATCH" --device "$GPU" --task val \
    "${routing_args[@]}" --out "$patches_json"
  python "$SCRIPT_DIR/audit_selector_coverage.py" \
    --patches "$patches_json" --pred "$pred_json" \
    --labels "$DATASET_ROOT/labels/val" --images "$DATASET_ROOT/images/val" \
    2>&1 | tee "$eval_dir/${eval_name}_selector_audit.log"
}

require_file "$DATA"
require_dir "$DATASET_ROOT/images/val"
require_dir "$DATASET_ROOT/labels/val"
require_file "$HESOD_REPO/$HYP"
ensure_weights

# Regenerate deterministic Gaussian masks. Merely verifying existing files is
# insufficient because a previous SAM run may have used the same dataset root.
python "$SCRIPT_DIR/gen_masks.py" \
  --esod-repo "$HESOD_REPO" --dataset-root "$DATASET_ROOT" \
  --splits train val --mask-mode gaussian --cls-ratio --overwrite

SPARSE_ARGS=()
if [ "$SPARSE_HEAD" = "1" ]; then
  SPARSE_ARGS=(--sparse-head)
fi

train_arm() {
  local run_name="$1" cfg="$2" selector_loss="$3"
  local ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"
  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"

  if [ "$REUSE_CHECKPOINTS" = "1" ] && [ -f "$ckpt" ]; then
    log "Reusing checkpoint: $ckpt"
  else
    log "Training $run_name ($selector_loss)"
    (
      cd "$HESOD_REPO"
      python train.py \
        --data "$DATA" --cfg "$cfg" --weights "$WEIGHTS" --hyp "$HYP" \
        --batch-size "$BATCH" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
        --selector-loss "$selector_loss" \
        --project "$RUN_ROOT/train" --name "$run_name" --exist-ok
    ) 2>&1 | tee "$results_dir/${run_name}_train.log"
  fi
  require_file "$ckpt"

  # Native upstream threshold routing: useful for paper/code reproduction and
  # for observing each selector's unconstrained patch demand.
  log "Evaluating $run_name with upstream threshold routing"
  (
    cd "$HESOD_REPO"
    python test.py \
      --data "$DATA" --weights "$ckpt" --batch-size "$BATCH" \
      --img-size "$IMG_SIZE" --device "$GPU" --save-json \
      --project "$RUN_ROOT/test" --name "$run_name" --exist-ok
  ) 2>&1 | tee "$results_dir/${run_name}_threshold_test.log"

  audit_eval "$run_name" "$ckpt" "$results_dir"

  # Locked-roster comparison: exact same K for every selector, with SparseHead
  # enabled unless explicitly disabled for a diagnostic compatibility run.
  local topk_name="${run_name}_topk${TOP_K}"
  local topk_dir="$RUN_ROOT/test/$topk_name"
  mkdir -p "$topk_dir"
  log "Evaluating $run_name at exact Top-K=$TOP_K (SparseHead=$SPARSE_HEAD)"
  (
    cd "$HESOD_REPO"
    python test.py \
      --data "$DATA" --weights "$ckpt" --batch-size "$BATCH" \
      --img-size "$IMG_SIZE" --device "$GPU" --save-json \
      --top-k "$TOP_K" "${SPARSE_ARGS[@]}" \
      --project "$RUN_ROOT/test" --name "$topk_name" --exist-ok
  ) 2>&1 | tee "$topk_dir/${topk_name}_test.log"

  audit_eval "$topk_name" "$ckpt" "$topk_dir" --top-k "$TOP_K"

  log "Measuring $run_name at exact Top-K=$TOP_K"
  (
    cd "$HESOD_REPO"
    python test.py \
      --data "$DATA" --weights "$ckpt" --batch-size 1 \
      --img-size "$IMG_SIZE" --device "$GPU" --task measure \
      --top-k "$TOP_K" "${SPARSE_ARGS[@]}" \
      --project "$RUN_ROOT/measure" --name "$topk_name" --exist-ok
  ) 2>&1 | tee "$topk_dir/${topk_name}_measure.log"

  require_file "$RUN_ROOT/measure/$topk_name/buckets.json"
  cp "$RUN_ROOT/measure/$topk_name/buckets.json" "$topk_dir/buckets.json"
}

# E1.0 and the locked five-arm roster.  E2.1 deliberately reuses the baseline
# architecture so it isolates coverage supervision from spectral evidence.
train_arm visdrone_e10_upstream \
  models/cfg/esod/visdrone_yolov5m.yaml upstream

if [ "$INCLUDE_PAPER" = "1" ]; then
  train_arm visdrone_e10_paper_loss \
    models/cfg/esod/visdrone_yolov5m.yaml paper
fi

train_arm visdrone_e21_semantic_coverage \
  models/cfg/esod/visdrone_yolov5m.yaml coverage
train_arm visdrone_e25_spectral_only \
  models/cfg/esod/visdrone_yolov5m_spectral_only.yaml coverage
train_arm visdrone_e24_gated \
  models/cfg/esod/visdrone_yolov5m_dual_evidence.yaml coverage
train_arm visdrone_e23_concat \
  models/cfg/esod/visdrone_yolov5m_dual_evidence_concat.yaml coverage
train_arm visdrone_e29_channel_pooled_concat \
  models/cfg/esod/visdrone_yolov5m_channel_pooled_concat.yaml coverage

log "Roster complete. Results: $RUN_ROOT/test"

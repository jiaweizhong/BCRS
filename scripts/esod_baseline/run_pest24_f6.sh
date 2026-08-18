#!/usr/bin/env bash
# F6 pilot: reliability gate + rescue-ranking + conditional-gate regularization
# (HESOD-Agri-Proposal.md SS4.2.2), the one pre-registered configuration
# agreed before touching this arm -- not a hyperparameter sweep. One retry
# (adjusting --lambda-rescue/--lambda-cond) is allowed if this first attempt
# fails the stop/go conditions below; use a distinct run_name for it
# (e.g. pest24_yolov5m_reliability_gate_f6_retry) rather than overwriting
# this run's artifacts.
#
# This is a STANDALONE script, not appended to run_pest24.sh: run_arm's
# eval/measure/audit steps run unconditionally on every invocation (only
# training is skip-if-checkpoint-exists), so appending F6 here would force a
# full re-run of eval for R0/R1/R2/R3/F5/Gate+SABL every time this pilot is
# invoked or retried -- exactly the problem hit earlier this session running
# run_pest24.sh just to pick up F5's eval. Shares the same preamble/run_arm
# pattern as run_pest24.sh verbatim (small accepted duplication for a one-off
# pilot, not a general N-arm harness).
#
# Stop/go conditions (evaluate against R2/F1's numbers: AP@.5:.95=0.348,
# TFR=8.74%; HESOD-Agri-Experiment-Plan.md SS11.2.5/SS11.2.6), at matched
# Occupy/GFLOPs vs. F1:
#   (a) selector-dropped count down >=10% relative OR Very Tiny recall up
#       >=1pp vs. F5 (vt_diagnose.py pest24_yolov5m_reliability_gate_f6);
#   (b) TFR <= 8.74% (tfr_diagnose.py pest24_yolov5m_reliability_gate_f6);
#   (c) AP@.5:.95 not down more than ~0.2-0.3pp vs. F5's 0.348;
#   (d) gate activation a_i not saturated (not -> 0 or -> 1 almost
#       everywhere on tiny-object/texture-background regions);
#   (e) at most ONE retry adjusting lambda_rescue/lambda_cond -- still
#       failing after that, stop the gate line and use concat+SABL (R3).
#
# Usage: nohup bash run_pest24_f6.sh > /root/pest24_f6.log 2>&1 &
#        (or run inside tmux/screen)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-1024}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="/root/autodl-tmp/Pest24_v1"
DATA_YAML="/root/autodl-tmp/Pest24_v1.yaml"
PEST24_CLASSES="Bollworm,Meadow borer,Gryllotalpa orientalis,Little Gecko,Agriotes fuscicollis Miwa,Nematode trench,Athetis lepigone,Scotogramma trifolii Rottemberg,Armyworm,Spodoptera cabbage,Anomala corpulenta,Spodoptera exigua,Plutella xylostella,holotrichia parallela,Rice planthopper,Yellow tiger,Land tiger,eight-character tiger,holotrichia oblita,Stem borer,Striped rice bore,Rice Leaf Roller,Spodoptera litura,Melahotus"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -d "$DATA_ROOT/images/train" ]; then
  log "Reorganizing Pest24 raw data -> $DATA_ROOT"
  python "$SCRIPT_DIR/reorganize_pest24.py" \
    --raw-root /root/data/Pest24/VOCdevkit/voc2007 \
    --out-root "$DATA_ROOT" \
    --splits train val test
else
  log "Reorganized data already present at $DATA_ROOT, skipping reorganize"
fi

log "Generating/verifying masks (Gaussian-only, no SAM)"
python "$SCRIPT_DIR/gen_masks.py" \
  --esod-repo "$ESOD_REPO" \
  --dataset-root "$DATA_ROOT" \
  --splits train val test

if [ ! -f "$DATA_YAML" ]; then
  log "Writing $DATA_YAML"
  cat > "$DATA_YAML" << 'YAML'
train: /root/autodl-tmp/Pest24_v1/images/train
val: /root/autodl-tmp/Pest24_v1/images/val
test: /root/autodl-tmp/Pest24_v1/images/test
nc: 24
names: ['Bollworm', 'Meadow borer', 'Gryllotalpa orientalis', 'Little Gecko',
        'Agriotes fuscicollis Miwa', 'Nematode trench', 'Athetis lepigone',
        'Scotogramma trifolii Rottemberg', 'Armyworm', 'Spodoptera cabbage',
        'Anomala corpulenta', 'Spodoptera exigua', 'Plutella xylostella',
        'holotrichia parallela', 'Rice planthopper', 'Yellow tiger',
        'Land tiger', 'eight-character tiger', 'holotrichia oblita',
        'Stem borer', 'Striped rice bore', 'Rice Leaf Roller',
        'Spodoptera litura', 'Melahotus']
YAML
fi

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
      --hyp data/hyps/hyp.pest24.yaml \
      --batch-size "$BATCH" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
      "${extra_flags[@]}" \
      --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_train.log"
  fi

  # Defense in depth alongside `set -euo pipefail`: catches a GPU dropping
  # mid-training (e.g. a cloud instance rebooting into a no-GPU state)
  # explicitly, instead of eval/measure/audit cascading into a handful of
  # near-instant, content-free failure logs if pipefail alone doesn't catch
  # it. See run_pest24_resolution.sh's identical guard for the incident this
  # was added after.
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

  log "Auditing $run_name"
  python "$SCRIPT_DIR/audit_buckets.py" \
    --pred "$results_dir/best_predictions.json" \
    --labels "$DATA_ROOT/labels/test" --images "$DATA_ROOT/images/test" \
    --classes "$PEST24_CLASSES" \
    2>&1 | tee "$results_dir/${run_name}_audit.log"
}

# F6: reliability gate + rescue-ranking + conditional-gate regularization.
# Pre-registered pilot config (HESOD-Agri-Proposal.md SS4.2.2, unvalidated
# candidate defaults): tau_low=0.3, margin=1.0, lambda_rescue=0.5,
# lambda_cond=0.1, alpha=1.0 (alpha fixed at model-construction time via the
# yaml, not a CLI flag). CIoU only -- no SABL yet, to avoid confounding
# attribution (mirrors R2/F5 being CIoU-only before R3/Gate+SABL).
run_arm "pest24_yolov5m_reliability_gate_f6" \
  "models/cfg/esod/pest24_yolov5m_reliability_gate.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 \
  --lambda-rescue 0.5 --lambda-cond 0.1 --tau-low 0.3 --rescue-margin 1.0

log "===== ALL DONE ====="
log "  F6 reliability_gate_f6: $RUN_ROOT/test/pest24_yolov5m_reliability_gate_f6/"

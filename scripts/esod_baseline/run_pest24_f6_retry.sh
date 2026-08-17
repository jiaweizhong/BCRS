#!/usr/bin/env bash
# F6 pilot, attempt 2 of 2 (final retry per run_pest24_f6.sh's pre-registered
# rule -- HESOD-Agri-Experiment-Plan.md SS11.2.7).
#
# Attempt 1 (tau_low=0.3 margin=1.0 lambda_rescue=0.5 lambda_cond=0.1) failed
# its own primary condition: Very Tiny recall FELL 4.24pp (64.03%->59.79%)
# instead of rising, even though every aggregate metric improved (best-ever
# AP/AP50/total-recall/TFR of any Pest24 arm). Diagnosis (SS11.2.7): P_rescue
# membership (y_i=1, q_i<tau_low) is not size-weighted, and Tiny+Small GT
# boxes outnumber Very Tiny ones ~40:1 (56,778 vs 1,415) -- L_rescue's
# gradient is plausibly diluted by the much larger Tiny/Small population
# rather than preferentially lifting Very Tiny cells.
#
# This attempt only adjusts lambda_rescue/lambda_cond, per the rule's allowed
# scope (tau_low/margin/architecture unchanged):
#   lambda_cond:   0.1 -> 0.05  (half the suppression pressure, in case it was
#                                crowding out the rescue signal)
#   lambda_rescue: 0.5 -> 1.0   (double the pull, in case attempt 1's effect
#                                was real but too weak relative to the
#                                Tiny/Small population it competes against)
#
# Flagged honestly (do not oversell this run): this CANNOT fix the diagnosed
# size-imbalance in P_rescue itself -- that needs a coverage-loss-style area
# weight (compute_coverage_loss's ref_area_cells/area_cells pattern), out of
# this retry's pre-registered scope. This tests whether a cheap reweighting
# is enough, not a guaranteed fix. If this also fails condition (a), the
# gate line closes per the rule -- fall back to concat+SABL (R3), the
# ladder's actual best arm (HESOD-Agri-Experiment-Plan.md SS11.2.6).
#
# Distinct run_name (pest24_yolov5m_reliability_gate_f6_retry) so attempt 1's
# artifacts are preserved side by side, not overwritten.
#
# Usage: nohup bash run_pest24_f6_retry.sh > /root/pest24_f6_retry.log 2>&1 &

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

  # Very-Tiny selector-dropped/head-localization/confusion breakdown and TFR
  # (HESOD-Agri-Experiment-Plan.md SS1.3.1/SS11.2.7) -- needed to check this
  # arm against the four pre-registered stop/go conditions, not optional
  # extras. Previously run by hand from an untracked /root/*.py copy; now
  # git-tracked here so they travel with the repo instead of living only on
  # one GPU box.
  log "vt_diagnose: $run_name"
  python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
    2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log"

  log "tfr_diagnose: $run_name"
  python "$SCRIPT_DIR/tfr_diagnose.py" "$run_name" \
    2>&1 | tee "$results_dir/${run_name}_tfr_diagnose.log"
}

run_arm "pest24_yolov5m_reliability_gate_f6_retry" \
  "models/cfg/esod/pest24_yolov5m_reliability_gate.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 \
  --lambda-rescue 1.0 --lambda-cond 0.05 --tau-low 0.3 --rescue-margin 1.0

log "===== ALL DONE ====="
log "  F6 retry: $RUN_ROOT/test/pest24_yolov5m_reliability_gate_f6_retry/"

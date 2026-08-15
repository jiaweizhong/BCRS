#!/usr/bin/env bash
# Pest24 generalization test: does the HESOD approach (ESOD baseline, HESOD's
# channel-pooled-concat selector, SSABNet's SABL box loss) transfer from
# aerial imagery (VisDrone/UAVDT/TinyPerson) to agricultural pest detection?
# Three single-variable arms, same recipe otherwise:
#   1. baseline            -- pest24_yolov5m.yaml
#   2. channel_pooled_concat -- pest24_yolov5m_channel_pooled_concat.yaml,
#      --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 (matches
#      the VisDrone/TinyPerson channel-pooled-concat recipe exactly, see
#      HESOD-Experiment-Plan.md SS2/SS4.3)
#   3. channel_pooled_concat_sabl -- same cfg as (2) plus --box-loss sabl
#      (SABL only changes the loss function, not the architecture -- see
#      loss.py's sabl_loss() and HESOD-Experiment-Plan.md's paper-review
#      section for why this was picked over size_weighted)
#
# Data: Pest24 (VOCdevkit/voc2007 layout), reorganized via reorganize_pest24.py
# into images/<split>/, labels/<split>/, masks generated via gen_masks.py
# (Gaussian-only, no SAM -- this is a fast generalization check, not a
# paper-comparable reproduction). nc=24, native 800x600, trained at img-size
# 1024 (modest upscale, not the aggressive upscale VisDrone/TinyPerson use,
# since Pest24's native resolution is already much lower to start with -- see
# HESOD-Experiment-Plan.md's Pest24 section for the size-distribution numbers
# behind this choice). hyp.pest24.yaml is hyp.uavdt.yaml verbatim -- not
# hand-tuned, deliberately, for a first read on transfer before spending
# effort on dataset-specific tuning.
#
# Usage: nohup bash run_pest24.sh > /root/pest24.log 2>&1 &
#        (or run inside tmux/screen)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-1024}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"  # dev tree -- ChannelPooledConcatEvidenceSegmenter and --box-loss sabl only exist here
DATA_ROOT="/root/autodl-tmp/Pest24_v1"
DATA_YAML="/root/autodl-tmp/Pest24_v1.yaml"
# audit_buckets.py --classes defaults to VisDrone's 10-class list and raises
# on any class id outside [0,9] -- must pass Pest24's 24 explicitly or the
# audit step crashes on the first class-10+ label it hits.
PEST24_CLASSES="Bollworm,Meadow borer,Gryllotalpa orientalis,Little Gecko,Agriotes fuscicollis Miwa,Nematode trench,Athetis lepigone,Scotogramma trifolii Rottemberg,Armyworm,Spodoptera cabbage,Anomala corpulenta,Spodoptera exigua,Plutella xylostella,holotrichia parallela,Rice planthopper,Yellow tiger,Land tiger,eight-character tiger,holotrichia oblita,Stem borer,Striped rice bore,Rice Leaf Roller,Spodoptera litura,Melahotus"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ============================================================
# One-time data prep: reorganize + generate masks (shared by all 3 arms)
# ============================================================
if [ ! -d "$DATA_ROOT/images/train" ]; then
  log "Reorganizing Pest24 raw data -> $DATA_ROOT"
  python "$SCRIPT_DIR/reorganize_pest24.py" \
    --raw-root /root/autodl-tmp/Pest24/VOCdevkit/voc2007 \
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

  local ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"

  log "Evaluating $run_name"
  python test.py \
    --data "$DATA_YAML" --weights "$ckpt" \
    --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" --save-json \
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

run_arm "pest24_yolov5m_baseline" \
  "models/cfg/esod/pest24_yolov5m.yaml"

run_arm "pest24_yolov5m_channel_pooled_concat" \
  "models/cfg/esod/pest24_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0

run_arm "pest24_yolov5m_channel_pooled_concat_sabl" \
  "models/cfg/esod/pest24_yolov5m_channel_pooled_concat.yaml" \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  baseline:                     $RUN_ROOT/test/pest24_yolov5m_baseline/"
log "  channel_pooled_concat:        $RUN_ROOT/test/pest24_yolov5m_channel_pooled_concat/"
log "  channel_pooled_concat_sabl:   $RUN_ROOT/test/pest24_yolov5m_channel_pooled_concat_sabl/"

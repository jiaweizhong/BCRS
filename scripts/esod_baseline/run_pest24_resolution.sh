#!/usr/bin/env bash
# Resolution comparison thread for the TP-YOLO discussion
# (HESOD-Agri-Experiment-Plan.md's TP-YOLO section): three arms, run in
# sequence, each with its own explicit --img-size (NOT a script-level
# env var -- unlike run_pest24.sh/run_pest24_f6.sh, this script trains
# more than one resolution in a single invocation, so run_arm() takes
# img_size as its own argument).
#
#   1. A0 @ 1024   -- dense YOLOv5m, no routing, matches our existing
#                     R0-F5/Gate+SABL arms' resolution
#   2. A0 @ 640    -- same dense architecture, TP-YOLO's likely (not
#                     confirmed in-paper) resolution -- isolates the pure
#                     resolution effect against A0@1024, with no routing
#                     confound at all
#   3. Concat+SABL @ 640 -- our current best routed arm (R3), retrained at
#                     640 -- tests whether the routing method itself still
#                     pays off at lower resolution
#
# A0 uses models/cfg/vanilla/pest24_yolov5m.yaml (HESOD-Agri-Proposal.md
# SS5's A0 row) -- identical to models/cfg/esod/pest24_yolov5m.yaml minus
# the DWConv+Segmenter+HeatMapParser 3-layer insert, so it isolates exactly
# the routing-vs-no-routing variable. Its eval step also passes
# --save-regions for uniformity with the routed arm below; this is a no-op
# for A0 (test.py's region-saving code is nested inside an
# `isinstance(p_det, tuple)` check that a dense model's output never
# satisfies, so no selected_regions.json gets written for A0, which is
# correct -- there is no routing to record).
#
# Arm 3 REQUIRES the uni_slicer fix (models/common.py,
# HeatMapParser.uni_slicer) -- an unmodified 640 run crashes on that
# function's old P3-divisible-by-32 assertion (P3=80 at 640, 80%32!=0). The
# fix pads the routed feature map up to an exact ratio*cluster_wh multiple
# before chunking instead of asserting the padding away.
#
# IMPORTANT: arm 3's action space is NOT the 64 cells the 1024-resolution
# arms use. At 640, P3=80, cluster_wh=make_divisible(80/8,4)=12, giving
# ratio_x=ratio_y=ceil(80/12)=7 -> 7x7=49 cells (confirmed by direct
# calculation) -- a different, smaller routing granularity than 1024's
# clean 8x8=64. Do not compare arm 3's Occupy/K numbers directly against
# the 1024-resolution arms' Occupy/K -- only AP/AP50/GFLOPs are valid for
# the cross-resolution comparison; Occupy/K stays within-resolution only.
#
# Standalone script, not appended to run_pest24.sh/run_pest24_f6.sh, for the
# same reason those are standalone: run_arm's eval/measure/audit steps run
# unconditionally on every invocation (only training is skip-if-checkpoint-
# exists), so appending here would force a full re-run of eval for
# R0-F5/Gate+SABL/F6 on every invocation of this script.
#
# Usage: nohup bash run_pest24_resolution.sh > /root/pest24_resolution.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-8}"
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
  local run_name="$1" model_cfg="$2" img_size="$3"
  shift 3
  local extra_flags=("$@")

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$ESOD_REPO"

  local ckpt="$RUN_ROOT/train/$run_name/weights/best.pt"
  if [ -f "$ckpt" ]; then
    log "===== $run_name already trained (found $ckpt), skipping training ====="
  else
    log "===== Training $run_name (img-size=$img_size) ====="
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights weights/pretrained/yolov5m.pt \
      --hyp data/hyps/hyp.pest24.yaml \
      --batch-size "$BATCH" --img-size "$img_size" --epochs "$EPOCHS" --device "$GPU" \
      "${extra_flags[@]}" \
      --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_train.log"
  fi

  # Defense in depth alongside `set -euo pipefail`: a GPU dropping mid-training
  # (e.g. a cloud instance rebooting into a no-GPU state) can make train.py
  # exit non-zero without writing a checkpoint -- pipefail already stops the
  # script at that point in the normal case, but this check catches it
  # explicitly rather than relying on that alone, and gives a clear error
  # instead of eval/measure/audit cascading into a handful of near-instant,
  # content-free failure logs (this exact failure mode happened once this
  # session, on the concat_sabl@640 arm, from a hand-run command that had
  # `set -e` but not `pipefail`).
  if [ ! -f "$ckpt" ]; then
    log "FATAL: $run_name training finished but $ckpt does not exist -- aborting before eval"
    exit 1
  fi

  log "Evaluating $run_name"
  python test.py \
    --data "$DATA_YAML" --weights "$ckpt" --task test \
    --batch-size "$BATCH" --img-size "$img_size" --device "$GPU" --save-json --save-regions \
    --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_test.log"

  log "Measuring $run_name (GFLOPs/FPS, batch=1)"
  python test.py \
    --data "$DATA_YAML" --weights "$ckpt" \
    --batch-size 1 --img-size "$img_size" --device "$GPU" --task measure \
    --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
    2>&1 | tee "$results_dir/${run_name}_measure.log"

  log "Auditing $run_name"
  python "$SCRIPT_DIR/audit_buckets.py" \
    --pred "$results_dir/best_predictions.json" \
    --labels "$DATA_ROOT/labels/test" --images "$DATA_ROOT/images/test" \
    --classes "$PEST24_CLASSES" \
    2>&1 | tee "$results_dir/${run_name}_audit.log"
}

# 1. A0 @ 1024 -- dense, matches our existing arms' resolution
run_arm "pest24_yolov5m_a0_1024" \
  "models/cfg/vanilla/pest24_yolov5m.yaml" 1024

# 2. A0 @ 640 -- dense, TP-YOLO's likely resolution, isolates pure resolution effect
run_arm "pest24_yolov5m_a0_640" \
  "models/cfg/vanilla/pest24_yolov5m.yaml" 640

# 3. Concat+SABL @ 640 -- our best routed arm, retrained at 640 (needs the
# uni_slicer fix; action space is 7x7=49 cells, not 64 -- see header comment)
run_arm "pest24_yolov5m_channel_pooled_concat_sabl_640" \
  "models/cfg/esod/pest24_yolov5m_channel_pooled_concat.yaml" 640 \
  --selector-loss coverage --lambda-cov 0.5 --pos-weight 2.0 --box-loss sabl

log "===== ALL DONE ====="
log "  A0 @1024:            $RUN_ROOT/test/pest24_yolov5m_a0_1024/"
log "  A0 @640:              $RUN_ROOT/test/pest24_yolov5m_a0_640/"
log "  Concat+SABL @640:    $RUN_ROOT/test/pest24_yolov5m_channel_pooled_concat_sabl_640/"

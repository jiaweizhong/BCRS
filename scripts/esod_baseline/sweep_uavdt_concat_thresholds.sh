#!/usr/bin/env bash
# Cheap, no-retrain sweep to test whether uavdt_yolov5m_channel_pooled_concat
# ("Concat-only", HESOD-Experiment-Plan.md SS3.2 arm 5)'s worse-than-either-
# single-branch mAP is explained by over-selection: HeatMapParser routing too
# many low-confidence candidate regions (confirmed via best_predictions.json
# row count, SS3.4 point 2 -- 7.49M rows vs R0's 2.35M for the same 373,997 GT).
#
# --hm-threshold and --top-k are applied to every HeatMapParser in the model
# AFTER loading the checkpoint (test.py:147-158) -- pure inference-time
# overrides, no retraining needed. If raising the threshold / capping top-k
# recovers mAP toward or past R0's 0.385/0.214, that supports the diagnosis
# (too many, too-low-quality candidates dragging precision down); if mAP
# stays flat or drops further even as the candidate pool shrinks, the
# problem is more likely in what got learned (the fusion itself), not how
# many regions get through at inference.
#
# Prerequisite: uavdt_yolov5m_channel_pooled_concat must have a completed,
# freshly-retrained checkpoint (the arm (3)/(5) noise-check rerun this sweep
# depends on -- ARMS=uavdt_yolov5m_spectral_only,uavdt_yolov5m_channel_pooled_concat)
# AND the GPU must be idle. Check before running:
#   ps aux | grep -E "[t]rain.py|[r]un_uavdt"
# Do not launch alongside any other training/eval job on the same GPU.
#
# Usage:
#   bash sweep_uavdt_concat_thresholds.sh [gpu]
#   nohup bash sweep_uavdt_concat_thresholds.sh > /root/uavdt_concat_sweep.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/UAVDT_fresh.yaml}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/UAVDT_fresh}"
CLASSES="car,truck,bus"
IMG_SIZE=1280
BATCH=8

CKPT="$RUN_ROOT/train/uavdt_yolov5m_channel_pooled_concat/weights/best.pt"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SWEEP] $*"; }

if [ ! -f "$CKPT" ]; then
  log "FATAL: $CKPT does not exist -- the concat-only rerun must complete first"
  exit 1
fi

cd "$ESOD_REPO"

run_one() {
  local tag="$1"
  shift
  local name="uavdt_concat_sweep_${tag}"
  local results_dir="$RUN_ROOT/test/$name"
  mkdir -p "$results_dir"

  log "===== $tag (${*:-no override}) ====="
  python test.py \
    --data "$DATA_YAML" --weights "$CKPT" --task val \
    --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" --save-json \
    "$@" \
    --project "$RUN_ROOT/test" --name "$name" --exist-ok \
    2>&1 | tee "$results_dir/${name}_test.log"

  python "$SCRIPT_DIR/audit_buckets.py" \
    --pred "$results_dir/best_predictions.json" \
    --labels "$DATA_ROOT/labels/test" --images "$DATA_ROOT/images/test" \
    --classes "$CLASSES" \
    2>&1 | tee "$results_dir/${name}_audit.log" || log "WARNING: audit failed for $tag, continuing"
}

# Baseline (no override) for reference alongside the swept values below.
run_one "baseline"

# --hm-threshold sweep: routing threshold is 0.3 for UAVDT by architecture
# default -- raise it in steps to see how many regions get cut and what
# happens to mAP.
run_one "hm040" --hm-threshold 0.4
run_one "hm050" --hm-threshold 0.5
run_one "hm060" --hm-threshold 0.6

# --top-k sweep: hard-cap selected regions per image instead of a threshold.
# Valid range is [1, 64] (test.py's own validation).
run_one "topk16" --top-k 16
run_one "topk32" --top-k 32

log "===== ALL SWEEP RUNS DONE ====="
log "Compare each \$RUN_ROOT/test/uavdt_concat_sweep_*/uavdt_concat_sweep_*_test.log's final 'all' row"
log "(P R mAP@.5 mAP@.5:.95 BPR Occupy) against R0 (0.385/0.214) and the sweep's own 'baseline' run."

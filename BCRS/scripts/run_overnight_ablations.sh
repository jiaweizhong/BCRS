#!/bin/bash
# =============================================================================
# BCRS Overnight Training Script
#
# Trains the following experiments sequentially (in estimated compute order,
# lightest first to catch failures early):
#
# Block A — VisDrone Ablation
#   [A1] E2.9  bcrs_channel_pooled_concat_visdrone — channel-pooled spectral branch + concat fusion
#   (E2.5 Spectral-Only trained in a previous run of this script — see
#   work_dirs/bcrs_spectral_only_visdrone_yolov5m/weights/best.pt — swapped
#   out here for E2.9 now that it's done; run tools/inference_sweep.sh to
#   sweep E2.5, this script no longer trains it.)
#
# Block B — UAVDT Cross-Dataset Validation
#   [B1] E1.0u esod_uavdt                          — ESOD baseline on UAVDT
#   [B2] E2.1u bcrs_dual_evidence_uavdt             — Semantic-Only (Coverage-Supervised) on UAVDT
#   [B3] E2.3u bcrs_dual_evidence_concat_uavdt      — Concat dual-evidence on UAVDT
#   [B4] E2.6u bcrs_channel_pooled_spectral_uavdt   — Channel-Pooled Spectral (Gated) on UAVDT
#   [B5] E2.9u bcrs_channel_pooled_concat_uavdt     — Channel-Pooled Concat on UAVDT (new)
#
#   NOTE (2026-08-06): B2's display name corrected — model.config uses the
#   plain Segmenter class (uavdt_yolov5m.yaml), same naming bug as VisDrone's
#   E2.1 (see BCRS-Experiment-Plan.md). All four UAVDT model yamls (baseline,
#   concat, channel-pooled, channel-pooled-concat) now uniformly use DWConv +
#   HeatMapParser threshold 0.5 — confirmed against ESOD.pdf (Table VI +
#   §IV-C prose: "we simply employ a depth-wise separable convolutional
#   block (DWConv)... DWConv achieves comparable best-possible-recalls
#   against SPP and ASPP with less computation and latency") that DWConv,
#   not SPP, is what the paper actually uses for ObjSeeker in every main
#   result. uavdt_yolov5m.yaml (baseline) had SPP + 0.3 and was the file
#   that had drifted — it has been corrected to match, not the reverse. No
#   UAVDT training had completed yet, so this fix costs nothing.
#
# Each block: train → test (K=64) to produce best_predictions.json
#
# Usage: cd /root/BCRS/BCRS && bash scripts/run_overnight_ablations.sh [UAVDT_ROOT]
#        (recommend: nohup bash scripts/run_overnight_ablations.sh > overnight.log 2>&1 &)
#
#   UAVDT_ROOT resolution order (no path is hardcoded in this script):
#     1. first positional argument, e.g.:
#          bash scripts/run_overnight_ablations.sh /root/autodl-tmp/UAVDT_processed
#     2. an already-exported UAVDT_ROOT env var, e.g.:
#          export UAVDT_ROOT=/root/autodl-tmp/UAVDT_processed
#          bash scripts/run_overnight_ablations.sh
#     3. neither present -> the script exits with an error instead of guessing.
#   Must point at a directory laid out as <root>/images/{train,test} and
#   <root>/labels/{train,test} (see configs/datasets/uavdt.yaml).
#
# Estimated time: ~6–7 h (RTX 5090, 50 epochs each, 6 models: 1 VisDrone + 5 UAVDT)
# =============================================================================

set -e

# NOTE (2026-08-06): configs/datasets/uavdt.yaml resolves ${UAVDT_ROOT:-...}
# via Python's os.environ in the bcrs subprocess — a local (non-exported) bash
# variable is invisible there, which was the root cause of Block B's earlier
# "Dataset not found" failure (the label path below picked up a value
# locally, but the training subprocess never saw it). `export` here keeps
# both sides in sync, matching the VISDRONE_ROOT pattern in README.md — no
# machine-specific path is hardcoded in this script.
UAVDT_ROOT="${1:-${UAVDT_ROOT:-}}"
if [ -z "${UAVDT_ROOT}" ]; then
  echo "ERROR: UAVDT_ROOT not set." >&2
  echo "  Usage: bash scripts/run_overnight_ablations.sh <UAVDT_ROOT>" >&2
  echo "     or: export UAVDT_ROOT=/path/to/UAVDT_processed && bash scripts/run_overnight_ablations.sh" >&2
  exit 1
fi
export UAVDT_ROOT

WORK_DIR="/root/BCRS/BCRS/work_dirs"
VISDRONE_LABELS="/root/autodl-tmp/VisDrone/labels/val"
UAVDT_LABELS="${UAVDT_ROOT}/labels/test"
AUDIT_TOOL="python tools/audit_failure_cases.py"

# ---------------------------------------------------------------------------
# Helper: train + test + audit one experiment
# ---------------------------------------------------------------------------
run_experiment() {
  local LABEL="$1"
  local CONFIG="$2"
  local CKPT="$3"
  local LABELS_DIR="$4"

  echo ""
  echo "============================================================"
  echo "  ${LABEL}"
  echo "  config: ${CONFIG}"
  echo "============================================================"

  echo "[$(date '+%H:%M:%S')] Starting training: ${LABEL}"
  bcrs train "${CONFIG}"
  echo "[$(date '+%H:%M:%S')] Training complete: ${LABEL}"

  echo "[$(date '+%H:%M:%S')] Running inference (K=64): ${LABEL}"
  bcrs test "${CONFIG}" \
    --set "test.checkpoint=${CKPT}" \
    --set test.save_json=true \
    --set test.patch_budget=64

  echo "[$(date '+%H:%M:%S')] Running audit: ${LABEL}"
  if [ -f "${CKPT%/weights/best.pt}_k64/best_predictions.json" ] 2>/dev/null; then
    # New naming convention (if inference_sweep.sh ran first)
    PRED="${CKPT%/weights/best.pt}_k64/best_predictions.json"
  else
    # Default output from bcrs test (uses config's output_dir)
    STEM=$(basename "$(dirname "$(dirname "${CKPT}")")")
    PRED="${WORK_DIR}/${STEM}/best_predictions.json"
    # Fallback: find any best_predictions.json in the work dir
    PRED=$(find "${WORK_DIR}" -maxdepth 2 -name "best_predictions.json" \
           -path "*$(basename $(dirname $(dirname ${CKPT})))*" | head -1)
  fi

  if [ -f "${PRED}" ] && [ -d "${LABELS_DIR}" ]; then
    ${AUDIT_TOOL} "${PRED}" "${LABELS_DIR}"
  else
    echo "  NOTE: Audit skipped — predictions or labels dir not found."
    echo "        pred: ${PRED}"
    echo "        labels: ${LABELS_DIR}"
  fi

  echo "[$(date '+%H:%M:%S')] DONE: ${LABEL}"
  echo "============================================================"
}

# ===========================================================================
# Announce plan
# ===========================================================================
echo "============================================================"
echo " BCRS Overnight Training Run"
echo " Start: $(date)"
echo ""
echo " Block A — VisDrone Ablation (1 model × 50 epochs)"
echo "   [A1] E2.9  Channel-Pooled Concat VisDrone"
echo ""
echo " Block B — UAVDT Cross-Dataset (5 models × 50 epochs)"
echo "   [B1] E1.0u ESOD Baseline UAVDT"
echo "   [B2] E2.1u Semantic-Only (Coverage-Supervised) UAVDT"
echo "   [B3] E2.3u Concat Dual-Evidence UAVDT"
echo "   [B4] E2.6u Channel-Pooled Spectral (Gated) UAVDT"
echo "   [B5] E2.9u Channel-Pooled Concat UAVDT"
echo ""
echo " Estimated: ~6-7 hours on RTX 5090"
echo "============================================================"

# ===========================================================================
# Block A: VisDrone Ablation — Channel-Pooled Concat (E2.9)
# ===========================================================================
echo ""
echo "##########################################################"
echo "  BLOCK A — VisDrone Ablation"
echo "##########################################################"

run_experiment \
  "[A1] E2.9 — Channel-Pooled Concat (VisDrone)" \
  "configs/experiments/bcrs_channel_pooled_concat_visdrone.yaml" \
  "${WORK_DIR}/bcrs_channel_pooled_concat_visdrone_yolov5m/weights/best.pt" \
  "${VISDRONE_LABELS}"

# ===========================================================================
# Block B: UAVDT Cross-Dataset Validation
# ===========================================================================
echo ""
echo "##########################################################"
echo "  BLOCK B — UAVDT Cross-Dataset Validation"
echo "##########################################################"

run_experiment \
  "[B1] E1.0u — ESOD Baseline (UAVDT)" \
  "configs/experiments/esod_uavdt.yaml" \
  "${WORK_DIR}/esod_uavdt_yolov5m/weights/best.pt" \
  "${UAVDT_LABELS}"

run_experiment \
  "[B2] E2.1u — Semantic-Only, Coverage-Supervised (UAVDT)" \
  "configs/experiments/bcrs_dual_evidence_uavdt.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_uavdt_yolov5m/weights/best.pt" \
  "${UAVDT_LABELS}"

run_experiment \
  "[B3] E2.3u — Concat Dual-Evidence (UAVDT)" \
  "configs/experiments/bcrs_dual_evidence_concat_uavdt.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_concat_uavdt_yolov5m/weights/best.pt" \
  "${UAVDT_LABELS}"

run_experiment \
  "[B4] E2.6u — Channel-Pooled Spectral, Gated (UAVDT)" \
  "configs/experiments/bcrs_channel_pooled_spectral_uavdt.yaml" \
  "${WORK_DIR}/bcrs_channel_pooled_spectral_uavdt_yolov5m/weights/best.pt" \
  "${UAVDT_LABELS}"

run_experiment \
  "[B5] E2.9u — Channel-Pooled Concat (UAVDT)" \
  "configs/experiments/bcrs_channel_pooled_concat_uavdt.yaml" \
  "${WORK_DIR}/bcrs_channel_pooled_concat_uavdt_yolov5m/weights/best.pt" \
  "${UAVDT_LABELS}"

# ===========================================================================
# Done
# ===========================================================================
echo ""
echo "============================================================"
echo " OVERNIGHT TRAINING COMPLETE"
echo " End: $(date)"
echo ""
echo " Trained checkpoints:"
echo "   work_dirs/bcrs_channel_pooled_concat_visdrone_yolov5m/weights/best.pt"
echo "   work_dirs/esod_uavdt_yolov5m/weights/best.pt"
echo "   work_dirs/bcrs_dual_evidence_uavdt_yolov5m/weights/best.pt"
echo "   work_dirs/bcrs_dual_evidence_concat_uavdt_yolov5m/weights/best.pt"
echo "   work_dirs/bcrs_channel_pooled_spectral_uavdt_yolov5m/weights/best.pt"
echo "   work_dirs/bcrs_channel_pooled_concat_uavdt_yolov5m/weights/best.pt"
echo ""
echo " Next steps:"
echo "   1. Run inference sweep: bash tools/inference_sweep.sh"
echo "      (add UAVDT entries to inference_sweep.sh when results are verified)"
echo "   2. Sync results to local: SERVER=xxx bash tools/sync_results.sh"
echo "============================================================"

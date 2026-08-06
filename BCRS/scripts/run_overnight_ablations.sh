#!/bin/bash
# =============================================================================
# BCRS Overnight Training Script
#
# Trains the following experiments sequentially (in estimated compute order,
# lightest first to catch failures early):
#
# Block A — VisDrone Ablation
#   [A1] E2.5  bcrs_spectral_only_visdrone    — spectral-only selector (ablation)
#
# Block B — UAVDT Cross-Dataset Validation
#   [B1] E1.0u esod_uavdt                     — ESOD baseline on UAVDT
#   [B2] E2.1u bcrs_dual_evidence_uavdt        — Gated dual-evidence on UAVDT
#   [B3] E2.3u bcrs_dual_evidence_concat_uavdt — Concat dual-evidence on UAVDT
#   [B4] E2.6u bcrs_channel_pooled_spectral_uavdt — Channel-pooled spectral on UAVDT
#
# Each block: train → test (K=64) to produce best_predictions.json
#
# Usage: cd /root/BCRS/BCRS && bash scripts/run_overnight_ablations.sh
#        (recommend: nohup bash scripts/run_overnight_ablations.sh > overnight.log 2>&1 &)
#
# Estimated time: ~5–6 h (RTX 5090, 50 epochs each, 5 models)
# =============================================================================

set -e

WORK_DIR="/root/BCRS/BCRS/work_dirs"
VISDRONE_LABELS="/root/autodl-tmp/VisDrone/labels/val"
# Adjust if UAVDT labels are in a different location
UAVDT_LABELS="${UAVDT_ROOT:-/root/autodl-tmp/UAVDT}/labels/test"
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
echo "   [A1] E2.5  Spectral-Only VisDrone"
echo ""
echo " Block B — UAVDT Cross-Dataset (4 models × 50 epochs)"
echo "   [B1] E1.0u ESOD Baseline UAVDT"
echo "   [B2] E2.1u Gated Dual-Evidence UAVDT"
echo "   [B3] E2.3u Concat Dual-Evidence UAVDT"
echo "   [B4] E2.6u Channel-Pooled Spectral UAVDT"
echo ""
echo " Estimated: ~5-6 hours on RTX 5090"
echo "============================================================"

# ===========================================================================
# Block A: VisDrone Ablation — Spectral-Only (E2.5)
# ===========================================================================
echo ""
echo "##########################################################"
echo "  BLOCK A — VisDrone Ablation"
echo "##########################################################"

run_experiment \
  "[A1] E2.5 — Spectral-Only Selector (VisDrone)" \
  "configs/experiments/bcrs_spectral_only_visdrone.yaml" \
  "${WORK_DIR}/bcrs_spectral_only_visdrone_yolov5m/weights/best.pt" \
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
  "[B2] E2.1u — Gated Dual-Evidence (UAVDT)" \
  "configs/experiments/bcrs_dual_evidence_uavdt.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_uavdt_yolov5m/weights/best.pt" \
  "${UAVDT_LABELS}"

run_experiment \
  "[B3] E2.3u — Concat Dual-Evidence (UAVDT)" \
  "configs/experiments/bcrs_dual_evidence_concat_uavdt.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_concat_uavdt_yolov5m/weights/best.pt" \
  "${UAVDT_LABELS}"

run_experiment \
  "[B4] E2.6u — Channel-Pooled Spectral (UAVDT)" \
  "configs/experiments/bcrs_channel_pooled_spectral_uavdt.yaml" \
  "${WORK_DIR}/bcrs_channel_pooled_spectral_uavdt_yolov5m/weights/best.pt" \
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
echo "   work_dirs/bcrs_spectral_only_visdrone_yolov5m/weights/best.pt"
echo "   work_dirs/esod_uavdt_yolov5m/weights/best.pt"
echo "   work_dirs/bcrs_dual_evidence_uavdt_yolov5m/weights/best.pt"
echo "   work_dirs/bcrs_dual_evidence_concat_uavdt_yolov5m/weights/best.pt"
echo "   work_dirs/bcrs_channel_pooled_spectral_uavdt_yolov5m/weights/best.pt"
echo ""
echo " Next steps:"
echo "   1. Run inference sweep: bash tools/inference_sweep.sh"
echo "      (add UAVDT entries to inference_sweep.sh when results are verified)"
echo "   2. Sync results to local: SERVER=xxx bash tools/sync_results.sh"
echo "============================================================"

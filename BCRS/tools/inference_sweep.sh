#!/bin/bash
# =============================================================================
# BCRS Complete Inference Sweep — All Trained Models
# Runs bcrs test (K=64) + audit for every trained model on VisDrone val set.
# Usage: cd /root/BCRS/BCRS && bash tools/inference_sweep.sh
# =============================================================================

set -e

WORK_DIR="/root/BCRS/BCRS/work_dirs"
VISDRONE_LABELS="/root/autodl-tmp/VisDrone/labels/val"
AUDIT_TOOL="python tools/audit_failure_cases.py"

# ---------------------------------------------------------------------------
# Step 0: Clean all existing test result directories
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Step 0: Cleaning old test result directories..."
echo "============================================================"
rm -rf \
  "${WORK_DIR}/esod_visdrone_yolov5m_test" \
  "${WORK_DIR}/bcrs_dual_evidence_visdrone_yolov5m_test" \
  "${WORK_DIR}/bcrs_dual_evidence_concat_visdrone_yolov5m_test" \
  "${WORK_DIR}/bcrs_dual_evidence_visdrone_spectral_yolov5m_test" \
  "${WORK_DIR}/bcrs_channel_pooled_spectral_visdrone_yolov5m_test" \
  "${WORK_DIR}/bcrs_spectral_only_visdrone_yolov5m_test"
echo "Done."

# ---------------------------------------------------------------------------
# Helper: run test + audit for one model
# ---------------------------------------------------------------------------
run_model() {
  local CONFIG="$1"
  local CKPT="$2"
  local TEST_DIR="${WORK_DIR}/$3"
  local LABEL="$4"

  echo ""
  echo "============================================================"
  echo " Testing: ${LABEL}"
  echo "  config : ${CONFIG}"
  echo "  ckpt   : ${CKPT}"
  echo "============================================================"

  bcrs test "${CONFIG}" \
    --set "test.checkpoint=${CKPT}" \
    --set test.save_json=true \
    --set test.patch_budget=64

  echo ""
  echo "--- Audit: ${LABEL} ---"
  ${AUDIT_TOOL} \
    "${TEST_DIR}/best_predictions.json" \
    "${VISDRONE_LABELS}"

  echo ""
  echo ">>> DONE: ${LABEL}"
  echo "============================================================"
}

# ---------------------------------------------------------------------------
# Step 1: ESOD Baseline (E1.0)
# ---------------------------------------------------------------------------
run_model \
  "configs/experiments/esod_visdrone.yaml" \
  "${WORK_DIR}/esod_visdrone_yolov5m/weights/best.pt" \
  "esod_visdrone_yolov5m_test" \
  "E1.0 — ESOD Baseline"

# ---------------------------------------------------------------------------
# Step 2: BCRS Dual Evidence — Original Gated (E2.1)
# ---------------------------------------------------------------------------
run_model \
  "configs/experiments/bcrs_dual_evidence_visdrone.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_visdrone_yolov5m/weights/best.pt" \
  "bcrs_dual_evidence_visdrone_yolov5m_test" \
  "E2.1 — BCRS Dual Evidence (Gated)"

# ---------------------------------------------------------------------------
# Step 3: BCRS Dual Evidence Concat (E2.3)
# ---------------------------------------------------------------------------
run_model \
  "configs/experiments/bcrs_dual_evidence_concat_visdrone.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_concat_visdrone_yolov5m/weights/best.pt" \
  "bcrs_dual_evidence_concat_visdrone_yolov5m_test" \
  "E2.3 — BCRS Dual Evidence Concat"

# ---------------------------------------------------------------------------
# Step 4: BCRS Dual Evidence Spectral (E2.4)
# ---------------------------------------------------------------------------
run_model \
  "configs/experiments/bcrs_dual_evidence_visdrone_spectral.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_visdrone_spectral_yolov5m/weights/best.pt" \
  "bcrs_dual_evidence_visdrone_spectral_yolov5m_test" \
  "E2.4 — BCRS Dual Evidence Spectral"

# ---------------------------------------------------------------------------
# Step 5: BCRS Channel-Pooled Spectral (E2.6)
# ---------------------------------------------------------------------------
run_model \
  "configs/experiments/bcrs_channel_pooled_spectral_visdrone.yaml" \
  "${WORK_DIR}/bcrs_channel_pooled_spectral_visdrone_yolov5m/weights/best.pt" \
  "bcrs_channel_pooled_spectral_visdrone_yolov5m_test" \
  "E2.6 — BCRS Channel-Pooled Spectral"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " ALL INFERENCE SWEEP COMPLETE"
echo " JSON results in work_dirs/*_test/best_predictions.json"
echo "============================================================"

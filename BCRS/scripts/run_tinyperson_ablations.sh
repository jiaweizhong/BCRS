#!/bin/bash
# =============================================================================
# BCRS TinyPerson Cross-Dataset Ablation Script
#
# Trains the 5 core BCRS ablation experiments on TinyPerson sequentially:
#
#   [T1] E1.0t esod_tinyperson_yolov5m                      — ESOD Baseline
#   [T2] E2.1t bcrs_semantic_only_tinyperson_yolov5m         — Semantic-Only (Coverage-Supervised)
#   [T3] E2.3t bcrs_dual_evidence_concat_tinyperson_yolov5m  — Concat Dual-Evidence
#   [T4] E2.6t bcrs_channel_pooled_spectral_tinyperson_yolov5m — Channel-Pooled Spectral (Gated)
#   [T5] E2.9t bcrs_channel_pooled_concat_tinyperson_yolov5m — Channel-Pooled Concat (SOTA Winner)
#
# Each experiment: train (50 epochs) → test (K=64) → audit failure cases
#
# Usage: cd /root/BCRS/BCRS && bash scripts/run_tinyperson_ablations.sh [TINYPERSON_ROOT]
#        (recommend: nohup bash scripts/run_tinyperson_ablations.sh /root/autodl-tmp/TinyPerson > tinyperson_overnight.log 2>&1 &)
#
#   TINYPERSON_ROOT resolution order:
#     1. first positional argument, e.g.:
#          bash scripts/run_tinyperson_ablations.sh /root/autodl-tmp/TinyPerson
#     2. an already-exported TINYPERSON_ROOT env var, e.g.:
#          export TINYPERSON_ROOT=/root/autodl-tmp/TinyPerson
#          bash scripts/run_tinyperson_ablations.sh
# =============================================================================

set -e

TINYPERSON_ROOT="${1:-${TINYPERSON_ROOT:-/root/autodl-tmp/TinyPerson}}"
if [ ! -d "${TINYPERSON_ROOT}" ]; then
  echo "ERROR: TINYPERSON_ROOT directory not found at ${TINYPERSON_ROOT}" >&2
  echo "  Usage: bash scripts/run_tinyperson_ablations.sh <TINYPERSON_ROOT>" >&2
  echo "     or: export TINYPERSON_ROOT=/path/to/TinyPerson && bash scripts/run_tinyperson_ablations.sh" >&2
  exit 1
fi
export TINYPERSON_ROOT

WORK_DIR="/root/BCRS/BCRS/work_dirs"
TINYPERSON_LABELS="${TINYPERSON_ROOT}/labels/val"
if [ ! -d "${TINYPERSON_LABELS}" ]; then
  TINYPERSON_LABELS="${TINYPERSON_ROOT}/labels/test"
fi
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

  local TEST_DIR
  TEST_DIR="$(dirname "${CKPT}")_test"
  local PREDS="${TEST_DIR}/best_predictions.json"

  if [ -f "${PREDS}" ] && [ -d "${LABELS_DIR}" ]; then
    echo "[$(date '+%H:%M:%S')] Auditing failure cases for: ${LABEL}"
    ${AUDIT_TOOL} "${PREDS}" "${LABELS_DIR}" || true
  else
    echo "[$(date '+%H:%M:%S')] Skipping audit: predictions (${PREDS}) or labels (${LABELS_DIR}) missing."
  fi
}

echo "============================================================"
echo "  Starting BCRS TinyPerson Core 5-Experiment Suite"
echo "  TINYPERSON_ROOT=${TINYPERSON_ROOT}"
echo "============================================================"

# [T1] E1.0t ESOD Baseline on TinyPerson
run_experiment \
  "[T1] E1.0t  esod_tinyperson" \
  "configs/experiments/esod_tinyperson.yaml" \
  "${WORK_DIR}/esod_tinyperson_yolov5m/weights/best.pt" \
  "${TINYPERSON_LABELS}"

# [T2] E2.1t BCRS Semantic-Only (Coverage-Supervised) on TinyPerson
run_experiment \
  "[T2] E2.1t  bcrs_semantic_only_tinyperson" \
  "configs/experiments/bcrs_semantic_only_tinyperson.yaml" \
  "${WORK_DIR}/bcrs_semantic_only_tinyperson_yolov5m/weights/best.pt" \
  "${TINYPERSON_LABELS}"

# [T3] E2.3t BCRS Concat Dual-Evidence on TinyPerson
run_experiment \
  "[T3] E2.3t  bcrs_dual_evidence_concat_tinyperson" \
  "configs/experiments/bcrs_dual_evidence_concat_tinyperson.yaml" \
  "${WORK_DIR}/bcrs_dual_evidence_concat_tinyperson_yolov5m/weights/best.pt" \
  "${TINYPERSON_LABELS}"

# [T4] E2.6t BCRS Channel-Pooled Spectral (Gated) on TinyPerson
run_experiment \
  "[T4] E2.6t  bcrs_channel_pooled_spectral_tinyperson" \
  "configs/experiments/bcrs_channel_pooled_spectral_tinyperson.yaml" \
  "${WORK_DIR}/bcrs_channel_pooled_spectral_tinyperson_yolov5m/weights/best.pt" \
  "${TINYPERSON_LABELS}"

# [T5] E2.9t BCRS Channel-Pooled Concat (SOTA Winner) on TinyPerson
run_experiment \
  "[T5] E2.9t  bcrs_channel_pooled_concat_tinyperson" \
  "configs/experiments/bcrs_channel_pooled_concat_tinyperson.yaml" \
  "${WORK_DIR}/bcrs_channel_pooled_concat_tinyperson_yolov5m/weights/best.pt" \
  "${TINYPERSON_LABELS}"

echo ""
echo "============================================================"
echo "  [$(date '+%H:%M:%S')] TinyPerson Core 5-Experiment Suite Complete!"
echo "============================================================"

#!/bin/bash
# =============================================================================
# BCRS Full Inference Sweep — All Models × All Budgets
#
# Replaces: inference_sweep.sh (K=64 only)
#           budget_curve_sweep.sh (K=16,32,48,64 for E1.0/E2.1/E2.3)
#
# Runs: E1.0 / E2.1 / E2.3 / E2.4 / E2.6  x  K={16, 32, 48, 64}
# Total: 5 models × 4 budgets = 20 inference runs
#
# Output naming convention: work_dirs/{stem}_k{K}/
#   e.g.  esod_visdrone_yolov5m_k64
#         bcrs_dual_evidence_visdrone_yolov5m_k16
#
# Summary: work_dirs/sweep_results.json  (structured, ready for plotting)
#
# Usage: cd /root/BCRS/BCRS && bash tools/inference_full_sweep.sh
# =============================================================================

set -e

WORK_DIR="/root/BCRS/BCRS/work_dirs"
VISDRONE_LABELS="/root/autodl-tmp/VisDrone/labels/val"
AUDIT_TOOL="python tools/audit_failure_cases.py"
SUMMARY_JSON="${WORK_DIR}/sweep_results.json"
K_VALUES=(16 32 48 64)

# Model registry: (EXP_ID, DISPLAY_NAME, STEM, CONFIG)
declare -a MODELS=(
  "E1.0|ESOD Baseline|esod_visdrone_yolov5m|configs/experiments/esod_visdrone.yaml"
  "E2.1|BCRS Dual Evidence Gated|bcrs_dual_evidence_visdrone_yolov5m|configs/experiments/bcrs_dual_evidence_visdrone.yaml"
  "E2.3|BCRS Dual Evidence Concat|bcrs_dual_evidence_concat_visdrone_yolov5m|configs/experiments/bcrs_dual_evidence_concat_visdrone.yaml"
  "E2.4|BCRS Dual Evidence Spectral|bcrs_dual_evidence_visdrone_spectral_yolov5m|configs/experiments/bcrs_dual_evidence_visdrone_spectral.yaml"
  "E2.6|BCRS Channel-Pooled Spectral|bcrs_channel_pooled_spectral_visdrone_yolov5m|configs/experiments/bcrs_channel_pooled_spectral_visdrone.yaml"
)

# ===========================================================================
# Step 0: Wipe ALL previous test/sweep directories in work_dirs
# ===========================================================================
echo "============================================================"
echo " Step 0: Wiping all previous test/sweep directories..."
echo "============================================================"

# Remove old _test naming (inference_sweep.sh legacy)
find "${WORK_DIR}" -maxdepth 1 -type d -name "*_test" -exec rm -rf {} + 2>/dev/null || true
# Remove new _kNN naming (budget_curve_sweep.sh)
for K in "${K_VALUES[@]}"; do
  find "${WORK_DIR}" -maxdepth 1 -type d -name "*_k${K}" -exec rm -rf {} + 2>/dev/null || true
done
# Remove old summary JSON if present
rm -f "${WORK_DIR}/sweep_results.json" "${WORK_DIR}/../budget_curve_results.json"

echo "  Done. work_dirs is clean."

# ===========================================================================
# JSON summary: open array
# ===========================================================================
mkdir -p "${WORK_DIR}"
printf '{\n  "sweep": "BCRS Full Inference Sweep",\n  "date": "%s",\n  "k_values": [16,32,48,64],\n  "models": ["E1.0","E2.1","E2.3","E2.4","E2.6"],\n  "results": [\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SUMMARY_JSON}"
FIRST_JSON_ENTRY=true

# ===========================================================================
# Helper: parse a float from grep output
# ===========================================================================
parse_audit_field() {
  # $1: full audit stdout, $2: grep pattern, $3: awk field index (1-based |-separated)
  echo "$1" | grep "$2" | head -1 | awk -F'|' "{gsub(/[ %]/,\"\",\$$3); print \$$3}"
}

# ===========================================================================
# Helper: run one (model, K) pair
# ===========================================================================
run_one() {
  local EXP_ID="$1"
  local DISPLAY="$2"
  local STEM="$3"
  local CONFIG="$4"
  local K="$5"

  local CKPT="${WORK_DIR}/${STEM}/weights/best.pt"
  local TEST_NAME="${STEM}_k${K}"
  local TEST_DIR="${WORK_DIR}/${TEST_NAME}"

  echo ""
  echo "------------------------------------------------------------"
  echo "  [${EXP_ID}] ${DISPLAY}  @  K=${K}"
  echo "  dir: ${TEST_NAME}"
  echo "------------------------------------------------------------"

  if [ ! -f "${CKPT}" ]; then
    echo "  WARNING: checkpoint not found at ${CKPT} — skipping."
    return
  fi

  # Run inference (output goes to both stdout and a run log)
  local LOG="${TEST_DIR}/run.log"
  mkdir -p "${TEST_DIR}"

  bcrs test "${CONFIG}" \
    --set "test.checkpoint=${CKPT}" \
    --set test.save_json=true \
    --set "test.patch_budget=${K}" \
    --set "test.project=${WORK_DIR}" \
    --set "test.name=${TEST_NAME}" \
    --set test.exist_ok=true 2>&1 | tee "${LOG}"

  # Parse ESOD diagnostic line from log
  local ESOD_LINE
  ESOD_LINE=$(grep "\[ESOD Validation Diagnostic\]" "${LOG}" | tail -1)
  local MAP50
  MAP50=$(echo "${ESOD_LINE}"  | grep -oP 'mAP@0\.5: \K[0-9.]+' || echo "0")
  local BPR
  BPR=$(echo "${ESOD_LINE}"    | grep -oP 'Patch BPR \(bpr\): \K[0-9.]+' || echo "0")
  local MP
  MP=$(echo "${ESOD_LINE}"     | grep -oP 'BBox Precision \(mp\): \K[0-9.]+' || echo "0")
  local MR
  MR=$(echo "${ESOD_LINE}"     | grep -oP 'BBox Recall \(mr\): \K[0-9.]+' || echo "0")

  # Parse PyCOCO metrics from log
  local COCO_AP
  COCO_AP=$(grep  "IoU=0.50:0.95 | area=   all | maxDets=100" "${LOG}" | grep -oP '= \K[0-9.]+' || echo "0")
  local COCO_AP50
  COCO_AP50=$(grep "IoU=0.50      | area=   all | maxDets=500" "${LOG}" | grep -oP '= \K[0-9.]+' || echo "0")
  local COCO_AR500
  COCO_AR500=$(grep "IoU=0.50:0.95 | area=   all | maxDets=500" "${LOG}" | grep -oP '= \K[0-9.]+' || echo "0")

  # Run audit tool and capture output
  echo ""
  echo "--- Audit: [${EXP_ID}] @ K=${K} ---"
  local AUDIT_OUT
  AUDIT_OUT=$(${AUDIT_TOOL} "${TEST_DIR}/best_predictions.json" "${VISDRONE_LABELS}" 2>&1)
  echo "${AUDIT_OUT}"

  # Parse audit fields
  local VT_RECALLED VT_RATE TINY_RECALLED TINY_RATE SMALL_RECALLED SMALL_RATE ML_RECALLED ML_RATE TOTAL_GT TOTAL_RECALLED TOTAL_RATE
  VT_RATE=$(parse_audit_field "${AUDIT_OUT}" "Very Tiny (<16x16)" 4)
  VT_RECALLED=$(parse_audit_field "${AUDIT_OUT}" "Very Tiny (<16x16)" 3)
  TINY_RATE=$(parse_audit_field "${AUDIT_OUT}" "^Tiny (16x16" 4)
  TINY_RECALLED=$(parse_audit_field "${AUDIT_OUT}" "^Tiny (16x16" 3)
  SMALL_RATE=$(parse_audit_field "${AUDIT_OUT}" "^Small (32x32" 4)
  SMALL_RECALLED=$(parse_audit_field "${AUDIT_OUT}" "^Small (32x32" 3)
  ML_RATE=$(parse_audit_field "${AUDIT_OUT}" "^Medium/Large" 4)
  ML_RECALLED=$(parse_audit_field "${AUDIT_OUT}" "^Medium/Large" 3)
  TOTAL_GT=$(parse_audit_field "${AUDIT_OUT}" "TOTAL GT TARGETS" 2)
  TOTAL_RECALLED=$(parse_audit_field "${AUDIT_OUT}" "TOTAL GT TARGETS" 3)
  TOTAL_RATE=$(parse_audit_field "${AUDIT_OUT}" "TOTAL GT TARGETS" 4)

  # Append JSON entry
  if [ "${FIRST_JSON_ENTRY}" = true ]; then
    FIRST_JSON_ENTRY=false
  else
    printf ',' >> "${SUMMARY_JSON}"
  fi

  printf '\n    {\n      "exp_id": "%s", "model": "%s", "stem": "%s", "K": %s,\n      "dir": "%s",\n      "esod": {"map50": %s, "bpr": %s, "mp": %s, "mr": %s},\n      "coco": {"ap": %s, "ap50": %s, "ar500": %s},\n      "audit": {\n        "total_gt": %s, "total_recalled": %s, "total_recall_pct": %s,\n        "very_tiny":    {"recalled": %s, "pct": %s},\n        "tiny":         {"recalled": %s, "pct": %s},\n        "small":        {"recalled": %s, "pct": %s},\n        "medium_large": {"recalled": %s, "pct": %s}\n      }\n    }' \
    "${EXP_ID}" "${DISPLAY}" "${STEM}" "${K}" \
    "${TEST_NAME}" \
    "${MAP50:-0}" "${BPR:-0}" "${MP:-0}" "${MR:-0}" \
    "${COCO_AP:-0}" "${COCO_AP50:-0}" "${COCO_AR500:-0}" \
    "${TOTAL_GT:-0}" "${TOTAL_RECALLED:-0}" "${TOTAL_RATE:-0}" \
    "${VT_RECALLED:-0}" "${VT_RATE:-0}" \
    "${TINY_RECALLED:-0}" "${TINY_RATE:-0}" \
    "${SMALL_RECALLED:-0}" "${SMALL_RATE:-0}" \
    "${ML_RECALLED:-0}" "${ML_RATE:-0}" \
    >> "${SUMMARY_JSON}"

  echo ""
  echo "  >>> DONE: [${EXP_ID}] @ K=${K}"
}

# ===========================================================================
# Main sweep loop: K-outer, model-inner (easy budget comparison)
# ===========================================================================
for K in "${K_VALUES[@]}"; do
  echo ""
  echo "============================================================"
  echo "  BUDGET K = ${K}  ($(( K * 100 / 64 ))% of full patch grid)"
  echo "============================================================"

  for MODEL_ENTRY in "${MODELS[@]}"; do
    IFS='|' read -r EXP_ID DISPLAY STEM CONFIG <<< "${MODEL_ENTRY}"
    run_one "${EXP_ID}" "${DISPLAY}" "${STEM}" "${CONFIG}" "${K}"
  done
done

# ===========================================================================
# Close JSON
# ===========================================================================
printf '\n  ]\n}\n' >> "${SUMMARY_JSON}"

# ===========================================================================
# Final quick-reference table (Python inline)
# ===========================================================================
echo ""
echo "============================================================"
echo " ALL INFERENCE COMPLETE"
echo " Results  : ${WORK_DIR}/<stem>_k<K>/"
echo " Summary  : ${SUMMARY_JSON}"
echo "============================================================"
echo ""

python3 - "${SUMMARY_JSON}" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

results = data["results"]
k_values = data["k_values"]
exp_ids  = data["models"]

# --- Budget curve: Very Tiny Recall by (model, K) ---
print("\n Budget Curve — Very Tiny Recall (<16×16 px)")
print(f" {'Exp':<6} | {'Model':<34} | " + " | ".join(f"K={k:>2}" for k in k_values))
print(" " + "-"*6 + "-+-" + "-"*34 + "-+-" + "-+-".join(["-"*6]*len(k_values)))
for exp_id in exp_ids:
    rows = {r["K"]: r for r in results if r["exp_id"] == exp_id}
    if not rows:
        continue
    display = next(iter(rows.values()))["model"]
    cells = []
    for k in k_values:
        pct = rows[k]["audit"]["very_tiny"]["pct"] if k in rows else "—"
        cells.append(f"{float(pct):>5.1f}%" if pct != "—" else "  —   ")
    print(f" {exp_id:<6} | {display:<34} | " + " | ".join(cells))

# --- Budget curve: Total Recall by (model, K) ---
print("\n Budget Curve — Total GT Recall")
print(f" {'Exp':<6} | {'Model':<34} | " + " | ".join(f"K={k:>2}" for k in k_values))
print(" " + "-"*6 + "-+-" + "-"*34 + "-+-" + "-+-".join(["-"*6]*len(k_values)))
for exp_id in exp_ids:
    rows = {r["K"]: r for r in results if r["exp_id"] == exp_id}
    if not rows:
        continue
    display = next(iter(rows.values()))["model"]
    cells = []
    for k in k_values:
        pct = rows[k]["audit"]["total_recall_pct"] if k in rows else "—"
        cells.append(f"{float(pct):>5.1f}%" if pct != "—" else "  —   ")
    print(f" {exp_id:<6} | {display:<34} | " + " | ".join(cells))

# --- Full detail at K=64 (primary claim) ---
print("\n Primary Claim Table — K=64 (Full Budget)")
print(f" {'Exp':<6} | {'mAP@0.5':>8} | {'BPR':>6} | {'COCO AP50':>9} | {'COCO AP':>7} | {'AR@500':>6} | {'VTiny%':>7} | {'Tiny%':>6} | {'Total%':>7}")
print(" " + "-"*100)
for r in [r for r in results if r["K"] == 64]:
    print(f" {r['exp_id']:<6} | {float(r['esod']['map50']):>8.3f} | {float(r['esod']['bpr']):>6.3f} | {float(r['coco']['ap50']):>9.3f} | {float(r['coco']['ap']):>7.3f} | {float(r['coco']['ar500']):>6.3f} | {float(r['audit']['very_tiny']['pct']):>6.1f}% | {float(r['audit']['tiny']['pct']):>5.1f}% | {float(r['audit']['total_recall_pct']):>6.1f}%")

PYEOF

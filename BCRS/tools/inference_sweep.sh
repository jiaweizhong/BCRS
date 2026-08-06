#!/bin/bash
# =============================================================================
# BCRS Full Inference Sweep — All Models × All Budgets
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
# Usage: cd /root/BCRS/BCRS && bash tools/inference_sweep.sh
# =============================================================================

set -eE
# Trap any unexpected errors to print location but continue the sweep
trap 'echo "  [TRAP] Command failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

WORK_DIR="/root/BCRS/BCRS/work_dirs"
VISDRONE_LABELS="/root/autodl-tmp/VisDrone/labels/val"
AUDIT_TOOL="python tools/audit_failure_cases.py"
SUMMARY_JSON="${WORK_DIR}/sweep_results.json"
K_VALUES=(16 32 48 64)
SWEEP_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Model registry: EXP_ID|DISPLAY|STEM|CONFIG
#
# NOTE (2026-08-06): E2.1 and E2.4 display names were corrected after auditing the
# actual model yaml / Segmenter class each config instantiates (see BCRS-Experiment-Plan.md
# "Efficiency" section). E2.1's model yaml (visdrone_yolov5m.yaml) uses the plain `Segmenter`
# class — identical to the E1.0 baseline, with NO spectral_branches/gated_fusions submodules —
# so it is a semantic-only, coverage-loss-supervised selector, not a dual-evidence fusion.
# E2.4's model yaml (visdrone_yolov5m_spectral.yaml) uses `DualEvidenceSegmenter`, which
# instantiates `GatedEvidenceFusion` (sigmoid gate, zero-sum semantic/spectral mix) — this is
# the actual "gated dual-evidence" architecture, previously mislabeled as "Spectral".
MODELS=(
  "E1.0|ESOD Baseline|esod_visdrone_yolov5m|configs/experiments/esod_visdrone.yaml"
  "E2.1|BCRS Semantic-Only (Coverage-Supervised)|bcrs_dual_evidence_visdrone_yolov5m|configs/experiments/bcrs_dual_evidence_visdrone.yaml"
  "E2.3|BCRS Dual Evidence Concat|bcrs_dual_evidence_concat_visdrone_yolov5m|configs/experiments/bcrs_dual_evidence_concat_visdrone.yaml"
  "E2.4|BCRS Dual Evidence Gated|bcrs_dual_evidence_visdrone_spectral_yolov5m|configs/experiments/bcrs_dual_evidence_visdrone_spectral.yaml"
  "E2.6|BCRS Channel-Pooled Spectral (Gated)|bcrs_channel_pooled_spectral_visdrone_yolov5m|configs/experiments/bcrs_channel_pooled_spectral_visdrone.yaml"
)

# ===========================================================================
# Step 0: Wipe ALL previous test/sweep directories in work_dirs
# ===========================================================================
echo "============================================================"
echo " Step 0: Wiping all previous test/sweep directories..."
echo "============================================================"
find "${WORK_DIR}" -maxdepth 1 -type d -name "*_test"  -exec rm -rf {} + 2>/dev/null || true
for K in "${K_VALUES[@]}"; do
  find "${WORK_DIR}" -maxdepth 1 -type d -name "*_k${K}" -exec rm -rf {} + 2>/dev/null || true
done
rm -f "${SUMMARY_JSON}"
echo "  Done. work_dirs is clean."

# ===========================================================================
# run_one: run inference for one (model, K) pair
#
# NOTE on output directory:
#   bcrs test ignores --set test.name and --set test.project; it always
#   outputs to work_dirs/{config.name}_test.  We let it write there, then
#   mv to our canonical {stem}_k{K}/ naming.
# ===========================================================================
run_one() {
  local EXP_ID="$1"
  local DISPLAY="$2"
  local STEM="$3"
  local CONFIG="$4"
  local K="$5"

  local CKPT="${WORK_DIR}/${STEM}/weights/best.pt"
  local DEFAULT_DIR="${WORK_DIR}/${STEM}_test"   # where bcrs test always writes
  local TARGET_DIR="${WORK_DIR}/${STEM}_k${K}"   # our canonical name
  local LOG="${TARGET_DIR}/run.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  [${EXP_ID}] ${DISPLAY}  @  K=${K}"
  echo "  target: ${STEM}_k${K}"
  echo "------------------------------------------------------------"

  if [ ! -f "${CKPT}" ]; then
    echo "  WARNING: checkpoint not found at ${CKPT} — skipping."
    return
  fi

  # Remove any stale default test dir before this run
  rm -rf "${DEFAULT_DIR}"

  # Run inference; bcrs test writes to {STEM}_test
  bcrs test "${CONFIG}" \
    --set "test.checkpoint=${CKPT}" \
    --set test.save_json=true \
    --set "test.patch_budget=${K}" \
    2>&1 | tee /tmp/bcrs_run.log

  # Rename default dir -> canonical k{K} dir
  if [ -d "${DEFAULT_DIR}" ]; then
    mv "${DEFAULT_DIR}" "${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"
    cp /tmp/bcrs_run.log "${LOG}"
  else
    echo "  WARNING: expected output dir ${DEFAULT_DIR} not found after bcrs test."
    echo "  Scanning for any new _test dir under work_dirs..."
    ls "${WORK_DIR}/" | grep "${STEM}" || true
    echo "  Skipping audit for [${EXP_ID}] @ K=${K} — results will show zeros in summary."
    return 0   # soft failure: log it, but continue the sweep
  fi

  # Run audit
  local PRED="${TARGET_DIR}/best_predictions.json"
  echo ""
  echo "--- Audit: [${EXP_ID}] @ K=${K} ---"
  if [ -f "${PRED}" ]; then
    ${AUDIT_TOOL} "${PRED}" "${VISDRONE_LABELS}" 2>&1 | tee -a "${LOG}"
  else
    echo "  WARNING: predictions not found at ${PRED} — audit skipped."
  fi

  echo ""
  echo "  >>> DONE: [${EXP_ID}] @ K=${K}"
}

# ===========================================================================
# Main sweep loop: K-outer, model-inner
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
# Build sweep_results.json from run logs using Python
# (Python handles all JSON serialization — no bash string formatting)
# ===========================================================================
echo ""
echo "============================================================"
echo " Building sweep_results.json from run logs..."
echo "============================================================"

python3 - "${WORK_DIR}" "${SUMMARY_JSON}" "${SWEEP_START}" <<'PYEOF'
import sys, json, re, os
from pathlib import Path

work_dir   = Path(sys.argv[1])
out_json   = sys.argv[2]
sweep_date = sys.argv[3]

MODELS = [
    ("E1.0", "ESOD Baseline",                          "esod_visdrone_yolov5m"),
    ("E2.1", "BCRS Semantic-Only (Coverage-Supervised)", "bcrs_dual_evidence_visdrone_yolov5m"),
    ("E2.3", "BCRS Dual Evidence Concat",               "bcrs_dual_evidence_concat_visdrone_yolov5m"),
    ("E2.4", "BCRS Dual Evidence Gated",                "bcrs_dual_evidence_visdrone_spectral_yolov5m"),
    ("E2.6", "BCRS Channel-Pooled Spectral (Gated)",    "bcrs_channel_pooled_spectral_visdrone_yolov5m"),
]
K_VALUES = [16, 32, 48, 64]

def parse_float(text, pattern, default=0.0):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

def parse_audit(log_text):
    """Parse size-bin recall table from audit output."""
    bins = {}
    patterns = {
        "very_tiny":    r"Very Tiny \(<16x16\)\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*([\d.]+)",
        "tiny":         r"Tiny \(16x16 - 32x32\)\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*([\d.]+)",
        "small":        r"Small \(32x32 - 96x96\)\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*([\d.]+)",
        "medium_large": r"Medium/Large \(>96x96\)\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, log_text)
        bins[key] = {
            "recalled": int(m.group(1)) if m else 0,
            "pct":      float(m.group(2)) if m else 0.0,
        }
    m = re.search(r"TOTAL GT TARGETS\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)", log_text)
    return {
        "total_gt":        int(m.group(1)) if m else 0,
        "total_recalled":  int(m.group(2)) if m else 0,
        "total_recall_pct": float(m.group(3)) if m else 0.0,
        **bins,
    }

results = []
for exp_id, display, stem in MODELS:
    for K in K_VALUES:
        log_path = work_dir / f"{stem}_k{K}" / "run.log"
        if not log_path.exists():
            print(f"  MISSING log: {stem}_k{K}/run.log — entry will have zeros")
            log_text = ""
        else:
            log_text = log_path.read_text(errors="replace")

        esod_line = next((l for l in log_text.splitlines() if "[ESOD Validation Diagnostic]" in l), "")
        entry = {
            "exp_id":  exp_id,
            "model":   display,
            "stem":    stem,
            "K":       K,
            "dir":     f"{stem}_k{K}",
            "esod": {
                "map50": parse_float(esod_line, r"mAP@0\.5: ([\d.]+)"),
                "bpr":   parse_float(esod_line, r"Patch BPR \(bpr\): ([\d.]+)"),
                "mp":    parse_float(esod_line, r"BBox Precision \(mp\): ([\d.]+)"),
                "mr":    parse_float(esod_line, r"BBox Recall \(mr\): ([\d.]+)"),
            },
            "coco": {
                "ap":    parse_float(log_text, r"IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = ([\d.]+)"),
                "ap50":  parse_float(log_text, r"IoU=0\.50      \| area=   all \| maxDets=500 \] = ([\d.]+)"),
                "ar500": parse_float(log_text, r"IoU=0\.50:0\.95 \| area=   all \| maxDets=500 \] = ([\d.]+)"),
            },
            "audit": parse_audit(log_text),
        }
        results.append(entry)
        status = "OK" if log_text else "MISSING"
        print(f"  [{status}] {exp_id} K={K:2d}  map50={entry['esod']['map50']:.3f}  VTiny={entry['audit']['very_tiny']['pct']:.1f}%  Total={entry['audit']['total_recall_pct']:.1f}%")

summary = {
    "sweep":    "BCRS Full Inference Sweep",
    "date":     sweep_date,
    "k_values": K_VALUES,
    "models":   [m[0] for m in MODELS],
    "results":  results,
}
with open(out_json, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Saved: {out_json}")
PYEOF

# ===========================================================================
# Final quick-reference tables
# ===========================================================================
echo ""
python3 - "${SUMMARY_JSON}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)

results  = data["results"]
k_values = data["k_values"]
exp_ids  = data["models"]

def get(results, exp_id, K):
    for r in results:
        if r["exp_id"] == exp_id and r["K"] == K:
            return r
    return None

print("\n Budget Curve — Very Tiny Recall (<16×16 px)")
hdr = f" {'Exp':<6} | {'Model':<34} | " + " | ".join(f"K={k:>2}" for k in k_values)
print(hdr)
print(" " + "-" * len(hdr))
for exp_id in exp_ids:
    r0 = get(results, exp_id, k_values[0])
    if not r0: continue
    cells = []
    for k in k_values:
        r = get(results, exp_id, k)
        pct = r["audit"]["very_tiny"]["pct"] if r else None
        cells.append(f"{pct:>5.1f}%" if pct is not None else "  —   ")
    print(f" {exp_id:<6} | {r0['model']:<34} | " + " | ".join(cells))

print("\n Budget Curve — Total GT Recall")
print(hdr)
print(" " + "-" * len(hdr))
for exp_id in exp_ids:
    r0 = get(results, exp_id, k_values[0])
    if not r0: continue
    cells = []
    for k in k_values:
        r = get(results, exp_id, k)
        pct = r["audit"]["total_recall_pct"] if r else None
        cells.append(f"{pct:>5.1f}%" if pct is not None else "  —   ")
    print(f" {exp_id:<6} | {r0['model']:<34} | " + " | ".join(cells))

print("\n Primary Claim — K=64 Full Budget")
print(f" {'Exp':<6} | {'mAP@0.5':>8} | {'BPR':>5} | {'AP50':>5} | {'AP':>5} | {'AR500':>5} | {'VTiny%':>7} | {'Tiny%':>6} | {'Total%':>7}")
print(" " + "-" * 80)
for exp_id in exp_ids:
    r = get(results, exp_id, 64)
    if not r: continue
    print(f" {exp_id:<6} | {r['esod']['map50']:>8.3f} | {r['esod']['bpr']:>5.3f} | "
          f"{r['coco']['ap50']:>5.3f} | {r['coco']['ap']:>5.3f} | {r['coco']['ar500']:>5.3f} | "
          f"{r['audit']['very_tiny']['pct']:>6.1f}% | {r['audit']['tiny']['pct']:>5.1f}% | "
          f"{r['audit']['total_recall_pct']:>6.1f}%")
PYEOF

echo ""
echo "============================================================"
echo " ALL INFERENCE COMPLETE"
echo " Logs     : work_dirs/<stem>_k<K>/run.log"
echo " Summary  : ${SUMMARY_JSON}"
echo "============================================================"

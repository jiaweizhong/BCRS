#!/bin/bash
# BCRS Phase 2 Overnight Multi-Ablation Batch Execution Script

set -e

echo "=========================================================="
echo " Starting BCRS Phase 2 Overnight Multi-Ablation Benchmark "
echo "=========================================================="

# 1. Main Phase 2 Model: Gated Dual-Evidence Spectral Fusion
echo -e "\n[1/3] Running Main Phase 2: Dual-Evidence Gated Fusion..."
bcrs train configs/experiments/bcrs_dual_evidence_visdrone_spectral.yaml
bcrs test configs/experiments/bcrs_dual_evidence_visdrone_spectral.yaml \
  --set test.checkpoint=work_dirs/bcrs_dual_evidence_visdrone_spectral_yolov5m/weights/best.pt \
  --set test.save_json=true \
  --set test.patch_budget=16

# 2. Phase 2 Ablation: Spectral-Only Selector
echo -e "\n[2/3] Running Phase 2 Ablation: Spectral-Only Selector..."
bcrs train configs/experiments/bcrs_spectral_only_visdrone.yaml
bcrs test configs/experiments/bcrs_spectral_only_visdrone.yaml \
  --set test.checkpoint=work_dirs/bcrs_spectral_only_visdrone_yolov5m/weights/best.pt \
  --set test.save_json=true \
  --set test.patch_budget=16

# 3. Phase 2 Ablation: Concat Fusion Selector
echo -e "\n[3/3] Running Phase 2 Ablation: Concat Fusion Selector..."
bcrs train configs/experiments/bcrs_dual_evidence_concat_visdrone.yaml
bcrs test configs/experiments/bcrs_dual_evidence_concat_visdrone.yaml \
  --set test.checkpoint=work_dirs/bcrs_dual_evidence_concat_visdrone_yolov5m/weights/best.pt \
  --set test.save_json=true \
  --set test.patch_budget=16

echo -e "\n=========================================================="
echo " Overnight Ablation Benchmark Completed Successfully!     "
echo "=========================================================="

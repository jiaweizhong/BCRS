#!/usr/bin/env bash
# SeaPerson (aka TinyPersonV2) 6-arm roster, run through the ACTIVE
# hesod/backends/hesod tree, mirroring run_uavdt.sh's resume-capable
# run_arm() structure.
#
# Arms (HESOD-Experiment-Plan.md, decided alongside the TinyPerson roster):
#   R0             -- Segmenter, upstream loss, CIoU box (original baseline)
#   semantic-only  -- SAME architecture as R0, coverage loss instead (E2.1
#                      style) -- isolates the loss-function effect from the
#                      selector-architecture effect, distinct from R0.
#   spectral-only  -- SpectralOnlySegmenter, coverage loss (E2.5 style)
#   concat-only    -- ChannelPooledConcatEvidenceSegmenter, coverage loss,
#                      CIoU box (E2.9 style, no SABL)
#   concat+SABL    -- same concat architecture, coverage loss, SABL box
#   concat+SABL+ISPPHead -- same + ISPPHead Detect (this project's best-known
#                      lightweight recipe, confirmed on TinyPerson v1)
#
# Prerequisite (run once, not by this script):
#   python scripts/data_prepare.py --dataset /root/autodl-tmp/seaperson
#   python scripts/esod_baseline/reorganize_seaperson.py \
#     --raw-root /root/autodl-tmp/seaperson --out-root /root/autodl-tmp/seaperson_v2
#
# Usage:
#   SMOKE=1 bash run_seaperson.sh       # ~minutes, validate the pipeline first
#   nohup bash run_seaperson.sh > /root/seaperson.log 2>&1 &
#   disown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:-0}"
RUN_ROOT="${RUN_ROOT:-$HOME/esod_baseline_runs}"
BATCH="${BATCH:-8}"
IMG_SIZE="${IMG_SIZE:-2048}"
ESOD_REPO="$SCRIPT_DIR/../../hesod/backends/hesod"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/seaperson_v2}"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/seaperson.yaml}"
HYP="data/hyps/hyp.seaperson.yaml"
CLASSES="person"
VAL_SPLIT="test"

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  EPOCHS=1
  SUFFIX="_smoke"
  log_prefix="SMOKE"
else
  EPOCHS="${EPOCHS:-50}"
  SUFFIX=""
  log_prefix="REAL"
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$log_prefix] $*"; }

completed_epochs() {
  local results_file="$1/results.txt"
  if [ ! -f "$results_file" ]; then
    echo 0
    return
  fi
  awk 'NF { count++ } END { print count + 0 }' "$results_file"
}

log "epochs=$EPOCHS img-size=$IMG_SIZE batch=$BATCH data=$DATA_YAML"

# Optional: comma-separated list of exact run_name(s) to run, e.g.
#   ARMS="seaperson_yolov5m_channel_pooled_concat_sabl,seaperson_yolov5m_channel_pooled_concat_sabl_isphead" bash run_seaperson.sh
# Unset/empty (default) runs every arm, unchanged from before. Names must
# match exactly what run_arm() below is called with, including any ${SUFFIX}
# (e.g. under SMOKE=1). This only skips arms outright -- it does not replace
# run_arm()'s own per-arm resume/skip logic for whatever does run.
ARMS="${ARMS:-}"

run_arm() {
  local run_name="$1" model_cfg="$2" selector_loss="$3" box_loss="${4:-upstream}"

  if [ -n "$ARMS" ]; then
    case ",$ARMS," in
      *",$run_name,"*) ;;
      *) log "===== $run_name not in \$ARMS, skipping ====="; return 0 ;;
    esac
  fi

  local results_dir="$RUN_ROOT/test/$run_name"
  mkdir -p "$results_dir"
  cd "$ESOD_REPO"

  local train_dir="$RUN_ROOT/train/$run_name"
  local ckpt="$train_dir/weights/best.pt"
  local last_ckpt="$train_dir/weights/last.pt"
  local done_epochs
  done_epochs="$(completed_epochs "$train_dir")"
  local training_was_already_done=0

  if [ "$done_epochs" -ge "$EPOCHS" ] && [ -f "$ckpt" ]; then
    log "===== $run_name already completed $done_epochs/$EPOCHS epochs, skipping training ====="
    training_was_already_done=1
  elif [ -f "$last_ckpt" ]; then
    log "===== Resuming $run_name from $last_ckpt ($done_epochs/$EPOCHS epochs completed) ====="
    python train.py --resume "$last_ckpt" \
      2>&1 | tee -a "$results_dir/${run_name}_train.log"
  elif [ -f "$ckpt" ] || [ "$done_epochs" -gt 0 ]; then
    log "FATAL: $run_name is incomplete ($done_epochs/$EPOCHS epochs) but $last_ckpt is missing"
    exit 1
  else
    log "===== Training $run_name (selector_loss=$selector_loss, box_loss=$box_loss) ====="
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights weights/pretrained/yolov5m.pt \
      --hyp "$HYP" \
      --batch-size "$BATCH" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
      --selector-loss "$selector_loss" --box-loss "$box_loss" \
      --project "$RUN_ROOT/train" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_train.log"
  fi

  done_epochs="$(completed_epochs "$train_dir")"
  if [ "$done_epochs" -lt "$EPOCHS" ]; then
    log "FATAL: $run_name has only $done_epochs/$EPOCHS completed epochs after training/resume"
    exit 1
  fi

  if [ ! -f "$ckpt" ]; then
    log "FATAL: $run_name training finished but $ckpt does not exist -- aborting before eval"
    exit 1
  fi

  if [ "$training_was_already_done" = "1" ] \
    && [ -f "$results_dir/best_predictions.json" ] \
    && [ -f "$RUN_ROOT/measure/$run_name/buckets.json" ]; then
    log "===== $run_name eval + measure already complete, skipping ====="
  else
    log "Evaluating $run_name"
    python test.py \
      --data "$DATA_YAML" --weights "$ckpt" --task "$VAL_SPLIT" \
      --batch-size "$BATCH" --img-size "$IMG_SIZE" --device "$GPU" --save-json --save-regions \
      --project "$RUN_ROOT/test" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_test.log"

    log "Measuring $run_name (GFLOPs/FPS, batch=1)"
    python test.py \
      --data "$DATA_YAML" --weights "$ckpt" \
      --batch-size 1 --img-size "$IMG_SIZE" --device "$GPU" --task measure \
      --project "$RUN_ROOT/measure" --name "$run_name" --exist-ok \
      2>&1 | tee "$results_dir/${run_name}_measure.log"
  fi

  if [ ! -f "$results_dir/best_predictions.json" ]; then
    log "WARNING: $run_name produced no predictions (expected at 1 epoch under SMOKE) -- skipping audit/vt_diagnose"
  else
    log "Auditing $run_name"
    python "$SCRIPT_DIR/audit_buckets.py" \
      --pred "$results_dir/best_predictions.json" \
      --labels "$DATA_ROOT/labels/$VAL_SPLIT" --images "$DATA_ROOT/images/$VAL_SPLIT" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_audit.log" || log "WARNING: audit_buckets.py failed, continuing"

    log "vt_diagnose: $run_name"
    python "$SCRIPT_DIR/vt_diagnose.py" "$run_name" \
      --labels-dir "$DATA_ROOT/labels/$VAL_SPLIT" --images-dir "$DATA_ROOT/images/$VAL_SPLIT" \
      --classes "$CLASSES" \
      2>&1 | tee "$results_dir/${run_name}_vt_diagnose.log" || log "WARNING: vt_diagnose.py failed, continuing"
  fi
}

# R0: original baseline -- upstream loss, CIoU box
run_arm "seaperson_yolov5m_baseline${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m.yaml" upstream upstream

# semantic-only: same architecture as R0, coverage loss instead -- isolates
# the loss-function effect from the selector-architecture effect
run_arm "seaperson_yolov5m_semantic_coverage${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m.yaml" coverage upstream

# spectral-only: no semantic head contributes to the routing logit
run_arm "seaperson_yolov5m_spectral_only${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_spectral_only.yaml" coverage upstream

# concat-only: channel-pooled concat evidence, CIoU box (no SABL)
run_arm "seaperson_yolov5m_channel_pooled_concat${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_concat.yaml" coverage upstream

# gated-fusion: same evidence branches as concat-only, learned input-dependent
# sigmoid gate instead of a fixed 1x1 conv over concatenated logits -- isolates
# the fusion MECHANISM against concat-only. Added after concat-only showed a
# real per-bucket trade-off (wins Very Tiny, loses Tiny/Small/Medium-Large,
# HESOD-Experiment-Plan.md SS8.2); matches VisDrone's own long-defined E2.4
# arm, never actually run there either.
run_arm "seaperson_yolov5m_channel_pooled_dual_evidence${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_dual_evidence.yaml" coverage upstream

# concat+SABL: same concat architecture, SABL box regression
run_arm "seaperson_yolov5m_channel_pooled_concat_sabl${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_concat.yaml" coverage sabl

# concat+SABL+ISPPHead: this project's best-known lightweight recipe
run_arm "seaperson_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_concat_isphead.yaml" coverage sabl

log "===== ALL DONE ====="
log "  R0:                    $RUN_ROOT/test/seaperson_yolov5m_baseline${SUFFIX}/"
log "  Semantic-only:         $RUN_ROOT/test/seaperson_yolov5m_semantic_coverage${SUFFIX}/"
log "  Spectral-only:         $RUN_ROOT/test/seaperson_yolov5m_spectral_only${SUFFIX}/"
log "  Concat-only:           $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_concat${SUFFIX}/"
log "  Gated-fusion:          $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_dual_evidence${SUFFIX}/"
log "  Concat+SABL:           $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_concat_sabl${SUFFIX}/"
log "  Concat+SABL+ISPPHead:  $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}/"

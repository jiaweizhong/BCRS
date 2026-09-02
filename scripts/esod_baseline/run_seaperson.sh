#!/usr/bin/env bash
# SeaPerson (aka TinyPersonV2) 8-arm roster, run through the ACTIVE
# hesod/backends/hesod tree, mirroring run_uavdt.sh's resume-capable
# run_arm() structure.
#
# Arms (HESOD-Experiment-Plan.md SS8, decided alongside the TinyPerson roster):
#   R0             -- Segmenter, upstream loss, CIoU box (original baseline)
#   semantic-only  -- SAME architecture as R0, coverage loss instead (E2.1
#                      style) -- isolates the loss-function effect from the
#                      selector-architecture effect, distinct from R0.
#   spectral-only  -- SpectralOnlySegmenter, coverage loss (E2.5 style)
#   concat-only    -- ChannelPooledConcatEvidenceSegmenter, coverage loss,
#                      CIoU box (E2.9 style, no SABL)
#   gated-fusion   -- same evidence branches as concat-only, learned gate
#                      instead of fixed concat (isolates the fusion mechanism)
#   concat+SABL    -- same concat architecture, coverage loss, SABL box
#   concat+SABL+ISPPHead -- same + ISPPHead Detect (this project's best-known
#                      lightweight recipe, confirmed on TinyPerson v1)
#   concat+ISPPHead -- same as concat+SABL+ISPPHead but upstream/CIoU box
#                      loss (no SABL) -- isolates ISPPHead's saving from
#                      SABL's mixed/inconclusive accuracy effect (SS8.1)
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
  # $5 is an optional per-arm batch-size override (falls back to the shared
  # $BATCH otherwise) -- needed for spectral-only's SpectralOnlySegmenter
  # (full-width, unpooled spectral branch), which OOMs at the shared
  # batch=8/4 and only trains at batch=2 (HESOD-Experiment-Plan.md SS8);
  # every other arm shares $BATCH. Added 2026-08-23 after
  # seaperson_yolov5m_spectral_only_run2 OOM'd immediately (0/714, 6s in)
  # re-running under a shared ARMS invocation that left $BATCH at its
  # default 8 -- the original spectral-only run avoided this only because
  # it was launched as its own separate `BATCH=2 bash run_seaperson.sh`
  # invocation, not mixed into one ARMS call with batch=8 arms.
  local run_name="$1" model_cfg="$2" selector_loss="$3" box_loss="${4:-upstream}" batch="${5:-$BATCH}"

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
    log "===== Training $run_name (selector_loss=$selector_loss, box_loss=$box_loss, batch=$batch) ====="
    python train.py \
      --data "$DATA_YAML" \
      --cfg "$model_cfg" \
      --weights weights/pretrained/yolov5m.pt \
      --hyp "$HYP" \
      --batch-size "$batch" --img-size "$IMG_SIZE" --epochs "$EPOCHS" --device "$GPU" \
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
      --batch-size "$batch" --img-size "$IMG_SIZE" --device "$GPU" --save-json --save-regions \
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

# spectral-only: no semantic head contributes to the routing logit. Forced
# batch=2 override (5th arg) -- full-width unpooled SpectralBranch OOMs at
# the shared batch=8 (see run_arm()'s own comment above).
run_arm "seaperson_yolov5m_spectral_only${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_spectral_only.yaml" coverage upstream 2

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

# concat+ISPPHead (no SABL): same config as concat+SABL+ISPPHead, upstream/CIoU
# box loss instead of sabl -- isolates the ISPPHead saving from SABL. Added
# after concat+SABL showed no clean accuracy win over concat-only (small,
# mixed result -- see HESOD-Experiment-Plan.md SS8.1's concat+SABL
# interpretation) while concat+SABL+ISPPHead's own head-only isolation
# showed ISPPHead alone is a clean win: tests whether dropping SABL entirely
# (keeping just ISPPHead's efficiency gain) matches or beats
# concat+SABL+ISPPHead's accuracy with the same or better efficiency.
run_arm "seaperson_yolov5m_channel_pooled_concat_isphead${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_concat_isphead.yaml" coverage upstream

# --- Noise-check re-runs (2026-08-23), NOT part of the roster proper ---
# concat+ISPPHead's single-run delta vs concat-only/concat+SABL+ISPPHead
# (HESOD-Experiment-Plan.md SS8.1) is within this project's own <3-seed
# "not yet confirmed" band (SS4). init_seeds(2) is fixed in train.py, but
# cudnn.benchmark=True/deterministic=False (also train.py) means a second
# run still isn't byte-identical -- gives a real, independent second data
# point without needing --seed plumbing. spectral-only queued right after,
# same reasoning (its near-parity with semantic-only is a small enough
# margin to be worth a stability check too). Separate _run2 names so the
# original results (already recorded in the doc) are never overwritten.
run_arm "seaperson_yolov5m_channel_pooled_concat_isphead_run2${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_concat_isphead.yaml" coverage upstream

# channel-pooled-spectral-only: NEW arm (2026-08-23, not a rerun), isolates
# the capacity-vs-evidence-sufficiency confound in spectral-only's own
# strong result -- see the yaml's own header comment for the full
# reasoning. Should run at the shared batch=8 (pooled, unlike spectral-only
# itself), not spectral-only's forced batch=2.
run_arm "seaperson_yolov5m_channel_pooled_spectral_only${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_spectral_only.yaml" coverage upstream

run_arm "seaperson_yolov5m_spectral_only_run2${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_spectral_only.yaml" coverage upstream 2

# --- Fusion-rule regression check (2026-09-01), NOT part of the roster
# proper. On UAVDT, concat's learned 1x1 combiner provably can't represent
# a max/union rule and measurably underperforms every single-evidence arm
# (HESOD-Experiment-Plan.md SS3.4-3.5) -- torch.max fusion fixed it there
# (+2.4pp mAP@.5, +4.01pp Total Recall over concat-only, beating a soft-OR
# alternative too). SeaPerson's own concat-only is NOT broken (arm 5,
# Dual-Concat, already the roster's peak mAP@.5, 0.772) -- this arm checks
# the UAVDT fix doesn't regress what's already working here before treating
# max as a universal improvement rather than a UAVDT-specific one. Named
# "..._max_sabl_isphead" (not "..._max_isphead") to match the concat
# family's own convention (arm 8 = "..._concat_sabl_isphead" = SABL+ISPPHead
# together; "..._concat_isphead" alone means ISPPHead-only, box-loss
# upstream) -- reserves "..._max_isphead" for an ISPPHead-only arm if one
# gets added later. ---
run_arm "seaperson_yolov5m_channel_pooled_max_sabl_isphead${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_max_isphead.yaml" coverage sabl

# max fusion alone (no SABL, no ISPPHead) -- isolates the fusion rule's own
# effect from the SABL/ISPPHead confound above. The full-recipe swap (arm
# above, HESOD-Experiment-Plan.md SS4.5) lost on every metric vs. arm (8),
# but that alone can't separate "max is worse than concat here" from "max
# interacts badly with SABL/ISPPHead specifically" -- the same confound
# UAVDT's own arm 9 (fusion-only) resolved before building its Full v2.
# Compares directly against arm (5) Dual-Concat (0.772 mAP@.5, the
# roster's own peak) under identical --selector-loss coverage --box-loss
# upstream flags -- only the fusion rule differs.
run_arm "seaperson_yolov5m_channel_pooled_max${SUFFIX}" \
  "models/cfg/esod/seaperson_yolov5m_channel_pooled_max.yaml" coverage upstream

log "===== ALL DONE ====="
log "  R0:                    $RUN_ROOT/test/seaperson_yolov5m_baseline${SUFFIX}/"
log "  Semantic-only:         $RUN_ROOT/test/seaperson_yolov5m_semantic_coverage${SUFFIX}/"
log "  Spectral-only:         $RUN_ROOT/test/seaperson_yolov5m_spectral_only${SUFFIX}/"
log "  Concat-only:           $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_concat${SUFFIX}/"
log "  Gated-fusion:          $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_dual_evidence${SUFFIX}/"
log "  Concat+SABL:           $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_concat_sabl${SUFFIX}/"
log "  Concat+SABL+ISPPHead:  $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_concat_sabl_isphead${SUFFIX}/"
log "  Concat+ISPPHead:       $RUN_ROOT/test/seaperson_yolov5m_channel_pooled_concat_isphead${SUFFIX}/"

#!/bin/bash
# -----------------------------------------------------------------------------
# TEMP unattended driver: chain GLM-5.1 profiler Step 1 -> 2 -> 3.
#
# Runs the three profiler rounds back-to-back so you can leave it running
# (the skew round alone is ~1-2h per TP). Each round logs to logs/ with a
# timestamp; a failing round stops the chain (resume mode makes a re-run
# cheap — already-measured shots are skipped).
#
#   Step 1  dense / per_sequence / attention   (--skip-moe --skip-skew, full TP)
#   Step 2  moe                                (first_k_dense_replace=0, TP=1)
#   Step 3  skew                               (--only-skew, full TP)
#
# Usage (repo root, vLLM container, H20):
#     ./profiler/profile-glm51-all.sh                 # real full sweep
#     SMOKE=1 ./profiler/profile-glm51-all.sh         # tiny grid, minutes
#     nohup ./profiler/profile-glm51-all.sh > logs/glm_all.out 2>&1 &   # detach
# -----------------------------------------------------------------------------

set -euo pipefail

# =============================================================================
# EDIT THESE
# =============================================================================

MODEL="zai-org/GLM-5.1"
HARDWARE="H20"
VARIANT="fp8"                 # REQUIRED (GLM torch_dtype is null)
# GLM-5.1 FP8 weights (~670GB) only fit on a full 8-card node, so tp2/tp4
# deployments are physically impossible -> never looked up by the simulator,
# no point profiling them. tp1 is kept as a single-GPU debug/cross-check
# baseline. Step 2 (moe) always uses TP=1 regardless.
TP_DEGREES="1,8"
BLOCK_SIZE=64                 # Hopper FlashMLA / FlashMLA_Sparse require 64

# Which rounds to run, comma-separated: 1=dense, 2=moe, 3=skew.
# Default "1,2,3" runs all three. Override to test one in isolation, e.g.
# STEPS=3 ./profiler/profile-glm51-all.sh   (skew only; resume keeps 1&2).
STEPS="${STEPS:-1,2,3}"

# =============================================================================
# SWEEP GRID — real scale by default. SMOKE=1 swaps in the tiny grid.
# =============================================================================

if [[ -n "${SMOKE:-}" ]]; then
    MAX_NUM_BATCHED_TOKENS=64
    MAX_NUM_SEQS=8
    ATTENTION_MAX_KV=512
    ATTENTION_CHUNK_FACTOR=4.0
    ATTENTION_KV_FACTOR=4.0
    MEASUREMENT_ITERATIONS=1
    SKEW_FACTOR=4.0
    TP_DEGREES="1"            # one TP only for the smoke chain
else
    MAX_NUM_BATCHED_TOKENS=2048
    MAX_NUM_SEQS=256
    ATTENTION_MAX_KV=16384
    ATTENTION_CHUNK_FACTOR=2.0
    ATTENTION_KV_FACTOR=2.0
    MEASUREMENT_ITERATIONS=3
    SKEW_FACTOR=2.0
fi

# =============================================================================
# EXECUTE — usually no need to touch below this line.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

STAMP="$(date +%Y%m%d_%H%M%S)"
CUR_STEP="(none)"
trap 'echo "[$(date +%H:%M:%S)] FAILED during ${CUR_STEP} (exit $?). Re-run is safe (resume)." >&2' ERR

# Flags common to all three rounds.
common_flags() {
    local -n _out=$1
    # --dtype bfloat16 = compute/activation dtype. MUST be set explicitly:
    # GLM-5.1's config uses transformers-5.x "dtype" (not "torch_dtype"),
    # so vLLM's default dtype=auto can't find torch_dtype and falls back to
    # float16 — which FlashMLA/FlashMLA_Sparse reject on Hopper ("No valid
    # attention backend"). FP8 weights come from the config's
    # quantization_config; the folder name comes from --variant fp8.
    _out=(--hardware "$HARDWARE" --variant "$VARIANT" --dtype bfloat16)
    _out+=(--block-size "$BLOCK_SIZE")
    _out+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    _out+=(--max-num-seqs "$MAX_NUM_SEQS")
    _out+=(--attention-max-kv "$ATTENTION_MAX_KV")
    _out+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
    _out+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
    _out+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
}

run_step() {
    local name="$1" logfile="$2"; shift 2
    CUR_STEP="$name"
    local start=$SECONDS
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] START ${name}  -> logs/${logfile}"
    echo "    $*"
    echo "================================================================"
    "$@" 2>&1 | tee "logs/${logfile}"
    local mins=$(( (SECONDS - start) / 60 ))
    echo "[$(date +%H:%M:%S)] DONE ${name}  (${mins} min)"
}

flags=()
common_flags flags

want() { [[ ",$STEPS," == *",$1,"* ]]; }

# --- Step 1: dense / per_sequence / attention (skip moe + skew, full TP) ---
if want 1; then
    run_step "Step 1 (dense)" "glm_profile_dense_${STAMP}.log" \
        python3 -m profiler profile "$MODEL" "${flags[@]}" \
            --tp "$TP_DEGREES" --skip-moe --skip-skew
fi

# --- Step 2: moe (force layer 0 to MoE, TP=1, resume keeps Step 1 shots) ---
if want 2; then
    run_step "Step 2 (moe)" "glm_profile_moe_${STAMP}.log" \
        python3 -m profiler profile "$MODEL" "${flags[@]}" \
            --tp "1" --skip-skew --hf-overrides '{"first_k_dense_replace":0}'
fi

# --- Step 3: skew only (full TP, no hf-overrides) ---
if want 3; then
    run_step "Step 3 (skew)" "glm_profile_skew_${STAMP}.log" \
        python3 -m profiler profile "$MODEL" "${flags[@]}" \
            --tp "$TP_DEGREES" --only-skew \
            --skew-n-factor "$SKEW_FACTOR" --skew-pc-factor "$SKEW_FACTOR" \
            --skew-kp-factor "$SKEW_FACTOR" --skew-kvs-factor "$SKEW_FACTOR"
fi

echo
echo "================================================================"
echo "ALL DONE. Bundle: profiler/perf/$HARDWARE/$MODEL/$VARIANT/"
echo "Logs: logs/glm_profile_{dense,moe,skew}_${STAMP}.log"
[[ -n "${SMOKE:-}" ]] && echo "NOTE: SMOKE run — data is coarse/inaccurate, pathway check only."
echo "================================================================"

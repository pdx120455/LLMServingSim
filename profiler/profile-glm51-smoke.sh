#!/bin/bash
# -----------------------------------------------------------------------------
# GLM-5.1 SMOKE profile — pathway verification, NOT real data.
#
# Purpose: prove the profiler runs end-to-end on H20 without hanging
# (engine boot/teardown, per-category sweeps, hook firing, CSV write),
# using a deliberately tiny sweep grid so each round finishes in minutes
# instead of hours. The resulting latencies are USELESS for accuracy —
# the grid is too coarse and the simulator will extrapolate wildly.
# Run the real full sweep (profiler/profile.sh) with FORCE=1 afterwards.
#
# It runs the two-round MoE-model flow automatically:
#   Round 1 (dense): SKIP_MOE=1, single profiled layer is dense
#                    -> fills dense.csv / per_sequence.csv / attention.csv
#   Round 2 (moe):   HF_OVERRIDES first_k_dense_replace=0 forces the
#                    single layer to be MoE -> fills moe.csv (resume mode
#                    skips the already-measured dense/attention shots)
#
# Usage (from repo root, inside the vLLM container, on H20):
#     ./profiler/profile-glm51-smoke.sh
# -----------------------------------------------------------------------------

set -euo pipefail

# =============================================================================
# EDIT THESE
# =============================================================================

MODEL="zai-org/GLM-5.1"
HARDWARE="H20"
VARIANT="fp8"            # REQUIRED: GLM-5.1 torch_dtype is null, so without
                        # this the bundle lands in a default/ folder the
                        # simulator won't resolve for --dtype fp8.
BLOCK_SIZE=64           # Hopper FlashMLA / FlashMLA_Sparse require 64.

# Only TP=1 for the smoke (one engine boot per round). Real sweep does 1,8.
TP_DEGREES="1"

# Which rounds to run: "dense", "moe", or "both" (default).
ROUNDS="both"

# =============================================================================
# SMOKE GRID — tiny on purpose. See profiler/profile.sh for what each knob
# controls. These shrink the SHOT COUNT only; each shot still runs a real
# single-layer FP8 forward on the GPU.
# =============================================================================

MAX_NUM_BATCHED_TOKENS=64     # dense / attention-chunk / moe-token axis
MAX_NUM_SEQS=8                # per_sequence / attention-n_decode axis
ATTENTION_MAX_KV=512          # attention 4D KV axes (the dominant cost)
ATTENTION_CHUNK_FACTOR=4.0    # coarsen attention chunk axis
ATTENTION_KV_FACTOR=4.0       # coarsen attention KV axes
MEASUREMENT_ITERATIONS=1      # 1 sample/shot (real sweep uses 3)
SKIP_SKEW=1                   # skip the 1-2h skew sweep entirely

# =============================================================================
# EXECUTE — usually no need to touch below this line.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Shared flags for both rounds.
common_flags() {
    local -n _out=$1
    # --dtype bfloat16 = compute/activation dtype. MUST be set explicitly:
    # GLM-5.1's config uses transformers-5.x "dtype" (not "torch_dtype"),
    # so vLLM's default dtype=auto falls back to float16 — which
    # FlashMLA/FlashMLA_Sparse reject on Hopper ("No valid attention
    # backend"). FP8 weights come from the config's quantization_config;
    # the folder name comes from --variant fp8.
    _out=(--hardware "$HARDWARE" --variant "$VARIANT" --dtype bfloat16 --tp "$TP_DEGREES")
    _out+=(--block-size "$BLOCK_SIZE")
    _out+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    _out+=(--max-num-seqs "$MAX_NUM_SEQS")
    _out+=(--attention-max-kv "$ATTENTION_MAX_KV")
    _out+=(--attention-chunk-factor "$ATTENTION_CHUNK_FACTOR")
    _out+=(--attention-kv-factor "$ATTENTION_KV_FACTOR")
    _out+=(--measurement-iterations "$MEASUREMENT_ITERATIONS")
    [[ -n "${SKIP_SKEW:-}" ]] && _out+=(--skip-skew)
}

run_dense_round() {
    local flags
    common_flags flags
    echo "=== Round 1/2: DENSE (SKIP_MOE=1) ==="
    python3 -m profiler profile "$MODEL" "${flags[@]}" --skip-moe
}

run_moe_round() {
    local flags
    common_flags flags
    echo "=== Round 2/2: MoE (first_k_dense_replace=0) ==="
    # Resume mode (default, no --force): dense/attention/per_seq shots are
    # already present from round 1 and get skipped; only moe.csv is filled.
    python3 -m profiler profile "$MODEL" "${flags[@]}" \
        --hf-overrides '{"first_k_dense_replace":0}'
}

case "$ROUNDS" in
    dense) run_dense_round ;;
    moe)   run_moe_round ;;
    both)  run_dense_round; run_moe_round ;;
    *) echo "ROUNDS must be dense|moe|both, got '$ROUNDS'" >&2; exit 1 ;;
esac

echo
echo "Smoke profile done. Bundle: profiler/perf/$HARDWARE/$MODEL/$VARIANT/"
echo "REMINDER: this is a coarse SMOKE — data is not accurate. Before the"
echo "real sweep run profiler/profile.sh with FORCE=1 to wipe these points."

<!--
DRAFT — upstream GitHub issue for casys-kaist/LLMServingSim (Phase 4.1).
Not posted. Review, then paste into a new issue at
https://github.com/casys-kaist/LLMServingSim/issues/new
Suggested labels: enhancement, model-support, profiler, simulator
-->

# [Feature/RFC] Support MLA + Sparse Attention + fine-grained MoE models (DeepSeek-V3.2 / GLM-5.1 family)

## Summary

I'd like to add first-class support for the **DeepSeek-V3.2 architecture family** to
LLMServingSim, using **GLM-5.1** (`architectures: ["GlmMoeDsaForCausalLM"]`,
`model_type: "glm_moe_dsa"`) as the concrete target. GLM-5.1 reuses vLLM's
`deepseek_v2.py` infrastructure and combines several features that the current
simulator/profiler do not yet model:

- **MLA** (Multi-head Latent Attention) — no fused `qkv_proj`; Q/KV go through
  LoRA-style down/up projections with a compressed latent KV cache.
- **DSA / Lightning Indexer** — sparse attention that only attends to the top-`index_topk`
  (=2048) tokens, so the attention cost model is fundamentally different from dense attention.
- **Fine-grained MoE with `first_k_dense_replace`** — the first *k* decoder layers are
  dense MLP, the rest are 256-expert MoE (`n_routed_experts=256`, `num_experts_per_tok=8`,
  `n_shared_experts=1`). A single profiled layer can no longer represent the whole model.
- **FP8** (e4m3, block `[128,128]`) weight quantization.
- **MTP** head (`num_nextn_predict_layers=1`) — out of scope for a first version.

I already have a working prototype on a branch (details below) and would like to align
on design direction before opening PRs.

## Motivation

DeepSeek-V3.2 / GLM-5.1 is the first non-vanilla architecture LLMServingSim would
support where attention is **not** standard MHA/GQA. MLA + sparse attention + fine-grained
MoE is now a common production shape (DeepSeek-V3/V3.2, GLM-4.5/5.1, …), so supporting
the family — not just one checkpoint — unlocks a large class of current models.

## What the current code assumes (and where it breaks)

| Assumption | Where | Breaks because |
|---|---|---|
| Attention is a single fused `qkv_proj` | `trace_generator`, `memory_model` | MLA splits into `fused_qkv_a_proj` / `q_b_proj` / `kv_b_proj` (+ down/up LoRA) |
| KV cache size = `2 × kv_heads × head_dim × ...` (GQA) | `memory_model.calculate_sizes` | MLA caches a compressed latent: `kv_lora_rank + qk_rope_head_dim` (=576), replicated across TP, **not** ×2 |
| Every decoder layer is the same kind | `trace_generator` block-copy, `memory_model.get_weight` | `first_k_dense_replace` mixes dense + MoE layers |
| MoE detected via `num_local_experts` / `num_experts` | `trace_generator`, `config_builder`, `memory_model` | DeepSeek/GLM family uses `n_routed_experts` → these models silently fall back to "dense" |
| Attention cost is dense over `kv_decode` | `trace_generator._lookup_attention*` | Sparse attention plateaus past `index_topk`; latent KV changes the curve |

## Proposed changes

### Profiler side
- New `profiler/models/glm_moe_dsa.yaml` mapping the vLLM `nn.Module` class hierarchy
  (MLA wrapper, Indexer, FusedMoE, …) onto the dense / per_sequence / attention / moe catalog.
- MoE category needs the shared expert handled correctly (it's computed **inside**
  `FusedMoE.forward`, so `moe.csv` already includes it — the forced-routing hook must not
  bypass the shared-expert kernel).
- `first_k_dense_replace`: profile dense-MLP layers and MoE layers separately
  (controllable today via `hf_overrides`).

### Simulator side
- `memory_model.py`: MLA branch for KV-cache sizing (latent, TP-replicated, not ×2) and
  per-layer dense/MoE weight accounting under `first_k_dense_replace`.
- `trace_generator.py`: drive emission from the yaml `sequence:` (MLA projections + Indexer
  as ordinary dense lookups; `kv_b_proj` time folded into the attention kernel to avoid
  double counting); per-layer dense/MoE segmentation in block-copy and interleaved paths.
- Open design questions on the attention/DSA cost model (see below).

## Model-agnostic bugs found along the way

While prototyping I hit several bugs that are **not** GLM-specific — they affect every
DeepSeek-V2/V3 and GLM-MoE model, and would make good standalone fixes regardless of
this feature:

1. **MoE path never activates for `n_routed_experts` models.** `trace_generator`'s layer
   dispatch, `config_builder`'s `is_moe` check, and `memory_model`'s `is_moe` check all
   only look at `num_local_experts` / `num_experts`. DeepSeek/GLM use `n_routed_experts`,
   so these models are treated as dense (wrong EP default, skipped EP-divides-experts
   validation, and severe weight under-counting for 256-expert layers).
2. **All DP topologies were broken with the power model.** `trace_generator` passed the
   full `comm_type` string (e.g. `"ALLREDUCE:1,0"`, with the `involved_dim` suffix) into
   `power_model.total_ring_data`, which only recognizes bare collective names and raises
   `Unknown collective`. Any MoE model with a 2D `involved_dim` config hits this.
3. **Default-dtype peek read the wrong path and the wrong key.** In `serving/__main__.py`,
   the `torch_dtype` peek opened the cluster config relative to cwd (but `main()` has
   already `chdir`'d into `astra-sim/`) and read a top-level `instances` key that only
   exists post-parse. Triggered by any model with `torch_dtype: null` and no `--dtype`.

If useful, I'm happy to split these out as small independent PRs ahead of the larger feature.

## What's already prototyped

On a working branch (`feat/glm5.1-support`):

- `profiler/models/glm_moe_dsa.yaml` (13 dense + 2 per_sequence + 1 attention + 1 moe;
  passes the existing pydantic catalog validation).
- `memory_model` MLA branch + per-layer `first_k_dense_replace` weight accounting.
- `trace_generator` yaml-`sequence`-driven emission with MLA projections, Indexer, and
  per-layer dense/MoE segmentation.
- The model-agnostic bug fixes above.
- Example cluster configs (single-node TP=1, TP=2/EP=2, and an 8×H20 TP=8/EP=8 template).
- **End-to-end smoke** (serving → Chakra converter → ASTRA-Sim → cycle feedback) passing
  on TP=1 and TP=2/EP=2 with a *placeholder* perf bundle. (Cycle values are not yet
  physically meaningful — real numbers need an H20 profiling run; this validates the
  execution path and IPC, not accuracy.)

## Open questions for maintainers

1. **DSA / Lightning Indexer cost model** — separate `indexer` catalog category vs.
   merging indexer time into the dense lookups? My v1 merges (smaller diff: only
   `trace_generator`), at the cost of being kv-length-insensitive on long-context decode.
   Is a dedicated category (touching `categories.py` / `writer.py`) preferred upstream?
2. **MLA attention lookup dimensionality** — the current 4D attention grid keys on
   `(prefill_chunk, kv_prefill, n_decode, kv_decode)`. With a compressed latent KV +
   sparse top-k, do you want an extra axis, or is reusing the 4D grid (recalibrated on
   real data) acceptable for a first version?
3. **`noaux_tc` expert routing** — first version approximates with `BALANCED`/`RR`.
   Is a faithful `noaux_tc` token distribution in scope, or fine to defer?
4. **MTP head** — confirm it's OK to ignore `num_nextn_predict_layers` in v1.

## Plan / scope

I'm tracking the full effort (architecture facts, decision log, validation steps) in a
`GLM5_1_SUPPORT_PLAN.md` I can share. Proposed PR split:
(a) profiler side — `glm_moe_dsa.yaml` + any `categories.py` changes;
(b) simulator side — `memory_model` + `trace_generator`;
plus the three model-agnostic bug fixes as small standalone PRs if you'd prefer them first.

Would love guidance on the open questions before I send the PRs.

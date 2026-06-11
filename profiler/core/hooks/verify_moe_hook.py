"""Standalone verification for the MoE forced-routing hook.

Run inside the vLLM environment on a machine with one GPU:

    python -m profiler.core.hooks.verify_moe_hook

Instantiates a small unquantized FusedMoE, runs a forward pass under
``force_moe_routing``, and asserts — via the router's capture_fn, which
sits inside ``select_experts`` downstream of ``_compute_routing`` — that
the fused kernel consumed the forged topk_ids. Also checks that the patch
is fully reverted on context exit and that forced routing actually changes
the kernel output.

Re-run this after any vLLM version bump before profiling an MoE model:
the hook targets internal vLLM APIs (see moe_hook.py module docstring).

Backend note: monolithic MoE backends (FlashInfer / TRT-LLM fused paths)
route inside the kernel and are incompatible with forced routing — the
hook raises if one is selected. If that happens here or in a real profile
run, disable the backend via the matching env switch (e.g.
``VLLM_USE_FLASHINFER_MOE_FP16=0`` / ``VLLM_USE_FLASHINFER_MOE_FP8=0``).
"""
import os

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.v1.worker.workspace import init_workspace_manager

from profiler.core.hooks.moe_hook import ExpertRoute, _layer_router, force_moe_routing

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 64
INTERMEDIATE = 128
NUM_TOKENS = 16
ACTIVATED = 5


def main():
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29512",
    )
    init_workspace_manager(torch.device("cuda:0"))

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(1, 1)
        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
        ).cuda()

        with torch.no_grad():
            for param in layer.parameters():
                if param.dtype.is_floating_point:
                    torch.nn.init.normal_(param, std=0.02)
        quant_method = getattr(layer, "quant_method", None)
        if quant_method is not None and hasattr(quant_method, "process_weights_after_loading"):
            quant_method.process_weights_after_loading(layer)

    entry = "forward_native" if hasattr(FusedMoE, "forward_native") else "forward"
    router = _layer_router(layer)
    print(f"FusedMoE entry: {entry}")
    print(f"router type: {type(router).__name__}")

    captured = []
    router.set_capture_fn(lambda ids: captured.append(ids.detach().clone()))

    route = ExpertRoute.forge(layer, NUM_TOKENS, ACTIVATED)

    hidden = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32, device="cuda")

    original_compute = router._compute_routing

    with set_forward_context(None, vllm_config):
        with force_moe_routing(route):
            out_forced = layer(hidden, logits)
        ids_forced = captured[-1]

        assert router._compute_routing == original_compute, "router forge leaked"
        out_natural = layer(hidden, logits)
        ids_natural = captured[-1]

    assert torch.equal(ids_forced.cpu().to(torch.int64), route.ids.cpu().to(torch.int64)), \
        "FAIL: forged topk_ids did not reach select_experts"
    distinct = ids_forced.unique().numel()
    assert distinct == ACTIVATED, f"FAIL: distinct experts {distinct} != {ACTIVATED}"
    assert not torch.equal(ids_natural.cpu().to(torch.int64), route.ids.cpu().to(torch.int64)), \
        "FAIL: natural routing equals forged route (forge leaked past restore?)"
    assert out_forced.shape == (NUM_TOKENS, HIDDEN)
    assert not torch.equal(out_forced, out_natural), \
        "FAIL: forced and natural outputs identical (routing had no kernel effect?)"

    print(f"forced distinct experts: {distinct} (target {ACTIVATED})")
    print(f"natural distinct experts: {ids_natural.unique().numel()}")
    print("MOE HOOK VERIFIED on vLLM", __import__("vllm").__version__)


if __name__ == "__main__":
    main()

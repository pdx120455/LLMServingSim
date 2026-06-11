"""MoE forced-routing hook.

To profile the MoE block cleanly across the (tokens, activated_experts)
grid we need to control which experts receive tokens — relying on the
(dummy-weighted) learned gate would give unpredictable activation
patterns and poor grid coverage.

This module provides:

* ``ExpertRoute.forge``: build the ``(topk_weights, topk_ids)`` tensors
  that would force a given number of experts to be activated over a
  given number of tokens.

* ``force_moe_routing``: a context manager that monkey-patches the
  FusedMoE forward entry for the duration of the block so that the
  router's ``_compute_routing`` returns our forged tensors instead
  of whatever the actual learned gate produces. The patch is
  reverted on exit.

vLLM version coverage:

* <= 0.20: ``FusedMoE.forward_native`` exists and the router lives at
  ``layer.router``.
* >= 0.21: ``forward_native`` is gone — ``FusedMoE.forward`` delegates
  to ``layer.runner`` (MoERunner) and the router lives at
  ``layer.runner.router``. ``BaseRouter`` keeps the same
  ``select_experts -> _compute_routing`` template, so forging
  ``_compute_routing`` still pins the (topk_weights, topk_ids) pair
  the fused kernel consumes. Monolithic quant methods
  (``quant_method.is_monolithic``) route *inside* the kernel and never
  call ``select_experts`` — forced routing cannot apply there, so the
  hook raises instead of silently profiling garbage.

These are internal vLLM APIs. A vLLM version bump may require updating
the monkey-patch to match renamed or restructured symbols.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Iterator

import torch


def _layer_router(layer):
    """Resolve the router object for a FusedMoE layer across vLLM versions.

    vLLM >= 0.21 moved the router into the MoERunner (``layer.runner.router``);
    earlier versions kept it directly on the layer (``layer.router``).
    """
    runner = getattr(layer, "runner", None)
    if runner is not None and hasattr(runner, "router"):
        return runner.router
    return layer.router


@dataclass
class ExpertRoute:
    """Precomputed tensors that pin routing to a specific expert set.

    Attributes:
        layer_name: Identifier of the FusedMoE layer this route targets.
            The runtime patch only applies when the currently-forwarding
            layer's ``layer_name`` matches.
        weights: Tensor of shape (num_tokens, top_k); each row is a
            uniform ``1/top_k`` distribution over the chosen experts.
        ids: Integer tensor of shape (num_tokens, top_k) naming the
            experts each token is routed to.
    """

    layer_name: str
    weights: torch.Tensor
    ids: torch.Tensor

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def forge(
        cls,
        layer,
        num_tokens: int,
        activated_experts: int,
    ) -> "ExpertRoute":
        """Allocate ``weights`` and ``ids`` for a single FusedMoE layer.

        Args:
            layer: The ``FusedMoE`` instance we'll patch. Used to read
                ``top_k`` and the expected indices dtype.
            num_tokens: Number of tokens being routed this call.
            activated_experts: How many distinct experts should receive
                at least one token. Must satisfy
                ``top_k <= activated_experts <= num_tokens * top_k``.
        """
        top_k = layer.top_k
        if activated_experts < top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) must be >= "
                f"top_k ({top_k})"
            )
        if activated_experts > num_tokens * top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) cannot exceed "
                f"num_tokens*top_k ({num_tokens * top_k})"
            )

        ids_rows = _cycle_expert_ids(num_tokens, top_k, activated_experts)

        # vLLM's router cares about the dtype of topk_ids: some kernels
        # expect int32, others a specific dtype reported by the router.
        indices_dtype = _layer_router(layer)._get_indices_type()
        device = next(layer.parameters()).device

        ids = torch.tensor(
            ids_rows,
            device=device,
            dtype=torch.int32 if indices_dtype is None else indices_dtype,
        )
        # Shape sanity check — the kernel will complain later with a
        # cryptic message if this is wrong, so we prefer to fail early.
        expected_shape = (num_tokens, top_k)
        if tuple(ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch: expected {expected_shape}, "
                f"got {tuple(ids.shape)}"
            )

        weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            device=device,
            dtype=torch.float32,
        )
        return cls(
            layer_name=layer.layer_name,
            weights=weights,
            ids=ids,
        )


def _cycle_expert_ids(
    num_tokens: int,
    top_k: int,
    activated_experts: int,
) -> list[list[int]]:
    """Assign expert ids deterministically so exactly ``activated_experts``
    distinct ids appear, cycled across the token dimension.

    The specific assignment doesn't matter for latency — only the count
    of distinct activations does. We use the simplest pattern:
    ``id = (token_idx * top_k + offset) % activated_experts``.
    """
    return [
        [
            (token_idx * top_k + offset) % activated_experts
            for offset in range(top_k)
        ]
        for token_idx in range(num_tokens)
    ]


# ---------------------------------------------------------------------------
# Context manager: live FusedMoE patch
# ---------------------------------------------------------------------------

@contextmanager
def force_moe_routing(route: ExpertRoute | None) -> Iterator[None]:
    """Patch the FusedMoE forward entry to use ``route`` when called.

    If ``route`` is None the function is a no-op (useful for dense
    profile categories where we still pass through the MoE-aware
    execute path).

    The patch is layer-scoped: only the specific FusedMoE whose
    ``layer_name`` matches ``route.layer_name`` is affected; other MoE
    layers (if any) fall through to their normal forward. This matters
    if the model has multiple MoE layers and we're profiling only one
    at a time. For the single-layer test model we override to 1
    decoder layer, this is moot.

    Entry point per vLLM version: ``forward_native`` when it exists
    (<= 0.20), otherwise ``forward`` (>= 0.21, which delegates to the
    MoERunner). In both, the router's ``select_experts`` template calls
    ``_compute_routing`` — forging that on the layer's own router
    instance for the duration of the call pins (topk_weights, topk_ids).
    """
    if route is None:
        yield
        return

    # Local import so that host-side code doesn't pay the vLLM import
    # cost just to read this module.
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    entry_name = (
        "forward_native" if hasattr(FusedMoE, "forward_native") else "forward"
    )
    original_entry = getattr(FusedMoE, entry_name)

    @wraps(original_entry)
    def hooked_entry(self, hidden_states, router_logits, *args, **kwargs):
        # Only patch the specific layer we care about. Any other
        # FusedMoE encountered during this forward pass uses its
        # normal routing.
        if self.layer_name != route.layer_name:
            return original_entry(self, hidden_states, router_logits, *args, **kwargs)

        # Monolithic quant methods (e.g. some FlashInfer/TRT-LLM FP8
        # paths) fuse routing into the kernel and never call
        # select_experts — the forge below would silently not apply and
        # the (tokens, activated_experts) grid would be garbage. Fail
        # loudly so the backend can be switched off instead.
        runner = getattr(self, "runner", None)
        quant_method = getattr(runner, "_quant_method", None) if runner is not None else None
        if quant_method is not None and getattr(quant_method, "is_monolithic", False):
            raise RuntimeError(
                f"FusedMoE layer {self.layer_name} uses a monolithic quant "
                f"method ({type(quant_method).__name__}) that routes inside "
                "the kernel; forced expert routing cannot apply. Disable the "
                "monolithic MoE backend (e.g. FlashInfer/TRT-LLM fused MoE) "
                "and re-profile."
            )

        # We also sanity-check that our forged topk_ids matches the
        # actual per-call token count. If hidden_states is padded
        # differently than expected, bail.
        expected_shape = (hidden_states.shape[0], self.top_k)
        if tuple(route.ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch for {self.layer_name}: "
                f"expected {expected_shape}, got {tuple(route.ids.shape)}"
            )

        # Forge _compute_routing on this layer's own router instance for
        # the duration of the call. select_experts keeps running its
        # normalization / EPLB / dtype steps around our forged values.
        # The router object is per-layer, and we restore in finally, so
        # the forge cannot leak to other layers or later calls.
        router = _layer_router(self)
        original_compute_routing = router._compute_routing

        @wraps(original_compute_routing)
        def forced_compute_routing(_hidden_states, _router_logits, _indices_type,
                                   *_args, **_kwargs):
            # Args are deliberately ignored — the whole point of forced
            # routing is that we return pre-forged values regardless of
            # the learned gate's logits. (>= 0.21 also passes input_ids;
            # the catch-alls absorb signature drift.)
            return route.weights, route.ids

        router._compute_routing = forced_compute_routing
        try:
            return original_entry(self, hidden_states, router_logits, *args, **kwargs)
        finally:
            # Always restore, even if the forward raises (otherwise
            # subsequent MoE calls in the same profile session would
            # keep returning our forged values).
            router._compute_routing = original_compute_routing

    setattr(FusedMoE, entry_name, hooked_entry)
    try:
        yield
    finally:
        # Restore the original class method so future calls (in tests,
        # or after this profile session ends) behave normally.
        setattr(FusedMoE, entry_name, original_entry)


# ---------------------------------------------------------------------------
# Helpers that run worker-side
# ---------------------------------------------------------------------------

def single_moe_layer(model_runner):
    """Return the model's lone ``FusedMoE`` layer.

    We run profiling with ``hf_overrides.num_hidden_layers=1`` so
    there's exactly one of every decoder sub-module — including MoE.
    If for some reason there are zero or more-than-one, raise so the
    caller can investigate rather than forge the wrong route.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    model = model_runner.get_model()
    moe_layers = [m for m in model.modules() if isinstance(m, FusedMoE)]
    if len(moe_layers) != 1:
        raise RuntimeError(
            f"Expected exactly one FusedMoE layer in the test model, "
            f"got {len(moe_layers)}"
        )
    return moe_layers[0]

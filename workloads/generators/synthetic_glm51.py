#!/usr/bin/env python3
"""Offline synthetic workload generator for GLM-5.1 bench/sim validation.

Step 5 of H20_RUNBOOK.md runs `python -m bench run` against the real
GLM-5.1 vLLM engine, then Step 6 replays the *same* dataset through the
simulator. On an offline intranet the ShareGPT generator can't be used
(it needs a HF tokenizer + dataset download), so this script produces a
flat-format JSONL with valid random token ids using only the Python
standard library — no tokenizer, no network.

Why random ids are fine: per-token compute cost (attention / GEMM) is
content-independent, so for *latency* validation only the token counts
and arrival pattern matter. bench pins the prompt via `input_tok_ids`
and the decode count via `output_toks`; the simulator replays the same
counts. Token *values* never enter the timing.

Output schema (one JSON object per line, flat format — bench skips any
row carrying a `sub_requests` key):

    {
      "input_toks":      <int>,            # == len(input_tok_ids)
      "output_toks":     <int>,            # == len(output_tok_ids)
      "arrival_time_ns": <int>,            # Poisson arrivals (see --sps)
      "input_tok_ids":   [<int>, ...],     # valid non-special GLM-5.1 ids
      "output_tok_ids":  [<int>, ...]      # placeholder ids, count is what matters
    }

Example (matches the runbook's Step 5/6 contract — same --num-reqs both sides):

    python3 workloads/generators/synthetic_glm51.py \
        --num-reqs 64 --sps 10 \
        --input-min 128 --input-max 1024 \
        --output-min 64 --output-max 512 \
        --output workloads/glm51_synth_64.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Stay well inside the attention sweep's kv cap (ATTENTION_MAX_KV=16384)
# and away from the long-context regime the DSA indexer cost (R12) does
# not model. The first validation should use modest lengths.
_DEFAULT_SEQ_GUARD = 4096

# Skip the low id band (bos/special tokens often live there) and the top
# band (eos / pad: GLM-5.1 uses 154820/154827/154829, pad 154820) so a
# generated prompt can never accidentally carry an eos that early-stops
# the engine.
_LOW_ID_RESERVE = 100


def _load_vocab_size(model_config: Path) -> int:
    with open(model_config) as f:
        cfg = json.load(f)
    vocab = cfg.get("vocab_size")
    if not isinstance(vocab, int) or vocab <= _LOW_ID_RESERVE + 1024:
        raise ValueError(
            f"vocab_size missing or too small in {model_config}: {vocab!r}"
        )
    return vocab


def _rand_ids(rng: random.Random, n: int, lo: int, hi: int) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(n)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="synthetic_glm51",
        description="Offline synthetic GLM-5.1 workload (flat JSONL).",
    )
    p.add_argument("--num-reqs", type=int, required=True, dest="num_reqs",
                   help="Number of requests. MUST match bench --num-reqs "
                        "and the simulator --num-reqs in Steps 5/6.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL path.")
    p.add_argument("--model-config", type=Path,
                   default=Path("configs/model/zai-org/GLM-5.1.json"),
                   dest="model_config",
                   help="Model config to read vocab_size from. "
                        "Default: configs/model/zai-org/GLM-5.1.json.")
    p.add_argument("--sps", type=float, default=10.0,
                   help="Mean arrival rate (requests/sec). Inter-arrival "
                        "times are exponential (Poisson process). Default 10.")
    p.add_argument("--first-arrival-sec", type=float, default=0.0,
                   dest="first_arrival_sec",
                   help="Offset (seconds) added to the first arrival.")
    p.add_argument("--input-min", type=int, default=128, dest="input_min")
    p.add_argument("--input-max", type=int, default=1024, dest="input_max")
    p.add_argument("--output-min", type=int, default=64, dest="output_min")
    p.add_argument("--output-max", type=int, default=512, dest="output_max")
    p.add_argument("--seq-guard", type=int, default=_DEFAULT_SEQ_GUARD,
                   dest="seq_guard",
                   help="Max allowed input+output tokens per request. "
                        f"Default {_DEFAULT_SEQ_GUARD}. Keeps lengths inside "
                        "the profiler attention grid and out of the "
                        "long-context regime (R12: indexer cost unmodeled).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (reproducible datasets).")
    args = p.parse_args(argv)

    # Validate ranges.
    if args.num_reqs <= 0:
        p.error("--num-reqs must be positive")
    if args.input_min < 1 or args.input_min > args.input_max:
        p.error("require 1 <= --input-min <= --input-max")
    if args.output_min < 1 or args.output_min > args.output_max:
        p.error("require 1 <= --output-min <= --output-max")
    if args.input_max + args.output_max > args.seq_guard:
        p.error(
            f"--input-max + --output-max = "
            f"{args.input_max + args.output_max} exceeds --seq-guard "
            f"{args.seq_guard}; raise --seq-guard only if you know the "
            f"profiler attention grid covers that length."
        )
    if args.sps <= 0:
        p.error("--sps must be positive")

    vocab = _load_vocab_size(args.model_config)
    id_lo = _LOW_ID_RESERVE
    id_hi = vocab - 1025  # leave the top band (eos/pad) untouched
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    time_ns = int(args.first_arrival_sec * 1_000_000_000)
    in_lens: list[int] = []
    out_lens: list[int] = []

    with open(args.output, "w") as f:
        for i in range(args.num_reqs):
            n_in = rng.randint(args.input_min, args.input_max)
            n_out = rng.randint(args.output_min, args.output_max)
            row = {
                "input_toks": n_in,
                "output_toks": n_out,
                "arrival_time_ns": int(time_ns),
                "input_tok_ids": _rand_ids(rng, n_in, id_lo, id_hi),
                "output_tok_ids": _rand_ids(rng, n_out, id_lo, id_hi),
            }
            f.write(json.dumps(row) + "\n")
            in_lens.append(n_in)
            out_lens.append(n_out)
            # Exponential inter-arrival after the first request.
            if i < args.num_reqs - 1:
                time_ns += int(rng.expovariate(args.sps) * 1_000_000_000)

    span_s = time_ns / 1e9
    print(
        f"Wrote {args.num_reqs} requests -> {args.output}\n"
        f"  vocab_size={vocab}  id range [{id_lo}, {id_hi}]\n"
        f"  input_toks  : min={min(in_lens)} max={max(in_lens)} "
        f"mean={sum(in_lens) / len(in_lens):.0f}\n"
        f"  output_toks : min={min(out_lens)} max={max(out_lens)} "
        f"mean={sum(out_lens) / len(out_lens):.0f}\n"
        f"  arrivals    : sps={args.sps}  span={span_s:.2f}s\n"
        f"  NEXT: pass the SAME --num-reqs ({args.num_reqs}) to "
        f"`bench run` (Step 5) and the simulator (Step 6)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

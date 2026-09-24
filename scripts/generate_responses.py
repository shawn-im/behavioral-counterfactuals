#!/usr/bin/env python3
from __future__ import annotations

"""Sample model responses for paired control/intervention prompt splits."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from behavior_pipeline.runtime_env import configure_runtime_caches  # noqa: E402

configure_runtime_caches(Path(__file__).resolve().parents[1])


def _configure_devices(devices: str | None) -> int:
    # vLLM chooses batch scheduling internally. We only expose the visible CUDA
    # devices and tensor-parallel size.
    """Internal helper for configure_devices."""
    if not devices:
        return 1
    devs = [d.strip() for d in devices.split(",") if d.strip()]
    if not devs:
        return 1
    ordinals = []
    for dev in devs:
        if dev.startswith("cuda:"):
            ordinals.append(dev.split(":", 1)[1])
        elif dev.isdigit():
            ordinals.append(dev)
        else:
            raise SystemExit(f"Invalid device '{dev}'. Use cuda:N or N.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ordinals)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return len(ordinals)


def _validate_balanced_rollouts(
    rows: list[dict], *, split: str, expected_rollouts: int
) -> None:
    grouped: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(rows):
        source_id = str(row.get("source_id") or row.get("prompt_id") or "")
        arm = str(row.get("arm") or row.get("condition") or "")
        if not source_id or arm not in {"control", "intervention"}:
            raise ValueError(
                f"{split} row {index} lacks a valid source_id and paired arm"
            )
        rollout_id = int(row.get("rollout_id", -1))
        grouped.setdefault(
            source_id, {"control": [], "intervention": []}
        )[arm].append(rollout_id)
    expected = list(range(int(expected_rollouts)))
    for source_id, arms in grouped.items():
        for arm in ("control", "intervention"):
            actual = sorted(arms[arm])
            if actual != expected:
                raise ValueError(
                    f"unbalanced generated rollouts for {split} source={source_id!r} "
                    f"arm={arm}: expected {expected}, found {actual}"
                )


def _seed_everything(seed: int, np_module, torch_module) -> None:
    random.seed(int(seed))
    np_module.random.seed(int(seed))
    torch_module.manual_seed(int(seed))
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(int(seed))


def _validate_vllm_prompt_token_ids(
    output,
    expected_prompt_ids: list[int],
    *,
    split: str,
    row_index: int,
) -> None:
    actual = getattr(output, "prompt_token_ids", None)
    if actual is None:
        raise RuntimeError(
            "vLLM did not return prompt_token_ids; exact generation/replay parity cannot be verified"
        )
    actual_ids = [int(value) for value in actual]
    expected_ids = [int(value) for value in expected_prompt_ids]
    if actual_ids == expected_ids:
        return
    mismatch = next(
        (
            index
            for index, (actual_id, expected_id) in enumerate(
                zip(actual_ids, expected_ids)
            )
            if actual_id != expected_id
        ),
        min(len(actual_ids), len(expected_ids)),
    )
    raise RuntimeError(
        "vLLM and Transformers tokenized a rendered prompt differently: "
        f"split={split} row={row_index} first_mismatch={mismatch} "
        f"vllm_length={len(actual_ids)} hf_length={len(expected_ids)}"
    )


def main() -> None:
    """Run this module as a command-line entrypoint."""
    ap = argparse.ArgumentParser(description="Generate responses with vLLM.")
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory with strict train/val/test JSONL prompt splits.",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--devices", default=None, help="Comma-separated CUDA devices, e.g. cuda:0,cuda:1")
    ap.add_argument("--max_new_tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top_p", type=float)
    ap.add_argument("--top_k", type=int)
    ap.add_argument("--repetition_penalty", type=float)
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--think_start")
    ap.add_argument("--think_end")
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--vllm_max_model_len", type=int, default=None)
    ap.add_argument("--vllm_model_impl", default="auto", help="Passed to vLLM as model_impl; use for compatibility debugging.")
    ap.add_argument("--n_rollouts", type=int, default=1, help="Number of sampled completions per prompt row.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from behavior_pipeline.io import load_split_dir, write_jsonl
    from behavior_pipeline.model_utils import (
        apply_chat_template,
        load_tokenizer,
        model_family,
        pretrained_load_kwargs,
        reasoning_trace_markers,
        resolve_generation_settings,
    )
    from behavior_pipeline.token_text import decode_token_ids
    from behavior_pipeline.trace import find_trace_indices

    tp_size = _configure_devices(args.devices)
    _seed_everything(int(args.seed), np, torch)
    think_start, think_end = reasoning_trace_markers(args.model_name)
    if args.think_start is not None:
        think_start = args.think_start
    if args.think_end is not None:
        think_end = args.think_end

    gen = resolve_generation_settings(
        args.model_name,
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
    )
    temperature = float(gen["temperature"])
    top_p = float(gen["top_p"])
    top_k = int(gen["top_k"])
    repetition_penalty = float(gen["repetition_penalty"])

    load_kwargs = pretrained_load_kwargs(args.model_name)
    tokenizer = load_tokenizer(args.model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    llm_kwargs = {}
    if model_family(args.model_name) == "gemma":
        llm_kwargs["dtype"] = "float32"
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0}
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = int(args.vllm_max_model_len)
    if args.vllm_model_impl and args.vllm_model_impl != "auto":
        llm_kwargs["model_impl"] = args.vllm_model_impl
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        trust_remote_code=bool(load_kwargs.get("trust_remote_code", False)),
        **llm_kwargs,
    )

    sampling_kwargs = {
        "max_tokens": int(args.max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if top_k is not None:
        sampling_kwargs["top_k"] = int(top_k)
    if repetition_penalty is not None:
        sampling_kwargs["repetition_penalty"] = float(repetition_penalty)
    if int(args.n_rollouts) > 1:
        sampling_kwargs["n"] = int(args.n_rollouts)
    sampling_kwargs["seed"] = int(args.seed)
    sampling_params = SamplingParams(**sampling_kwargs)

    print(
        "[sampling] "
        f"max_tokens={sampling_kwargs['max_tokens']} temperature={sampling_kwargs['temperature']} "
        f"top_p={sampling_kwargs['top_p']} top_k={sampling_kwargs.get('top_k')} "
        f"repetition_penalty={sampling_kwargs.get('repetition_penalty')} tp_size={tp_size}"
    )

    splits = load_split_dir(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        for split, rows in splits.items():
            # Render all prompts first so tokenization, vLLM input text, and
            # saved rendered_prompt agree exactly.
            rendered_prompts = [
                str(r["prompt"]) if args.no_chat_template else apply_chat_template(tokenizer, str(r["prompt"]))
                for r in rows
            ]
            prompt_id_rows = [tokenizer(p, add_special_tokens=False).input_ids for p in rendered_prompts]
            # Pass the exact audited HF token IDs to vLLM. This prevents either
            # runtime from silently applying a different tokenizer pipeline.
            tokenized_prompts = [
                TokensPrompt(prompt_token_ids=[int(value) for value in prompt_ids])
                for prompt_ids in prompt_id_rows
            ]
            outputs = llm.generate(tokenized_prompts, sampling_params)
            generated = []
            for i, (row, rendered, prompt_ids, output) in enumerate(zip(rows, rendered_prompts, prompt_id_rows, outputs)):
                _validate_vllm_prompt_token_ids(
                    output, prompt_ids, split=split, row_index=i
                )
                # Save both ids and text. Later activation extraction can avoid
                # retokenization and use trace_token_indices directly.
                for rollout_id, output_obj in enumerate(output.outputs or []):
                    raw_response = output_obj.text if output_obj is not None else ""
                    response_ids = [int(x) for x in (getattr(output_obj, "token_ids", None) or tokenizer(raw_response, add_special_tokens=False).input_ids)]
                    response = decode_token_ids(tokenizer, response_ids)
                    full_ids = [int(x) for x in prompt_ids] + response_ids
                    source_id = str(row.get("source_id") or row.get("example_id") or f"{split}:{i}")
                    generated.append(
                        {
                            **row,
                            "prompt_id": source_id,
                            "condition": str(row.get("arm", "")),
                            "rollout_id": int(rollout_id),
                            "n_rollouts": int(args.n_rollouts),
                            "rendered_prompt": rendered,
                            "response": response,
                            "finish_reason": getattr(output_obj, "finish_reason", None),
                            "full_text": decode_token_ids(tokenizer, full_ids),
                            "prompt_input_ids": [int(x) for x in prompt_ids],
                            "response_input_ids": response_ids,
                            "input_ids": full_ids,
                            "trace_token_indices": find_trace_indices(tokenizer, prompt_ids, response_ids, think_start, think_end),
                        }
                    )
                if (i + 1) % 25 == 0 or (i + 1) == len(rows):
                    print(f"[generate] {split}: {i + 1}/{len(rows)}")
            splits[split] = generated
            _validate_balanced_rollouts(
                generated,
                split=split,
                expected_rollouts=int(args.n_rollouts),
            )

        for split, rows in splits.items():
            write_jsonl(out / f"{split}.jsonl", rows)
        (out / "generation_metadata.json").write_text(
            json.dumps(
                {
                    "model_name": str(args.model_name),
                    "tokenizer": {
                        "class": type(tokenizer).__name__,
                        "backend": (
                            str(tokenizer.backend)
                            if getattr(tokenizer, "backend", None) is not None
                            else None
                        ),
                        "whitespace_roundtrip_validated": True,
                        "vllm_input": "prompt_token_ids",
                    },
                    "seed": int(args.seed),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "repetition_penalty": float(repetition_penalty),
                    "max_new_tokens": int(args.max_new_tokens),
                    "n_rollouts": int(args.n_rollouts),
                    "chat_template": not bool(args.no_chat_template),
                    "splits": {
                        split: {
                            "rows": len(rows),
                            "sources": len(
                                {str(row["source_id"]) for row in rows}
                            ),
                        }
                        for split, rows in splits.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        del llm


if __name__ == "__main__":
    main()

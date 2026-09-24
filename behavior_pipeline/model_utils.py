from __future__ import annotations

"""Model-loading and tokenizer utilities.

This module is the only place that should know about model-family quirks such
as layer paths, generation defaults, chat templates, and reasoning markers.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
)


def resolve_transformer_layers(model: nn.Module):
    # Different model families nest transformer blocks under different names.
    # Keep this resolver broad so feature extraction stays model-agnostic.
    """Return the model's ordered transformer block collection."""
    candidate_paths = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
        ("model", "decoder", "layers"),
        ("language_model", "model", "layers"),
        ("layers",),
    ]
    for path in candidate_paths:
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    if hasattr(model, "get_decoder"):
        decoder = model.get_decoder()
        if hasattr(decoder, "layers"):
            return decoder.layers
    raise AttributeError(f"Cannot locate transformer layers for {type(model).__name__}")


def resolve_hidden_size(model: nn.Module) -> int:
    """Infer the residual-stream width from common model config fields."""
    configs = [getattr(model, "config", None)]
    for cfg in list(configs):
        if cfg is None:
            continue
        for name in ("text_config", "language_config", "decoder_config"):
            nested = getattr(cfg, name, None)
            if nested is not None:
                configs.append(nested)
    for cfg in configs:
        if cfg is None:
            continue
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(cfg, attr, None)
            if value:
                return int(value)
    raise AttributeError("Cannot infer model hidden size")


def resolve_final_norm(model: nn.Module, layers=None) -> nn.Module:
    """Locate the normalization applied after the final transformer block."""
    candidate_paths = [
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "decoder", "final_layer_norm"),
        ("language_model", "model", "norm"),
        ("language_model", "norm"),
        ("model", "language_model", "model", "norm"),
        ("model", "language_model", "norm"),
        ("norm",),
    ]
    for path in candidate_paths:
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if isinstance(obj, nn.Module):
            return obj
    if layers is not None:
        for module in model.modules():
            if getattr(module, "layers", None) is layers:
                for name in ("norm", "final_layer_norm", "ln_f"):
                    candidate = getattr(module, name, None)
                    if isinstance(candidate, nn.Module):
                        return candidate
    raise AttributeError(
        f"Cannot locate final transformer normalization for {type(model).__name__}"
    )


@dataclass
class LoadedModel:
    """Loaded transformer model, tokenizer, and resolved layer collection."""
    model: nn.Module
    tokenizer: object
    layers: object


def validate_tokenizer_roundtrip(tokenizer, model_name: str | None = None) -> None:
    """Check that encoding and decoding preserve ordinary spaces and newlines.

    The model's serialized tokenizer determines token boundaries. This check
    rejects a loader whose decoded output changes whitespace in the probe.
    """
    probe = "Tokenizer contract: keep two words.\nKeep this newline."
    encoded = tokenizer(probe, add_special_tokens=False).input_ids
    decoded = tokenizer.decode(
        encoded,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != probe:
        raise RuntimeError(
            "Tokenizer failed the whitespace round-trip contract for "
            f"{model_name or getattr(tokenizer, 'name_or_path', '<unknown>')}: "
            f"expected={probe!r}, decoded={decoded!r}. This would corrupt model "
            "inputs; load the checkpoint's serialized tokenizer.json directly."
        )


def load_tokenizer(model_name: str):
    """Load the model tokenizer without overriding its serialized pipeline."""
    load_kwargs = pretrained_load_kwargs(model_name)
    if model_family(model_name) == "deepseek":
        # DeepSeek R1/V3 tokenizer configs identify a byte-level BPE tokenizer
        # as LlamaTokenizerFast. Generic loading preserves tokenizer.json;
        # affected Transformers v5 Llama loaders replace it and remove spaces.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, **load_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    validate_tokenizer_roundtrip(tokenizer, model_name)
    return tokenizer


def load_causal_lm(model_name: str, device: str, dtype: str) -> LoadedModel:
    # Training-time activation extraction uses standard transformers rather
    # than vLLM because we need layer hooks.
    """Load a causal LM with transformers and return it with tokenizer/layers."""
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[str(dtype)]
    tokenizer = load_tokenizer(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, layers=resolve_transformer_layers(model))


def model_short_name(model_name: str | None) -> str:
    """Return the final path component of a model name or checkpoint path."""
    if not model_name:
        return ""
    return Path(str(model_name).rstrip("/")).name


def model_family(model_name: str | None) -> str:
    """Classify the model name into a small family label used for quirks."""
    short = model_short_name(model_name).lower()
    if "deepseek" in short:
        return "deepseek"
    if "gemma" in short:
        return "gemma"
    if "qwen" in short:
        return "qwen"
    return "other"


def pretrained_load_kwargs(model_name: str | None) -> dict:
    """Return Hugging Face loading kwargs required for the model family."""
    return {"trust_remote_code": True} if model_family(model_name) in {"deepseek", "gemma", "qwen"} else {}


def fallback_generation_settings(model_name: str | None) -> dict:
    # Used only when the model does not ship a usable GenerationConfig. These
    # defaults were chosen to match the reasoning-space style runs.
    """Return family-specific sampling defaults when no generation config exists."""
    family = model_family(model_name)
    if family == "deepseek":
        return {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "repetition_penalty": 1.0, "do_sample": True}
    if family == "qwen":
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.0, "do_sample": True}
    if family == "gemma":
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.0, "do_sample": True}
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05, "do_sample": True}


def load_generation_settings(model_name: str) -> dict:
    """Load the sampling defaults used by standard response generation."""
    fallback = fallback_generation_settings(model_name)
    load_kwargs = pretrained_load_kwargs(model_name)
    try:
        generation_config = GenerationConfig.from_pretrained(
            model_name, **load_kwargs
        )
    except OSError:
        generation_config = AutoConfig.from_pretrained(
            model_name, **load_kwargs
        )
    return {
        "temperature": (
            float(generation_config.temperature)
            if getattr(generation_config, "temperature", None) is not None
            else fallback["temperature"]
        ),
        "top_p": (
            float(generation_config.top_p)
            if getattr(generation_config, "top_p", None) is not None
            else fallback["top_p"]
        ),
        # DeepSeek's recommended top-p sampling does not apply an additional
        # top-k truncation. Some generic GenerationConfig defaults expose 50
        # here even when the model recipe does not use top-k.
        "top_k": (
            fallback["top_k"]
            if model_family(model_name) == "deepseek"
            else int(generation_config.top_k)
            if getattr(generation_config, "top_k", None) is not None
            else fallback["top_k"]
        ),
        "repetition_penalty": (
            float(generation_config.repetition_penalty)
            if getattr(generation_config, "repetition_penalty", None) is not None
            else fallback["repetition_penalty"]
        ),
        "do_sample": bool(
            getattr(generation_config, "do_sample", fallback["do_sample"])
        ),
    }


def resolve_generation_settings(
    model_name: str,
    overrides: dict | None = None,
) -> dict:
    """Apply optional run overrides to the model's standard sampling defaults."""
    resolved = load_generation_settings(model_name)
    for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
        value = (overrides or {}).get(key)
        if value is not None:
            resolved[key] = value
    if not bool(resolved["do_sample"]):
        resolved["temperature"] = 0.0
    return resolved


def reasoning_trace_markers(model_name: str | None) -> tuple[str, str]:
    # Trace extraction defaults to model-family reasoning delimiters. Callers
    # can override them from the CLI for new model formats.
    """Return start/end text markers used to identify reasoning tokens."""
    if model_family(model_name) == "gemma":
        return ("<|channel>thought\n", "<channel|>")
    return ("<think>", "</think>")


def apply_chat_template(tokenizer, prompt: str, *, enable_thinking: bool | None = None) -> str:
    # Prepared prompts are plain user text. Generation may wrap them in the
    # model's chat template so response token ids align with actual sampling.
    """Render a plain user prompt with the tokenizer chat template when available."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        chat_kwargs = {}
        if model_family(getattr(tokenizer, "name_or_path", None)) == "gemma":
            chat_kwargs["enable_thinking"] = True if enable_thinking is None else bool(enable_thinking)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
    return prompt

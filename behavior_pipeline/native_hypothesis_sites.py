from __future__ import annotations

"""Internal sites implementation for :mod:`behavior_pipeline.native_hypothesis`."""


import re
import unicodedata
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch.nn as nn

from .counterfactual_runtime import Runtime, token_texts


PATCH_SITE_ORDER = (
    "filler", "question_span", "newlines_and_punctuation",
    "question_and_filler", "question_and_newlines_and_punctuation",
)

PATCH_SITE_LABELS = {
    "filler": "Filler",
    "question_span": "Question span",
    "newlines_and_punctuation": "Newlines and punctuation",
    "question_and_filler": "Both",
    "question_and_newlines_and_punctuation": "Both",
}

# Shared presentation order; keep analysis condition identifiers unchanged.
PATCH_PLOT_ORDER = (
    "question_span", "filler", "newlines_and_punctuation",
    "question_and_filler", "question_and_newlines_and_punctuation",
)
PATCH_PLOT_LABELS = {
    "question_span": "Question",
    "filler": "Filler",
    "newlines_and_punctuation": "Filler",
    "question_and_filler": "Both",
    "question_and_newlines_and_punctuation": "Both",
}

_EXCLUDED_FILLER_DELIMITER_RE = re.compile(r"^<[^>]*(?:turn|channel|think)[^>]*>$", re.IGNORECASE)

_QUESTION_START = "Is the author in the wrong? Answer "

_QUESTION_END = "Answer nothing else."

_LINE_PREFIX_RE = re.compile(r"(?m)^(?:(?P<before>[ \t]*)\*(?P<after>[ \t]+)|(?P<indent>(?: {2,}|\t+)[ \t]*))")


def load_native_runtime(config: dict, *, device_override: str | None = None) -> Runtime:
    """Load the frozen base model for native causal tests without the learned editor."""
    from .model_utils import load_causal_lm, resolve_final_norm

    device = str(device_override or config["model"]["device"])
    loaded = load_causal_lm(
        str(config["model"]["name"]),
        device=device,
        dtype=str(config["model"]["dtype"]),
    )
    model = loaded.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer_module = loaded.layers[int(config["model"]["layer"])]
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise AttributeError(
            f"{type(model).__name__} does not expose output embeddings"
        )
    return Runtime(
        model=model,
        tokenizer=loaded.tokenizer,
        layer_module=layer_module,
        edit_module=nn.Identity(),
        checkpoint_config={},
        checkpoint_metrics={},
        device=device,
        final_norm=resolve_final_norm(model, loaded.layers),
        output_embeddings=output_embeddings,
    )


def canonical_direction_sign(direction: str) -> float:
    """Return +1 for adding the suggestion and -1 for removing it."""
    if str(direction) == "c_to_i":
        return 1.0
    if str(direction) == "i_to_c":
        return -1.0
    raise ValueError(f"Unknown counterfactual direction: {direction}")


def _display_token(text: str) -> str:
    """Preserve spaces/tabs and mark real line breaks with ``↵``."""
    text = str(text).replace("▁", " ").replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "↵\n")


def _is_newline_token(text: str) -> bool:
    value = str(text)
    return "\n" in value and value.strip() == ""


def _is_newline_or_punctuation_token(text: str) -> bool:
    value = str(text)
    return bool(value) and all(
        char.isspace() or unicodedata.category(char).startswith("P")
        for char in value
    ) and any(
        char in "\r\n" or unicodedata.category(char).startswith("P")
        for char in value
    )


def _question_text(prompt_text: str) -> str | None:
    """Extract the standardized Scruples answer question from a rendered prompt."""
    text = str(prompt_text).replace("▁", " ")
    start = text.rfind(_QUESTION_START)
    if start < 0:
        return None
    end = text.find(_QUESTION_END, start)
    return text[start:end + len(_QUESTION_END)] if end >= 0 else None


def _token_span_for_text(token_text: list[str], needle: str) -> tuple[int, int] | None:
    """Map a literal decoded substring back to an inclusive token span.

    Individual-token decoding is used because it avoids depending on tokenizer
    offset mappings, which are not uniformly available for the model families
    supported by this repository.
    """
    if not needle:
        return None
    token_text = [str(value).replace("▁", " ") for value in token_text]
    needle = str(needle).replace("▁", " ")
    joined = "".join(str(value) for value in token_text)
    start_char = joined.rfind(str(needle))
    if start_char < 0:
        return None
    end_char = start_char + len(str(needle))
    cursor = 0
    first = None
    last = None
    for index, value in enumerate(token_text):
        next_cursor = cursor + len(str(value))
        if first is None and next_cursor > start_char:
            first = index
        if cursor < end_char:
            last = index
        cursor = next_cursor
        if cursor >= end_char:
            break
    if first is None or last is None:
        return None
    return int(first), int(last)


def _terminal_newline_index(prompt_token_text: list[str]) -> int | None:
    for index in range(len(prompt_token_text) - 1, -1, -1):
        if _is_newline_token(prompt_token_text[index]):
            return int(index)
    return None


def _line_prefix_token_indices(token_text: list[str]) -> list[int]:
    """Return whitespace tokens that form structural response-line prefixes.

    For a bullet-formatted line we intentionally select only the spaces/tabs
    immediately before and after the leading ``*``; the bullet character itself
    is *not* part of the filler site. For non-bulleted indented
    lines, leading indentation of at least two spaces (or one tab) is selected.
    Newline tokens themselves are handled separately.

    Patching is token-level: only nonempty, entirely whitespace tokens qualify.
    Tokens combining whitespace with words, bullets, or other non-whitespace
    characters are excluded even when they overlap a prefix whitespace range.
    """
    joined = "".join(str(value) for value in token_text)
    if not joined:
        return []
    char_ranges: list[tuple[int, int]] = []
    for match in _LINE_PREFIX_RE.finditer(joined):
        before = match.span("before") if match.group("before") is not None else (-1, -1)
        after = match.span("after") if match.group("after") is not None else (-1, -1)
        indent = match.span("indent") if match.group("indent") is not None else (-1, -1)
        for start, end in (before, after, indent):
            if start >= 0 and end > start:
                char_ranges.append((int(start), int(end)))
    if not char_ranges:
        return []
    token_ranges: list[tuple[int, int]] = []
    cursor = 0
    for value in token_text:
        next_cursor = cursor + len(str(value))
        token_ranges.append((cursor, next_cursor))
        cursor = next_cursor
    selected: list[int] = []
    for index, (token_start, token_end) in enumerate(token_ranges):
        if not str(token_text[index]).isspace():
            continue
        if any(token_end > start and token_start < end for start, end in char_ranges):
            selected.append(int(index))
    return selected


def detect_patch_sites(
    record: dict,
    tokenizer,
) -> dict[str, dict]:
    """Locate source/target patch sites without assuming equal prompt lengths."""
    source_prompt_ids = record["prompt_ids"].detach().long().cpu()
    target_prompt_ids = record["target_prompt_ids"].detach().long().cpu()
    response_ids = record["response_ids"].detach().long().cpu()
    # One-character replacement preserves every token and character offset.
    source_prompt_texts = [text.replace("▁", " ") for text in token_texts(tokenizer, source_prompt_ids)]
    target_prompt_texts = [text.replace("▁", " ") for text in token_texts(tokenizer, target_prompt_ids)]
    response_texts = [text.replace("▁", " ") for text in token_texts(tokenizer, response_ids)]

    source_prompt_len = int(source_prompt_ids.numel())
    target_prompt_len = int(target_prompt_ids.numel())

    source_terminal = _terminal_newline_index(source_prompt_texts)
    target_terminal = _terminal_newline_index(target_prompt_texts)

    response_relative = []
    for response_index, text in enumerate(response_texts):
        if _is_newline_token(text):
            response_relative.append(int(response_index))
    line_prefix_relative = [
        index for index in _line_prefix_token_indices(response_texts)
        if not _EXCLUDED_FILLER_DELIMITER_RE.fullmatch(response_texts[index].strip())
    ]
    response_relative = sorted(set(response_relative) | set(line_prefix_relative))

    response_source = [source_prompt_len + index for index in response_relative]
    response_target = [target_prompt_len + index for index in response_relative]

    question = _question_text(str(record.get("source_prompt_text", "")))
    source_question = (
        _token_span_for_text(source_prompt_texts, question) if question else None
    )
    target_question = (
        _token_span_for_text(target_prompt_texts, question) if question else None
    )
    source_question_indices = (
        list(range(source_question[0], source_question[1] + 1))
        if source_question is not None
        else []
    )
    target_question_indices = (
        list(range(target_question[0], target_question[1] + 1))
        if target_question is not None
        else []
    )
    if len(source_question_indices) != len(target_question_indices):
        # A mismatched exact span is not safe to patch positionwise.  Keep the
        # failure visible in metadata rather than heuristically stretching it.
        source_question_indices = []
        target_question_indices = []

    all_source = response_source
    all_target = response_target

    # Preserve source↔target alignment when constructing the union.  Absolute
    # prompt indices differ between the two conditions, so deduplicating source
    # and target positions independently could destroy the pairing used by
    # native activation patching.
    combined_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for pair in list(zip(all_source, all_target)) + list(
        zip(source_question_indices, target_question_indices)
    ):
        pair = (int(pair[0]), int(pair[1]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        combined_pairs.append(pair)
    combined_source = [source for source, _target in combined_pairs]
    combined_target = [target for _source, target in combined_pairs]

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    punctuation_relative = [
        index for index, text in enumerate(response_texts)
        if int(response_ids[index]) not in special_ids
        and _is_newline_or_punctuation_token(text)
    ]
    punctuation_pairs = [
        (source_prompt_len + index, target_prompt_len + index)
        for index in punctuation_relative
    ]
    if source_terminal is not None and target_terminal is not None:
        punctuation_pairs.insert(0, (source_terminal, target_terminal))
    question_punctuation_pairs = list(dict.fromkeys(
        list(zip(source_question_indices, target_question_indices)) + punctuation_pairs
    ))

    return {
        "newlines_and_punctuation": {
            "source_indices": [source for source, _ in punctuation_pairs],
            "target_indices": [target for _, target in punctuation_pairs],
            "response_relative_indices": punctuation_relative,
            "definition": "terminal prompt newline plus response tokens containing only Unicode punctuation/whitespace, with at least one punctuation character or newline; excludes special tokens and plain spaces/tabs",
        },
        "question_and_newlines_and_punctuation": {
            "source_indices": [source for source, _ in question_punctuation_pairs],
            "target_indices": [target for _, target in question_punctuation_pairs],
            "response_relative_indices": punctuation_relative,
            "question_text": question,
            "definition": "aligned union of the full question span and newlines-and-punctuation sites",
        },
        "filler": {
            "source_indices": all_source,
            "target_indices": all_target,
            "response_relative_indices": response_relative,
            "definition": "response-only whitespace/newline tokens and leading line-prefix whitespace around bullets/indentation (excluding the bullet token itself); excludes prompt positions and special delimiters",
        },
        "question_span": {
            "source_indices": source_question_indices,
            "target_indices": target_question_indices,
            "response_relative_indices": [],
            "question_text": question,
        },
        "question_and_filler": {
            "source_indices": combined_source,
            "target_indices": combined_target,
            "response_relative_indices": response_relative,
            "question_text": question,
            "definition": "aligned union of the full question span and response filler positions",
        },
        "detected": {
            "source_prompt_tokens": source_prompt_len,
            "target_prompt_tokens": target_prompt_len,
            "response_tokens": int(response_ids.numel()),
            "response_filler_relative_indices": response_relative,
            "response_filler_text": [
                _display_token(response_texts[index]) for index in response_relative
            ],
            "response_line_prefix_relative_indices": line_prefix_relative,
            "response_line_prefix_text": [
                _display_token(response_texts[index]) for index in line_prefix_relative
            ],
            "source_terminal_delimiter_index": source_terminal,
            "target_terminal_delimiter_index": target_terminal,
            "source_question_span": list(source_question) if source_question else None,
            "target_question_span": list(target_question) if target_question else None,
        },
    }


def select_native_record_summaries(
    summaries: list[dict],
    *,
    directions: Iterable[str],
    one_per_source_direction: bool,
    maximum: int,
) -> list[dict]:
    allowed = {str(value) for value in directions}
    rows = [row for row in summaries if str(row.get("direction")) in allowed]
    if bool(one_per_source_direction):
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(str(row.get("source_id")), str(row.get("direction")))].append(row)
        selected = []
        for key in sorted(grouped):
            direction = key[1]
            preferred_arm = "control" if direction == "c_to_i" else "intervention"

            def rank(row: dict):
                response_id = str(row.get("response_id", ""))
                preferred = preferred_arm in response_id.lower()
                return (0 if preferred else 1, response_id, str(row.get("record_id", "")))

            selected.append(sorted(grouped[key], key=rank)[0])
        rows = selected
    else:
        rows = sorted(
            rows,
            key=lambda row: (
                str(row.get("source_id", "")),
                str(row.get("direction", "")),
                str(row.get("response_id", "")),
            ),
        )
    if int(maximum) > 0:
        rows = rows[: int(maximum)]
    return rows


def partition_summaries_by_source(
    summaries: list[dict], shard_count: int
) -> list[list[dict]]:
    """Keep both directions together while balancing estimated token work."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        groups[str(row.get("source_id", ""))].append(row)
    shards = [[] for _ in range(max(1, int(shard_count)))]
    loads = [0 for _ in shards]
    weighted_sources = []
    for source_id, rows in groups.items():
        cost = sum(
            max(
                1,
                int(row.get("prompt_tokens", 0))
                + int(row.get("response_tokens", 0)),
            )
            for row in rows
        )
        weighted_sources.append((int(cost), source_id))
    for cost, source_id in sorted(
        weighted_sources, key=lambda item: (-item[0], item[1])
    ):
        destination = min(
            range(len(shards)), key=lambda index: (loads[index], index)
        )
        shards[destination].extend(groups[source_id])
        loads[destination] += int(cost)
    return shards


def _site_metric_aggregate(
    rows: list[dict],
    *,
    site: str,
    variant_key: str,
    full_key: str | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Aggregate KL-based site metrics with source-clustered bootstrap intervals."""
    selected = [row for row in rows if site in row.get("sites", {})]
    if not selected:
        return {"records": 0, "sources": 0}

    def metric(sample_rows: list[dict]) -> float:
        source = float(np.mean([float(row["source_kl_mean"]) for row in sample_rows]))
        variant = float(np.mean([float(row["sites"][site][variant_key]) for row in sample_rows]))
        if source <= 1e-12:
            return 0.0
        if full_key is None:
            return float(1.0 - variant / source)
        full = float(np.mean([float(row[full_key]) for row in sample_rows]))
        return float((variant - full) / source)

    value = metric(selected)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_source[str(row.get("source_id", ""))].append(row)
    source_ids = sorted(by_source)
    bootstrap = []
    if source_ids and int(bootstrap_samples) > 0:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(bootstrap_samples)):
            sampled_ids = rng.choice(source_ids, size=len(source_ids), replace=True)
            sample_rows = []
            for source_id in sampled_ids:
                sample_rows.extend(by_source[str(source_id)])
            bootstrap.append(metric(sample_rows))
    return {
        "records": len(selected),
        "sources": len(source_ids),
        "value": value,
        "bootstrap_95_ci": (
            [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
            if bootstrap
            else None
        ),
    }

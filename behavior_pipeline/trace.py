from __future__ import annotations

"""Response trace extraction and strict final-answer parsing."""

import re
from dataclasses import dataclass

from .token_text import decode_token_ids


_REASONING_MARKERS = re.compile(r"<think>|</think>|<\|channel>thought\b|<channel\|>")
_REASONING_END = {"<think>": "</think>", "<|channel>thought": "<channel|>"}
_END_MARKERS = re.compile(
    r"(?:\s*(?:<turn\|>|<\|im_end\|>|<\|endoftext\|>|<eos>|</s>|"
    r"<｜end of sentence｜>|<\|end of sentence\|>))+\s*$"
)


@dataclass(frozen=True)
class AnswerParse:
    status: str
    letter: str | None = None
    start: int | None = None
    end: int | None = None
    format: str | None = None


def parse_choice(text: str, valid: set[str] | None = None, *,
                 prompt_text: str = "", finish_reason: str | None = None,
                 truncated: bool = False) -> AnswerParse:
    """Parse once, retaining answer-character offsets in the original text.

    An exact single-letter line takes priority over explicit labeled answers.
    Recognized unfinished reasoning and known truncation are unparsed. Only
    uppercase answer letters are accepted; arbitrary prose is never searched.
    """
    if truncated or str(finish_reason).lower() in {"length", "max_tokens", "max_length", "abort", "aborted", "cancelled", "canceled"}:
        return AnswerParse("truncated")
    text = str(text or "").replace("▁", " ")
    pending = None
    for marker in _REASONING_MARKERS.finditer(str(prompt_text).replace("▁", " ")):
        value = marker.group()
        if value in _REASONING_END:
            pending = _REASONING_END[value]
        elif value == pending:
            pending = None
    start = 0
    for marker in _REASONING_MARKERS.finditer(text):
        value = marker.group()
        if value in _REASONING_END:
            pending = _REASONING_END[value]
        elif pending is None or value == pending:
            pending = None
            # Also handles an opener supplied by a chat template when prompt
            # context is unavailable: a closing marker ends preceding thought.
            start = marker.end()
    if pending is not None:
        return AnswerParse("unfinished_reasoning")
    ending = _END_MARKERS.search(text, start)
    end = ending.start() if ending else len(text)
    # Search a slice but translate every captured letter back to original offsets.
    region = text[start:end]
    letters = re.escape("".join(sorted(valid or {"A", "B", "C", "D"})))
    patterns = [
        ("single_letter", rf"(?m)^[ \t\r]*(?:\(|\[)?[ \t]*(?P<answer>[{letters}])[ \t]*(?:\)|\])?[ \t\r]*$"),
        ("final_answer", rf"(?i:\bfinal[ \t]+answer\b)[ \t]*(?::|=|-|(?i:is\b))?[ \t]*(?:\(|\[)?(?P<answer>[{letters}])\b"),
        ("labeled_answer", rf"(?m)^[ \t]*(?i:answer|choice)[ \t]*[:=\-][ \t]*(?:\(|\[)?(?P<answer>[{letters}])\b"),
    ]
    for kind, pattern in patterns:
        matches = list(re.finditer(pattern, region))
        if matches:
            match = matches[-1]
            left, right = match.span("answer")
            return AnswerParse("parsed", match.group("answer"), start + left, start + right, kind)
    return AnswerParse("unparsed")


def find_trace_indices(
    tokenizer,
    prompt_ids: list[int],
    response_ids: list[int],
    think_start: str | None = "<think>",
    think_end: str | None = "</think>",
) -> list[int]:
    # Return absolute token positions inside full_text/input_ids. If reasoning
    # delimiters are missing, fall back to the whole response so training still
    # has a trace.
    """Find trace indices."""
    offset = len(prompt_ids)
    if not response_ids:
        return []
    response_text = decode_token_ids(tokenizer, response_ids)
    if not think_start or not think_end:
        return list(range(offset, offset + len(response_ids)))
    match = re.search(re.escape(think_start) + r"(.*?)" + re.escape(think_end), response_text, flags=re.DOTALL)
    if match is None:
        return list(range(offset, offset + len(response_ids)))
    prefix = response_text[: match.start(1)]
    trace = match.group(1)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    trace_ids = tokenizer(trace, add_special_tokens=False).input_ids
    start = offset + min(len(prefix_ids), len(response_ids))
    end = min(offset + len(response_ids), start + len(trace_ids))
    return list(range(start, end))


def extract_choice_letter(text: str, valid: set[str] | None = None, **kwargs) -> str | None:
    """Return the answer letter found by the shared span-aware parser."""
    return parse_choice(text, valid, **kwargs).letter

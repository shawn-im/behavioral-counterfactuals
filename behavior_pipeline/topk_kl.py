from __future__ import annotations

"""Shared utilities for conditional top-k KL edit distillation."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .tasks import FINAL_QUESTION_PREFIX
from .token_text import token_id_texts
from .numerics import reject_invalid, validate_kl


LOG_PROB_MIN = -80.0
MASKED_LOGIT = -1.0e9
ALL_TOKENS_POSITION_POLICY = "all_tokens"
FINAL_QUESTION_RESPONSE_POSITION_POLICY = "final_question_and_response_tokens"


class TopKKLDistillDataset(Dataset):
    """Load cached top-k KL rows for one explicit train, validation, or test split."""

    def __init__(
        self,
        cache_dir: Path,
        split: str,
    ):
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.meta = json.loads((self.cache_dir / "metadata.json").read_text(encoding="utf-8"))
        if (
            self.meta.get("cache_format") != "topk_kl_distill"
        ):
            raise ValueError(
                "top-k KL analysis requires a topk_kl_distill cache; "
                "rebuild the cache from saved generation token IDs"
            )
        if self.meta.get("token_id_policy") != "saved_generation_ids_required":
            raise ValueError(
                "top-k KL caches must require saved generation token IDs; "
                "rebuild this cache with the current cache builder"
            )
        split_names = set(self.meta.get("splits", {}))
        if split_names != {"train", "val", "test"}:
            raise ValueError(
                "top-k KL caches must contain exactly train, val, and test splits; "
                f"found {sorted(split_names)}"
            )
        if "splits" not in self.meta or self.split not in self.meta["splits"]:
            raise ValueError(f"cache metadata has no split {self.split!r}")
        rows = []
        for shard in self.meta["splits"][self.split].get("shards", []):
            obj = torch.load(
                self.cache_dir / shard["path"], map_location="cpu"
            )
            for row in obj["rows"]:
                tagged = dict(row)
                tagged.setdefault("cache_source_split", self.split)
                rows.append(tagged)
        self.source_splits = (self.split,)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[int(idx)]


def sanitize_logp(x: torch.Tensor, *, device=None) -> torch.Tensor:
    """Return finite fp32 log-probs in [log(eps), 0]."""
    if device is not None:
        x = x.to(device=device)
    x = x.float()
    reject_invalid(x, torch.isnan(x) | torch.isposinf(x), "log probabilities")
    return x.clamp(min=LOG_PROB_MIN, max=0.0)


def masked_logsumexp(logp: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logp.masked_fill(~support_mask, MASKED_LOGIT), dim=-1, keepdim=True)


def conditional_topk_kl_raw(
    target_logp: torch.Tensor,
    student_logp: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward KL after renormalizing target/student distributions over support."""
    device = student_logp.device
    support_mask = support_mask.to(device=device, dtype=torch.bool)
    target_logp = sanitize_logp(target_logp.to(device).masked_fill(~support_mask, 0.0))
    student_logp = sanitize_logp(student_logp.masked_fill(~support_mask, 0.0))
    valid_per_token = support_mask.any(dim=-1)
    if not bool(valid_per_token.all()):
        support_mask = support_mask.clone()
        support_mask[~valid_per_token, 0] = True
        target_logp = target_logp.clone()
        student_logp = student_logp.clone()
        target_logp[~valid_per_token, 0] = 0.0
        student_logp[~valid_per_token, 0] = 0.0
    target_cond_logp = target_logp - masked_logsumexp(target_logp, support_mask)
    student_cond_logp = student_logp - masked_logsumexp(student_logp, support_mask)
    p_cond = target_cond_logp.exp().masked_fill(~support_mask, 0.0)
    kl_elem = p_cond * (target_cond_logp - student_cond_logp)
    return validate_kl(kl_elem.masked_fill(~support_mask, 0.0).sum(dim=-1).float())


def conditional_topk_kl(
    target_logp: torch.Tensor,
    student_logp: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Numerically guarded nonnegative conditional top-k KL."""
    token_kl = conditional_topk_kl_raw(target_logp, student_logp, support_mask)
    return token_kl.clamp_min(0.0)


def prompt_ids_for_direction(row: dict, direction: str) -> torch.Tensor:
    return row["control_prompt_ids"] if direction == "c_to_i" else row["intervention_prompt_ids"]


def final_question_start_index(tokenizer, prompt_ids: torch.Tensor) -> int:
    """Locate the token containing the final standardized Scruples question.

    The returned token is included in the edit region. It can also contain
    leading whitespace when the tokenizer joins whitespace and ``Is`` into one
    token.
    """
    ids = [int(value) for value in prompt_ids.detach().cpu().tolist()]
    pieces = token_id_texts(tokenizer, ids)
    if len(pieces) != len(ids):
        raise RuntimeError("tokenizer returned the wrong number of prompt pieces")
    text = "".join(pieces)
    decode_kwargs = dict(
        skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    decoded = tokenizer.decode(ids, **decode_kwargs)
    start_char = text.rfind(FINAL_QUESTION_PREFIX)
    if decoded != text and FINAL_QUESTION_PREFIX in decoded:
        # SentencePiece-style decoders can strip the leading space from each
        # independently decoded token. Locate the boundary in a full decode
        # instead, then search decoded prefixes to preserve the saved token IDs.
        text = decoded
        start_char = text.rfind(FINAL_QUESTION_PREFIX)
        low, high = 1, len(ids)
        while low < high:
            middle = (low + high) // 2
            prefix = tokenizer.decode(ids[:middle], **decode_kwargs)
            if len(prefix) > start_char:
                high = middle
            else:
                low = middle + 1
        return low - 1
    if start_char < 0:
        raise RuntimeError(
            "could not locate the final Scruples question in the tokenized prompt "
            "using either token pieces or full-prompt decoding"
        )
    cursor = 0
    for index, piece in enumerate(pieces):
        next_cursor = cursor + len(piece)
        if next_cursor > start_char:
            return int(index)
        cursor = next_cursor
    raise RuntimeError("final question character offset has no corresponding token")


def target_logps_for_direction(row: dict, direction: str) -> tuple[torch.Tensor, torch.Tensor]:
    if direction == "c_to_i":
        return row["logp_intervention"].float(), row["logp_intervention_other"].float()
    return row["logp_control"].float(), row["logp_control_other"].float()


def source_logps_for_direction(row: dict, direction: str) -> tuple[torch.Tensor, torch.Tensor]:
    if direction == "c_to_i":
        return row["logp_control"].float(), row["logp_control_other"].float()
    return row["logp_intervention"].float(), row["logp_intervention_other"].float()


def cached_no_edit_evaluation(rows: Sequence[dict], *, directions: str) -> dict:
    """Evaluate the exact zero-edit baseline directly from teacher-cache rows."""
    direction_list = (
        ["c_to_i", "i_to_c"]
        if str(directions) == "both"
        else [str(directions)]
    )
    evaluation_rows = []
    for row in rows:
        support_mask = row["support_mask"].to(dtype=torch.bool)
        for direction in direction_list:
            target_logp, _target_other = target_logps_for_direction(
                row, direction
            )
            source_logp, _source_other = source_logps_for_direction(
                row, direction
            )
            token_kl = conditional_topk_kl(
                target_logp, source_logp, support_mask
            )
            support_mass = (
                sanitize_logp(source_logp.masked_fill(~support_mask, 0.0))
                .exp()
                .masked_fill(~support_mask, 0.0)
                .sum(dim=-1)
                .clamp(min=0.0, max=1.0)
            )
            mean_loss = float(mean_token_kl(token_kl))
            evaluation_rows.append(
                {
                    "source_id": str(row.get("source_id", "")),
                    "response_id": str(row.get("response_id", "")),
                    "direction": direction,
                    "loss": mean_loss,
                    "source_loss": mean_loss,
                    "student_support_mass": support_mass,
                }
            )
    return aggregate_evaluation_rows(evaluation_rows)


def conditional_logp_from_support_logits(
    support_logits: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Return log-probabilities normalized only over a fixed cached support.

    This is exactly the quantity needed by ``conditional_topk_kl_raw``.  It
    avoids constructing a full-vocabulary softmax when the logits have already
    been projected onto the source/target union support.
    """
    logits_f = support_logits.float()
    support_mask = support_mask.to(device=logits_f.device, dtype=torch.bool)
    valid_per_token = support_mask.any(dim=-1)
    if not bool(valid_per_token.all()):
        support_mask = support_mask.clone()
        support_mask[~valid_per_token, 0] = True
    masked = logits_f.masked_fill(~support_mask, float("-inf"))
    reject_invalid(masked, torch.isnan(masked) | torch.isposinf(masked), "support logits")
    normalizer = torch.logsumexp(masked, dim=-1, keepdim=True)
    reject_invalid(normalizer, ~torch.isfinite(normalizer), "support log normalizer")
    logp = masked - normalizer
    return sanitize_logp(logp.masked_fill(~support_mask, 0.0))


def logp_on_support_from_logits(
    logits: torch.Tensor,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather full-softmax log-probs on cached support and return support mass."""
    logits_f = logits.float()
    reject_invalid(logits_f, torch.isnan(logits_f) | torch.isposinf(logits_f), "full-vocabulary logits")
    support_ids = support_ids.to(device=logits_f.device, dtype=torch.long)
    support_mask = support_mask.to(device=logits_f.device, dtype=torch.bool)
    safe_ids = support_ids.masked_fill(~support_mask, 0)
    log_z = torch.logsumexp(logits_f, dim=-1, keepdim=True)
    reject_invalid(log_z, ~torch.isfinite(log_z), "full-vocabulary log normalizer")
    logq_support = logits_f.gather(-1, safe_ids) - log_z
    support_mass = logq_support.exp().masked_fill(~support_mask, 0.0).sum(dim=-1)
    reject_invalid(support_mass, ~torch.isfinite(support_mass), "support mass")
    support_mass = support_mass.float().clamp(min=0.0, max=1.0)
    return logq_support, support_mass


def recovery_fraction(source_kl: float, edit_kl: float, *, eps: float = 1e-12) -> float:
    """Fractional KL recovery; zero when the source KL baseline is effectively zero."""
    source = float(source_kl)
    if source <= float(eps):
        return 0.0
    return float(1.0 - float(edit_kl) / source)


def mean_token_kl(token_kl: torch.Tensor) -> torch.Tensor:
    """Return one sequence-direction KL mean with explicit fp32 accumulation."""
    if token_kl.numel() == 0:
        raise ValueError("cannot average an empty token KL tensor")
    return token_kl.float().mean()


def normalize_sequence_gradients(parameters, sequence_count: int) -> None:
    """Normalize accumulated sequence-loss sums by their exact record count."""
    if int(sequence_count) <= 0:
        raise ValueError("sequence_count must be positive")
    scale = float(sequence_count)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(scale)


def aggregate_sequence_kl(
    source_kl_means: Sequence[float],
    edited_kl_means: Sequence[float],
) -> dict:
    """Aggregate equally weighted sequence-direction KL means."""
    if len(source_kl_means) != len(edited_kl_means):
        raise ValueError("source and edited KL sequences must have equal length")
    if not source_kl_means:
        raise ValueError("cannot aggregate zero sequence-direction records")
    for name, values in (("source KL", source_kl_means), ("edited KL", edited_kl_means)):
        for index, value in enumerate(values):
            if not np.isfinite(value):
                raise RuntimeError(f"{name}: non-finite value {value} at sequence {index}")
    source = float(np.mean(np.asarray(source_kl_means, dtype=np.float64)))
    edited = float(np.mean(np.asarray(edited_kl_means, dtype=np.float64)))
    return {
        "source_kl_mean": source,
        "edited_kl_mean": edited,
        "recovery": recovery_fraction(source, edited),
    }


def mean_recovered_utility(
    source_token_kl: torch.Tensor, edited_token_kl: torch.Tensor
) -> torch.Tensor:
    """Mean conditional KL reduction over one aligned response."""
    if source_token_kl.shape != edited_token_kl.shape:
        raise ValueError("source and edited token KL arrays must align")
    return mean_token_kl(source_token_kl.float() - edited_token_kl.float())


def mean_kl_increase(
    full_token_kl: torch.Tensor, ablated_token_kl: torch.Tensor
) -> torch.Tensor:
    """Mean conditional KL increase caused by one ablation."""
    if full_token_kl.shape != ablated_token_kl.shape:
        raise ValueError("full and ablated token KL arrays must align")
    return mean_token_kl(ablated_token_kl.float() - full_token_kl.float())


def aggregate_evaluation_rows(rows: Sequence[dict]) -> dict:
    """Aggregate sequence means and a true corpus support-mass quantile."""
    if not rows:
        return {}
    for index, row in enumerate(rows):
        context = (f"sequence {index} source_id={row.get('source_id')} "
                   f"response_id={row.get('response_id')} direction={row.get('direction')}")
        for key in ("loss", "source_loss"):
            if not np.isfinite(float(row[key])):
                raise RuntimeError(f"{context}: non-finite {key}={row[key]}")
        mass = row["student_support_mass"]
        reject_invalid(mass, ~torch.isfinite(mass), f"{context}: support mass")
    edited = [float(row["loss"]) for row in rows]
    source = [float(row["source_loss"]) for row in rows]
    metrics = aggregate_sequence_kl(source, edited)
    masses = torch.cat(
        [
            row["student_support_mass"].detach().float().cpu().reshape(-1)
            for row in rows
        ]
    )
    support_summary = summarize_support_masses(masses)
    return {
        "loss": metrics["edited_kl_mean"],
        **metrics,
        "support_mass_p10": support_summary["p10"],
        "support_mass_min": support_summary["minimum"],
        "support_mass_histogram": support_summary["histogram"],
        "_loss_sum": float(sum(edited)),
        "_source_loss_sum": float(sum(source)),
        "_sequence_count": len(edited),
        "_student_support_mass": masses,
        "_sequence_rows": [
            {
                "source_id": str(row.get("source_id", "")),
                "response_id": str(row.get("response_id", "")),
                "direction": str(row.get("direction", "")),
                "source_kl_mean": float(row["source_loss"]),
                "edited_kl_mean": float(row["loss"]),
            }
            for row in rows
        ],
    }


def summarize_support_masses(masses: torch.Tensor, *, bins: int = 50) -> dict:
    """Return exact tail statistics and a compact fixed-bin histogram."""
    values = masses.detach().float().cpu().reshape(-1)
    values = values[torch.isfinite(values)].clamp(0.0, 1.0)
    edges = torch.linspace(0.0, 1.0, int(bins) + 1, dtype=torch.float64)
    counts = (
        torch.histc(values, bins=int(bins), min=0.0, max=1.0).to(torch.int64)
        if values.numel()
        else torch.zeros(int(bins), dtype=torch.int64)
    )
    return {
        "p10": float(torch.quantile(values, 0.1)) if values.numel() else None,
        "minimum": float(values.min()) if values.numel() else None,
        "histogram": {
            "positions": int(values.numel()),
            "counts": counts.tolist(),
            "edges": edges.tolist(),
        },
    }

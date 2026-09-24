from __future__ import annotations

"""Internal figures implementation for :mod:`behavior_pipeline.counterfactual_report`."""

import json
import math
from pathlib import Path

import numpy as np

from .counterfactual_runtime import (
    read_json,
)
from .io import read_jsonl
from .trace import extract_choice_letter
from .counterfactual_report_html import _method_label
from .counterfactual_report_plot_style import (
    ATTRIBUTION_METHOD_COLORS,
    OKABE_ITO_COLORS,
    PLOT_COLORS,
    PLOT_STYLE,
    SCATTER_COLORS,
    apply_report_plot_style,
)
from .statistics import bootstrap_mean_confidence_interval

BAR_TITLE_PAD = 12


def _paper_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
        apply_report_plot_style(matplotlib, plt)
        return plt, PercentFormatter
    except Exception:
        return None, None


def _annotate_bar_values(
    ax,
    bars,
    *,
    percent: bool = False,
    digits: int = 3,
    fontsize: float = 11.5,
) -> None:
    """Place the plotted value at the edge of every finite bar."""
    labels = []
    for patch in bars:
        value = float(patch.get_height())
        if not math.isfinite(value):
            labels.append("")
        elif percent:
            labels.append(f"{value:.1%}")
        else:
            labels.append(f"{value:.{digits}g}")
    try:
        ax.bar_label(bars, labels=labels, padding=3, fontsize=fontsize)
    except AttributeError:
        # Matplotlib < 3.4 fallback.
        for patch, label in zip(bars, labels):
            if not label:
                continue
            value = float(patch.get_height())
            ax.annotate(
                label,
                (patch.get_x() + patch.get_width() / 2.0, value),
                xytext=(0, 3 if value >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=fontsize,
            )


def _add_y_zero_line(ax) -> None:
    """Draw the common black y=0 reference when zero is in view."""
    low, high = ax.get_ylim()
    if low <= 0.0 <= high:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5, zorder=3)


def _add_x_zero_line(ax) -> None:
    """Draw the common black x=0 reference when zero is in view."""
    low, high = ax.get_xlim()
    if low <= 0.0 <= high:
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5, zorder=3)


def _asymmetric_errors(values: list[float], intervals: list) -> np.ndarray:
    """Convert absolute confidence intervals to Matplotlib asymmetric errors."""
    lower, upper = [], []
    for value, interval in zip(values, intervals):
        if (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
            and interval[0] is not None
            and interval[1] is not None
        ):
            try:
                lo, hi = float(interval[0]), float(interval[1])
            except (TypeError, ValueError):
                lo = hi = float(value)
            lower.append(max(0.0, float(value) - lo))
            upper.append(max(0.0, hi - float(value)))
        else:
            lower.append(float("nan"))
            upper.append(float("nan"))
    return np.asarray([lower, upper], dtype=np.float64)


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    n = int(total)
    if n <= 0:
        return None
    p = min(1.0, max(0.0, float(successes) / n))
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _write_recovery_figure(methods: list[dict], path: Path) -> Path | None:
    if not methods:
        return None
    plt, PercentFormatter = _paper_matplotlib()
    if plt is None:
        return None
    rows = [row for row in methods if row.get("test_recovery") is not None]
    if not rows:
        return None
    labels = [
        _method_label(str(row.get("method", ""))).replace(
            "Fixed vector", "Fixed\nvector"
        )
        for row in rows
    ]
    values = [float(row.get("test_recovery", 0.0)) for row in rows]
    intervals = [row.get("recovery_bootstrap_95_ci") for row in rows]
    upper_values = [
        float(interval[1])
        if isinstance(interval, (list, tuple)) and len(interval) == 2 and interval[1] is not None
        else value
        for value, interval in zip(values, intervals)
    ]
    ymax = max(0.60, math.ceil(max(max(upper_values), 0.0) * 20.0) / 20.0 + 0.05)
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    bars = ax.bar(
        np.arange(len(rows)), values,
        yerr=_asymmetric_errors(values, intervals), capsize=3,
        color=[
            OKABE_ITO_COLORS[("blue", "orange", "green")[index % 3]]
            for index in range(len(rows))
        ],
    )
    _annotate_bar_values(ax, bars, percent=True, fontsize=10)
    ax.set_xticks(np.arange(len(rows)), labels)
    ax.set_ylim(0.0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Recovery", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title("Recovery across Architectures", fontsize=12, pad=BAR_TITLE_PAD)
    _add_y_zero_line(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path


def _activation_recovery_value(row: dict) -> float:
    """Use saved recovery, or reconstruct it only with a valid source baseline."""
    saved = row.get("test_activation_recovery")
    if saved is not None and math.isfinite(float(saved)):
        return float(saved)
    source = row.get("test_activation_source_nmse")
    edited = row.get("test_activation_nmse")
    if source is not None and edited is not None:
        source, edited = float(source), float(edited)
        if math.isfinite(source) and source > 0 and math.isfinite(edited):
            return 1.0 - edited / source
    return float("nan")


def _objective_rows(comparison: dict) -> tuple[list[dict], list[str], list[float], list[float]]:
    rows = list(comparison.get("objectives", []))
    labels = [
        "Behavior\nreconstruction" if str(row.get("objective")) == "behavior_kl" else "Activation\nreconstruction"
        for row in rows
    ]
    behavior = [float(row.get("test_behavior_recovery", float("nan"))) for row in rows]
    activation = [float(row.get("test_activation_nmse", float("nan"))) for row in rows]
    return rows, labels, behavior, activation


def _write_objective_comparison_figures(
    comparison: dict,
    *,
    behavior_path: Path,
    activation_path: Path,
) -> tuple[Path | None, Path | None]:
    """Write separate behavior-recovery and activation-NMSE objective figures."""
    rows, labels, behavior, activation = _objective_rows(comparison)
    recovery = [_activation_recovery_value(row) for row in rows]
    has_recovery = bool(rows) and all(math.isfinite(value) for value in recovery)
    if has_recovery:
        activation = recovery
    if len(rows) < 2 or not all(math.isfinite(value) for value in behavior + activation):
        return None, None
    plt, PercentFormatter = _paper_matplotlib()
    if plt is None:
        return None, None
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    bars = ax.bar(
        x,
        behavior,
        color=[PLOT_COLORS["primary"], PLOT_COLORS["secondary"]],
    )
    _annotate_bar_values(ax, bars, percent=True, fontsize=10)
    ax.set_xticks(x, labels)
    # Keep negative recovery visible when comparing objectives.
    _add_y_zero_line(ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Counterfactual KL recovery", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(
        "Behavioral recovery\nby training objective",
        fontsize=12,
        pad=BAR_TITLE_PAD,
    )
    fig.tight_layout()
    behavior_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(behavior_path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    bars = ax.bar(
        x,
        activation,
        color=[PLOT_COLORS["primary"], PLOT_COLORS["secondary"]],
    )
    _annotate_bar_values(ax, bars, percent=has_recovery, fontsize=10)
    ax.set_xticks(x, labels)
    if has_recovery:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_ylabel("NMSE recovery" if has_recovery else "Activation NMSE (lower is better)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(
        "Activation Recovery" if has_recovery else "Activation reconstruction",
        fontsize=12,
        pad=BAR_TITLE_PAD,
    )
    _add_y_zero_line(ax)
    fig.tight_layout()
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(activation_path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return behavior_path, activation_path


def _write_attribution_ablation_figure(
    rows: list[dict],
    *,
    method: str,
    validation: dict,
    path: Path,
) -> Path | None:
    """Plot one method's top-magnitude and random attribution sample."""
    x_key = f"{method}_total_attribution_credit"
    points = [
        row for row in rows
        if _is_evaluated_for_method(row, method)
        and row.get(x_key) is not None and row.get("mean_kl_increase") is not None
        and math.isfinite(float(row[x_key])) and math.isfinite(float(row["mean_kl_increase"]))
    ]
    if not points:
        return None
    plt, _ = _paper_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.scatter(
        [float(row[x_key]) for row in points],
        [float(row["mean_kl_increase"]) for row in points],
        s=PLOT_STYLE["scatter_size"],
        alpha=PLOT_STYLE["scatter_alpha"],
        color=SCATTER_COLORS["attribution"],
        linewidths=0,
        rasterized=True,
    )
    method_label = "IG" if method == "ig" else "RelP"
    ax.set_xlabel(f"{method_label} Attribution Score", fontsize=11)
    ax.set_ylabel("KL Change after Ablation", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    _add_scatter_headroom(ax)
    _add_correlation_summary_box(
        ax,
        median=validation.get("per_response_validation", {}).get("median_spearman"),
        pooled=validation.get("aggregate", {}).get("attribution_vs_ablation", {}).get("spearman"),
    )
    _add_x_zero_line(ax)
    _add_y_zero_line(ax)
    ax.set_title("Attribution vs. Causal Effect", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path


def _write_edit_norm_ablation_figure(
    rows: list[dict],
    *,
    method: str,
    validation: dict,
    path: Path,
) -> Path | None:
    """Plot edit norm on the same events sampled for one attribution method."""
    points = [
        row for row in rows
        if _is_evaluated_for_method(row, method)
        and row.get("direct_edit_norm") is not None
        and row.get("causal_effect_size") is not None
        and math.isfinite(float(row["direct_edit_norm"]))
        and math.isfinite(float(row["causal_effect_size"]))
    ]
    if not points:
        return None
    plt, _ = _paper_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.scatter(
        [float(row["direct_edit_norm"]) for row in points],
        [float(row["causal_effect_size"]) for row in points],
        s=PLOT_STYLE["scatter_size"],
        alpha=PLOT_STYLE["scatter_alpha"],
        color=SCATTER_COLORS["edit_norm"],
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("Edit Norm", fontsize=11)
    ax.set_ylabel("Causal Effect Size", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    _add_scatter_headroom(ax)
    _add_correlation_summary_box(
        ax,
        median=validation.get("edit_norm_per_response_validation", {}).get("median_spearman"),
        pooled=validation.get("aggregate", {}).get("edit_norm_vs_ablation", {}).get("spearman"),
    )
    ax.set_title("Edit Norm vs. Causal Effect Size", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path


def _add_correlation_summary_box(ax, *, median, pooled) -> None:
    """Add pooled and typical within-response rank agreement to a scatter plot."""
    try:
        median_value = float(median)
        pooled_value = float(pooled)
    except (TypeError, ValueError):
        return
    if not math.isfinite(median_value) or not math.isfinite(pooled_value):
        return
    ax.text(
        0.96,
        0.96,
        f"Median ρ = {median_value:.2f}\nPooled ρ = {pooled_value:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#AEB7C2", "alpha": 0.92},
        zorder=5,
    )


def _add_scatter_headroom(ax, fraction: float = 0.12) -> None:
    """Reserve a small amount of vertical space for an in-axis summary box."""
    lower, upper = ax.get_ylim()
    span = upper - lower
    if span > 0:
        ax.set_ylim(lower, upper + fraction * span)


def _is_evaluated_for_method(row: dict, method: str) -> bool:
    evaluation_method = row.get("evaluation_method")
    if evaluation_method is not None:
        return str(evaluation_method) == str(method)
    metadata = row.get("selection_groups", {}).get(str(method), {})
    return metadata.get("evaluation_group", metadata.get("group")) in {"top", "random"}


def _raw_scruples_label_map(scruples_dir: Path, split: str = "test") -> dict[str, str]:
    filename = {
        "train": "train.scruples-anecdotes.jsonl",
        "val": "dev.scruples-anecdotes.jsonl",
        "test": "test.scruples-anecdotes.jsonl",
    }[str(split)]
    path = Path(scruples_dir) / filename
    if not path.is_file():
        return {}
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(
                row.get("id") or row.get("example_id") or row.get("question_id")
                or row.get("qid") or index
            )
            label = str(row.get("binarized_label") or "").strip().upper()
            if label in {"WRONG", "RIGHT"}:
                result[source_id] = label
    return result


def _load_accuracy_tokenizer(model_name: str):
    """Load only the tokenizer, not model weights, for saved-response decoding."""
    from .model_utils import load_tokenizer

    return load_tokenizer(model_name)


def _answer_accuracy_summary(config: dict) -> dict:
    """Compare generated control/intervention answer accuracy on Scruples test."""
    cache_dir = Path(str(config.get("data", {}).get("cache_dir", "")))
    cache_metadata = read_json(cache_dir / "metadata.json", {})
    generated_dir = Path(str(cache_metadata.get("data_dir", "")))
    test_path = generated_dir / "test.jsonl"
    scruples_dir = Path(str(config.get("data", {}).get("scruples_dir", "")))
    if not test_path.is_file() or not scruples_dir.is_dir():
        return {"status": "unavailable", "reason": "generated test rows or Scruples source directory missing"}
    labels = _raw_scruples_label_map(scruples_dir, "test")
    if not labels:
        return {"status": "unavailable", "reason": "no WRONG/RIGHT binarized_label values found in Scruples test split"}
    rows = read_jsonl(test_path)
    tokenizer = None
    if any(row.get("response_input_ids") is not None for row in rows):
        model_name = config.get("model", {}).get("name") or cache_metadata.get("model_name")
        if not model_name:
            raise ValueError("Answer accuracy requires a model tokenizer to decode saved response IDs")
        tokenizer = _load_accuracy_tokenizer(str(model_name))
    by_arm = {}
    for arm in ("control", "intervention"):
        selected = [row for row in rows if str(row.get("arm", "")) == arm]
        parsed = 0
        correct = 0
        for row in selected:
            label = labels.get(str(row.get("source_id", "")))
            if label not in {"WRONG", "RIGHT"}:
                continue
            response = str(row.get("response", ""))
            prompt = str(row.get("rendered_prompt", ""))
            # Decode saved IDs with channel delimiters intact to separate reasoning
            # from the final answer. Propagate decoding failures rather than
            # parsing text that may omit those boundaries.
            if row.get("response_input_ids") is not None:
                response = tokenizer.decode(
                    row["response_input_ids"], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if row.get("prompt_input_ids") is not None:
                    prompt = tokenizer.decode(
                        row["prompt_input_ids"], skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
            choice = extract_choice_letter(
                response, {"A", "B"},
                prompt_text=prompt,
                finish_reason=row.get("finish_reason"),
                truncated=bool(row.get("response_truncated", False)),
            )
            if choice is None:
                continue
            parsed += 1
            swapped = bool((row.get("metadata") or {}).get("ab_swapped", False))
            yes_letter, no_letter = (("B", "A") if swapped else ("A", "B"))
            expected = yes_letter if label == "WRONG" else no_letter
            correct += int(choice == expected)
        by_arm[arm] = {
            "responses": len(selected),
            "parsed_with_label": parsed,
            "correct": correct,
            # Accuracy is conditional on a parsed A/B answer AND an available
            # ground-truth label; unparsed/unlabeled responses are excluded,
            # not counted as incorrect. `responses` retains the full arm count.
            "accuracy": (float(correct / parsed) if parsed else None),
        }
    return {"status": "complete", "split": "test", "arms": by_arm}


def _write_answer_accuracy_figure(summary: dict, path: Path) -> Path | None:
    if summary.get("status") != "complete":
        return None
    arms = summary.get("arms", {})
    values = [arms.get(arm, {}).get("accuracy") for arm in ("control", "intervention")]
    if any(value is None for value in values):
        return None
    plt, PercentFormatter = _paper_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    numeric = [float(value) for value in values]
    intervals = [
        _wilson_interval(
            int(arms.get(arm, {}).get("correct", 0)),
            int(arms.get(arm, {}).get("parsed_with_label", 0)),
        )
        for arm in ("control", "intervention")
    ]
    bars = ax.bar(
        ["Control", "Incorrect suggestion"], numeric,
        yerr=_asymmetric_errors(numeric, intervals), capsize=3,
        color=[PLOT_COLORS["primary"], "#d55e00"],
    )
    _annotate_bar_values(ax, bars, percent=True, fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Responses with correct final answer", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(
        "Final-answer accuracy\nbefore and after intervention",
        fontsize=12,
        pad=BAR_TITLE_PAD,
    )
    _add_y_zero_line(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path


def _recovery_audit(config: dict, run_dir: Path, teacher: dict, comparison: dict) -> dict:
    manifest=read_json(run_dir/"manifest.json",{})
    checkpoint_sha=str((manifest.get("checkpoint") or {}).get("sha256", ""))
    matches=[row for row in comparison.get("methods",[]) if str((row.get("checkpoint_identity") or {}).get("sha256", "")) == checkpoint_sha and checkpoint_sha]
    current=matches[0] if len(matches)==1 else None
    canonical=float(current["test_recovery"]) if current is not None else None
    replay=float(teacher["recovery"]) if teacher.get("recovery") is not None else None
    same_count=(int(comparison.get("sequence_direction_records",-1)) == int(teacher.get("records",-2)))
    if current is None:
        status="canonical checkpoint row not found"
    elif same_count:
        status=(
            "same checkpoint and sequence-direction count; analysis replay is an "
            "independent execution whose bf16 padded-batch shape can change the "
            "edited forward slightly, so condition KLs are reported below"
        )
    else:
        status="checkpoint matched, but evaluation populations differ"
    canonical_source = float(current["test_source_kl_mean"]) if current is not None and current.get("test_source_kl_mean") is not None else None
    canonical_edited = float(current["test_edited_kl_mean"]) if current is not None and current.get("test_edited_kl_mean") is not None else None
    replay_source = float(teacher["source_kl_mean"]) if teacher.get("source_kl_mean") is not None else None
    replay_edited = float(teacher["edited_kl_mean"]) if teacher.get("edited_kl_mean") is not None else None
    return {
        "checkpoint_sha256": checkpoint_sha,
        "canonical_method": current.get("method") if current else None,
        "canonical_test_recovery": canonical,
        "analysis_replay_recovery": replay,
        "recovery_difference_analysis_minus_canonical": (replay-canonical if replay is not None and canonical is not None else None),
        "canonical_source_kl_mean": canonical_source,
        "analysis_source_kl_mean": replay_source,
        "source_kl_difference_analysis_minus_canonical": (replay_source-canonical_source if replay_source is not None and canonical_source is not None else None),
        "canonical_edited_kl_mean": canonical_edited,
        "analysis_edited_kl_mean": replay_edited,
        "edited_kl_difference_analysis_minus_canonical": (replay_edited-canonical_edited if replay_edited is not None and canonical_edited is not None else None),
        "canonical_sequence_direction_records": comparison.get("sequence_direction_records"),
        "analysis_sequence_direction_records": teacher.get("records"),
        "same_sequence_direction_count": same_count,
        "status": status,
    }


def _write_support_coverage_figure(distribution: dict, path: Path) -> Path | None:
    """Plot a precomputed conditional-support mass histogram."""
    try:
        counts = np.asarray(distribution.get("counts", []), dtype=np.float64)
        edges = np.asarray(distribution.get("edges", []), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not (
        counts.size
        and counts.sum() > 0
        and edges.size == counts.size + 1
        and np.isfinite(counts).all()
        and np.isfinite(edges).all()
    ):
        return None
    plt, PercentFormatter = _paper_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.stairs(
        counts,
        edges,
        fill=True,
        color=PLOT_COLORS["primary"],
        alpha=1.0,
        linewidth=1.0,
    )
    ax.set_xlim(max(0.0, float(edges[0])), min(1.0, float(edges[-1])))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(distribution.get("x_label", "Minimum Support Mass"), fontsize=11)
    ax.set_ylabel("Tokens", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(distribution.get("title", "Distribution of Support Mass"), fontsize=12, fontweight="bold")
    p10 = distribution.get("exact_p10", distribution.get("p10"))
    minimum = distribution.get("exact_minimum", distribution.get("minimum"))
    try:
        summary_text = f"P10 = {float(p10):.3f}\nMin = {float(minimum):.3f}"
    except (TypeError, ValueError):
        summary_text = None
    if summary_text is not None:
        ax.text(
            0.04,
            0.96,
            summary_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#AEB7C2",
                "alpha": 0.92,
            },
            zorder=5,
        )
    _add_y_zero_line(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path


def _top_edit_attribution_tokens(
    teacher_records: list[dict],
    attribution_rows: list[dict],
    *,
    top_fraction: float = 0.05,
    limit: int = 30,
) -> dict:
    """Aggregate downstream IG credit by the token at each applied edit event.

    The first ranking counts how often a token occurrence is among the largest
    ``top_fraction`` absolute IG credits *within its record*. The second ranks
    tokens by the magnitude of their signed cumulative IG credit, while also
    reporting cumulative absolute credit to make cancellation visible.
    """
    fraction = float(top_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1]")
    by_record = {str(record.get("record_id", "")): record for record in teacher_records}
    stats: dict[tuple[int, str], dict] = {}
    total_occurrences = 0
    for row in attribution_rows:
        if str(row.get("mode", "teacher_forced")) != "teacher_forced":
            continue
        record = by_record.get(str(row.get("record_id", "")))
        if not record:
            continue
        tokens = list(record.get("tokens", []))
        credits = [float(value) for value in row.get("edit_event_credit", [])]
        n = min(len(tokens), len(credits))
        if n <= 0:
            continue
        valid = [index for index in range(n) if math.isfinite(credits[index])]
        if not valid:
            continue
        k = max(1, int(math.ceil(len(valid) * fraction)))
        selected = set(sorted(valid, key=lambda i: abs(credits[i]), reverse=True)[:k])
        for index in valid:
            token = tokens[index]
            token_id = int(token.get("token_id", -1))
            raw = str(token.get("token_text", ""))
            display = raw.replace("\u200b", "")
            if display == "":
                display = "·"
            key = (token_id, raw)
            item = stats.setdefault(
                key,
                {
                    "token_id": token_id,
                    "token_text": display,
                    "occurrences": 0,
                    "top_fraction_count": 0,
                    "signed_sum": 0.0,
                    "absolute_sum": 0.0,
                },
            )
            value = credits[index]
            item["occurrences"] += 1
            item["top_fraction_count"] += int(index in selected)
            item["signed_sum"] += value
            item["absolute_sum"] += abs(value)
            total_occurrences += 1
    rows = list(stats.values())
    for item in rows:
        item["top_fraction_rate"] = (
            float(item["top_fraction_count"] / item["occurrences"])
            if item["occurrences"] else 0.0
        )
        item["mean_signed_credit"] = (
            float(item["signed_sum"] / item["occurrences"])
            if item["occurrences"] else 0.0
        )
    by_count = sorted(
        rows,
        key=lambda item: (
            -int(item["top_fraction_count"]),
            -float(item["absolute_sum"]),
            -int(item["occurrences"]),
            int(item["token_id"]),
        ),
    )[: max(0, int(limit))]
    by_sum = sorted(
        rows,
        key=lambda item: (
            -abs(float(item["signed_sum"])),
            -float(item["absolute_sum"]),
            -int(item["occurrences"]),
            int(item["token_id"]),
        ),
    )[: max(0, int(limit))]
    return {
        "top_fraction": fraction,
        "total_token_occurrences": total_occurrences,
        "by_top_fraction_count": by_count,
        "by_signed_sum_magnitude": by_sum,
    }


def _write_attribution_method_correlation_figure(
    validation_by_method: dict[str, dict],
    methods: list[str],
    path: Path,
) -> Path | None:
    """Compare attribution and edit-norm correlations on each method's sample."""
    rows = []
    for method in methods:
        method_validation = validation_by_method.get(method, {})
        aggregate = method_validation.get("aggregate", {})
        attribution_value = method_validation.get("per_response_validation", {}).get("median_spearman")
        norm_value = method_validation.get("edit_norm_per_response_validation", {}).get("median_spearman")
        pooled_attribution = aggregate.get("attribution_vs_ablation", {}).get("spearman")
        pooled_norm = aggregate.get("edit_norm_vs_ablation", {}).get("spearman")
        if (
            attribution_value is None
            or norm_value is None
            or pooled_attribution is None
            or pooled_norm is None
        ):
            continue
        try:
            attribution_value = float(attribution_value)
            norm_value = float(norm_value)
            pooled_attribution = float(pooled_attribution)
            pooled_norm = float(pooled_norm)
        except (TypeError, ValueError):
            continue
        if all(
            math.isfinite(value)
            for value in (attribution_value, norm_value, pooled_attribution, pooled_norm)
        ):
            rows.append(
                (
                    method,
                    pooled_attribution,
                    attribution_value,
                    pooled_norm,
                    norm_value,
                )
            )
    if not rows:
        return None
    plt, _ = _paper_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    width = 0.34
    categories = [
        label
        for row in rows
        for label in (f"{row[0].upper()}\nPooled", f"{row[0].upper()}\nMedian")
    ]
    x = np.arange(len(categories), dtype=np.float64)
    attribution_values = [value for row in rows for value in (row[1], row[2])]
    norm_values = [value for row in rows for value in (row[3], row[4])]
    attribution_bars = ax.bar(
        x - width / 2,
        attribution_values,
        width,
        label="Attribution",
    )
    norm_bars = ax.bar(
        x + width / 2,
        norm_values,
        width,
        label="Edit norm → |effect|",
    )
    _annotate_bar_values(ax, attribution_bars)
    _annotate_bar_values(ax, norm_bars)
    ax.set_xticks(x, categories)
    _add_y_zero_line(ax)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Spearman ρ")
    ax.set_title(
        "Signed attribution and edit norm versus causal effects",
        pad=BAR_TITLE_PAD,
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path


def _per_sequence_spearman_values(ablation: dict, methods: list[str]) -> dict[str, list[float]]:
    """Return finite within-sequence Spearman values for each attribution method.

    Stage 3 stores these directly in ``per_response_validation``.
    """
    result: dict[str, list[float]] = {}
    validation_by_method = ablation.get("validation_by_method", {})
    for method in methods:
        direct = (
            validation_by_method.get(method, {})
            .get("per_response_validation", {})
            .get("spearman_values", [])
        )
        values = []
        for value in direct:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        result[method] = values
    return result


def _write_attribution_correlation_distribution_figure(
    ablation: dict,
    method: str,
    path: Path,
) -> Path | None:
    """Plot one method's within-sequence attribution/ablation Spearman rho."""
    values = _per_sequence_spearman_values(ablation, [method]).get(method, [])
    if not values:
        return None
    plt, PercentFormatter = _paper_matplotlib()
    if plt is None:
        return None
    bins = np.linspace(-1.0, 1.0, 21, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    array = np.asarray(values, dtype=np.float64)
    weights = np.full(array.shape, 1.0 / max(1, int(array.size)), dtype=np.float64)
    ax.hist(
        array,
        bins=bins,
        weights=weights,
        color=ATTRIBUTION_METHOD_COLORS.get(method, PLOT_COLORS["primary"]),
        alpha=1.0,
        label=f"median {float(np.median(array)):.2f}",
    )
    _add_x_zero_line(ax)
    ax.set_xlim(-1.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Spearman ρ", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title("Correlation Across Responses", fontsize=12)
    ax.legend(frameon=False)
    _add_y_zero_line(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path


def _write_causal_effect_size_comparison_figure(
    validation: dict,
    *,
    method: str,
    path: Path,
) -> Path | None:
    """Compare mean response-level effect magnitude for top and random events."""
    distribution = validation.get("causal_effect_size_distribution", {})
    top = [float(value) for value in distribution.get("top", {}).get("values", [])]
    random_values = [
        float(value) for value in distribution.get("random", {}).get("values", [])
    ]
    if not top and not random_values:
        return None
    plt, _ = _paper_matplotlib()
    if plt is None:
        return None
    groups = [top, random_values]
    means = [float(np.mean(values)) if values else 0.0 for values in groups]
    intervals = [
        bootstrap_mean_confidence_interval(values, seed=1701 + index)
        for index, values in enumerate(groups)
    ]
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    bars = ax.bar(
        np.arange(2),
        means,
        yerr=_asymmetric_errors(means, intervals),
        capsize=4,
        color=[PLOT_COLORS["primary"], PLOT_COLORS["secondary"]],
    )
    _annotate_bar_values(ax, bars, fontsize=10)
    ax.set_xticks(np.arange(2), ["Top", "Random"])
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Mean Causal Effect Size", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    method_label = "IG" if method == "ig" else "RelP"
    ax.set_title(
        f"{method_label} Top Attribution\nvs. Random",
        fontsize=12,
        pad=BAR_TITLE_PAD,
    )
    _add_y_zero_line(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_STYLE["dpi"])
    plt.close(fig)
    return path

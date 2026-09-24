from __future__ import annotations

from .native_hypothesis_sites import PATCH_SITE_ORDER, PATCH_SITE_LABELS, PATCH_PLOT_ORDER, PATCH_PLOT_LABELS

"""Internal report implementation for :mod:`behavior_pipeline.native_hypothesis`."""

import html
import json
import math
from pathlib import Path

import numpy as np

from .counterfactual_runtime import token_texts
from .native_hypothesis_sites import (
    _display_token,
)
from .counterfactual_report_plot_style import (
    PLOT_COLORS,
    PLOT_STYLE,
    apply_report_plot_style,
)


def _fmt(value, digits=4):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(value):
        return "–"
    return f"{value:.{digits}g}"


def _annotate_bar_values(ax, bars, *, percent: bool = False) -> None:
    labels = []
    for patch in bars:
        value = float(patch.get_height())
        if not math.isfinite(value):
            labels.append("")
        elif percent:
            labels.append(f"{value:.1%}")
        else:
            labels.append(f"{value:.3g}")
    try:
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    except AttributeError:
        pass


def _site_label(site: str) -> str:
    return PATCH_SITE_LABELS.get(str(site), str(site).replace("_", " ").title())


def _response_effect_heatmap(
    token_texts: list[str],
    values: list[float],
    *,
    label: str,
    hover_details: list[str] | None = None,
) -> str:
    if not token_texts or not values:
        return '<p class="muted">No response-level causal effect available.</p>'
    count = min(len(token_texts), len(values))
    scale = max((abs(float(values[index])) for index in range(count)), default=0.0)
    if scale <= 0.0:
        scale = 1.0
    spans = []
    for index in range(count):
        value = float(values[index])
        strength = min(1.0, abs(value) / scale)
        if value >= 0:
            background = f"rgba(190,45,55,{0.10 + 0.70*strength:.3f})"
        else:
            background = f"rgba(31,95,180,{0.10 + 0.70*strength:.3f})"
        text = html.escape(_display_token(str(token_texts[index])))
        title_parts = [f"t={index}", f"effect={value:.6g}"]
        if hover_details is not None and index < len(hover_details):
            extra = str(hover_details[index]).strip()
            if extra:
                title_parts.append(extra)
        title = html.escape("\n".join(title_parts), quote=True)
        spans.append(
            f'<span class="effect-token" style="background:{background}" title="{title}">{text}</span>'
        )
    return (
        f'<div class="heatmap-label">{html.escape(label)} · raw scale max |effect|={scale:.4g}</div>'
        f'<div class="effect-heatmap">{"".join(spans)}</div>'
    )


def _native_likelihood_hover(source: dict, target: dict, condition: dict) -> list[str]:
    source_p = list(source.get("per_response_observed_token_probability", []))
    target_p = list(target.get("per_response_observed_token_probability", []))
    condition_p = list(condition.get("per_response_observed_token_probability", []))
    source_lp = list(source.get("per_response_observed_token_logp", []))
    condition_lp = list(condition.get("per_response_observed_token_logp", []))
    probability_table = {
        int(row.get("response_index", -1)): list(row.get("alternatives", []))
        for row in condition.get("qualitative_probability_table", [])
        if isinstance(row, dict)
    }
    count = max(len(source_p), len(target_p), len(condition_p), len(source_lp), len(condition_lp))
    result = []
    for index in range(count):
        def value(values):
            return float(values[index]) if index < len(values) else float("nan")
        sp, tp, cp = value(source_p), value(target_p), value(condition_p)
        slp, clp = value(source_lp), value(condition_lp)
        parts = []
        if math.isfinite(sp): parts.append(f"P(observed token) source: {sp:.6g}")
        if math.isfinite(tp): parts.append(f"P(observed token) target: {tp:.6g}")
        if math.isfinite(cp): parts.append(f"P(observed token) patched: {cp:.6g}")
        if math.isfinite(sp) and math.isfinite(cp): parts.append(f"ΔP patched-source: {cp-sp:+.6g}")
        if math.isfinite(slp) and math.isfinite(clp): parts.append(f"Δlog P patched-source: {clp-slp:+.6g}")
        alternatives = probability_table.get(index, [])
        if alternatives:
            parts.append("Top likelihood alternatives:")
            for alt in alternatives[:5]:
                token_text = str(alt.get("token_text", "")).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "↵\n")
                parts.append(
                    f"{token_text!r}: src {float(alt.get('source', 0.0)):.3g}, "
                    f"tgt {float(alt.get('target', 0.0)):.3g}, "
                    f"patch {float(alt.get('native_patch', 0.0)):.3g}"
                )
        result.append("\n".join(parts))
    return result


def _metric_value(metric: dict, *keys: str) -> float | None:
    for key in keys:
        value = metric.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _metric_errors(metrics: dict[str, dict], sites: list[str], values: list[float]) -> np.ndarray:
    lower, upper = [], []
    for site, value in zip(sites, values):
        ci = metrics.get(site, {}).get("bootstrap_95_ci")
        if ci and len(ci) == 2 and math.isfinite(float(value)):
            lower.append(max(0.0, float(value) - float(ci[0])))
            upper.append(max(0.0, float(ci[1]) - float(value)))
        else:
            lower.append(0.0)
            upper.append(0.0)
    return np.asarray([lower, upper], dtype=np.float64)


def write_native_quantitative_figures(output: Path, summary: dict) -> dict[str, Path]:
    """Write the aggregate native activation-patching figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_report_plot_style(matplotlib, plt)
    except Exception:
        return {}

    output = Path(output)
    paths: dict[str, Path] = {}
    native = summary.get("native_patching", {})
    patch_metrics = native.get("sites", {})
    control_metrics = native.get("random_controls", {})
    native_sites = [
        site for site in PATCH_PLOT_ORDER
        if _metric_value(patch_metrics.get(site, {}), "patch_recovery", "value") is not None
        or _metric_value(control_metrics.get(site, {}), "patch_recovery", "value") is not None
    ]
    if native_sites:
        patch_values = [
            float(_metric_value(patch_metrics.get(site, {}), "patch_recovery", "value") or 0.0)
            for site in native_sites
        ]
        control_values = [
            float(_metric_value(control_metrics.get(site, {}), "patch_recovery", "value") or 0.0)
            for site in native_sites
        ]
        x = np.arange(len(native_sites), dtype=np.float64)
        width = 0.36
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        patch_bars = ax.bar(
            x - width / 2,
            patch_values,
            width,
            label="Native patch",
            yerr=_metric_errors(patch_metrics, native_sites, patch_values),
            capsize=3,
            color=PLOT_COLORS["native_patch"],
        )
        control_bars = ax.bar(
            x + width / 2,
            control_values,
            width,
            label="Equal-size random patch",
            yerr=_metric_errors(control_metrics, native_sites, control_values),
            capsize=3,
            color=PLOT_COLORS["secondary"],
        )
        _annotate_bar_values(ax, patch_bars)
        _annotate_bar_values(ax, control_bars)
        ax.axhline(0.0, linewidth=0.8)
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper * 1.35)
        ax.set_xticks(x, [PATCH_PLOT_LABELS[site] for site in native_sites])
        ax.set_ylabel("KL recovery", fontsize=PLOT_STYLE["label_size"] + 1)
        ax.tick_params(axis="both", labelsize=PLOT_STYLE["tick_size"] + 1)
        ax.set_title(
            "Native activation patching",
            fontsize=PLOT_STYLE["title_size"] + 1,
            pad=12,
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output / "native_activation_patching.png"
        fig.savefig(path, dpi=PLOT_STYLE["dpi"])
        plt.close(fig)
        paths["native_patching"] = path
    return paths


def _example_rank(payload: dict) -> float:
    score = 0.0
    for site in PATCH_SITE_ORDER:
        fixed = payload.get("native_fixed_prefix", {}).get("patches", {}).get(site, {})
        score = max(score, abs(float(fixed.get("recovery", 0.0) or 0.0)))
    return score


def _select_native_report_payloads(
    output: Path,
    summary: dict,
    *,
    maximum: int,
) -> list[dict]:
    """Select report examples by streaming native JSON records, not retaining all payloads."""
    output = Path(output)
    record_dir = output / "records"
    selected: list[dict] = []
    ranked_paths: list[tuple[float, str, str, Path]] = []
    for path in sorted(record_dir.glob("*.json")):
        rid = path.stem
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranked_paths.append((
            _example_rank(payload),
            str(payload.get("source_id", "")),
            str(payload.get("direction", "")),
            path,
        ))
        del payload
    ranked_paths.sort(key=lambda row: (-row[0], row[1], row[2]))
    remaining = max(0, int(maximum) - len(selected))
    for _score, _source, _direction, path in ranked_paths[:remaining]:
        selected.append(json.loads(path.read_text(encoding="utf-8")))
    return selected[: int(maximum)]


def build_native_hypothesis_report(
    config: dict,
    output: Path,
    payloads: list[dict] | None,
    summary: dict,
) -> None:
    """Build the semantic-site and native-patching causal report."""
    output = Path(output)
    settings = config.get("native_analysis", {})
    max_examples = int(settings.get("report_examples", 20))
    native = summary.get("native_patching", {})
    figure_paths = write_native_quantitative_figures(output, summary)

    patch_rows = []
    for site in PATCH_SITE_ORDER:
        patch_row = native.get("sites", {}).get(site, {})
        control_row = native.get("random_controls", {}).get(site, {})
        patch_value = _metric_value(patch_row, "patch_recovery", "value")
        control_value = _metric_value(control_row, "patch_recovery", "value")
        if patch_value is None and control_value is None:
            continue
        patch_ci = patch_row.get("bootstrap_95_ci") or [None, None]
        control_ci = control_row.get("bootstrap_95_ci") or [None, None]
        patch_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(_site_label(site))}</strong></td>"
            f"<td>{_fmt(patch_value)}</td>"
            f"<td>[{_fmt(patch_ci[0])}, {_fmt(patch_ci[1])}]</td>"
            f"<td>{_fmt(control_value)}</td>"
            f"<td>[{_fmt(control_ci[0])}, {_fmt(control_ci[1])}]</td>"
            "</tr>"
        )

    ordered = (
        list(payloads)[:max_examples]
        if payloads is not None
        else _select_native_report_payloads(output, summary, maximum=max_examples)
    )

    examples = []
    for payload in ordered:
        rid = str(payload.get("record_id", ""))
        direction = str(payload.get("direction", ""))

        site_sections = []
        response_tokens = payload.get("response_token_texts", [])
        native_fixed = payload.get("native_fixed_prefix", {})
        source_native = native_fixed.get("source", {}) if isinstance(native_fixed, dict) else {}
        target_native = native_fixed.get("target", {}) if isinstance(native_fixed, dict) else {}
        for site in PATCH_SITE_ORDER:
            patch = payload.get("native_fixed_prefix", {}).get("patches", {}).get(site, {})
            control = payload.get("native_fixed_prefix", {}).get("random_controls", {}).get(site, {})
            if not patch and not control:
                continue
            parts = [f"<h4>{html.escape(_site_label(site))}</h4>"]
            if patch:
                parts.append(
                    '<div class="metric-row">'
                    f'<span>Native patch recovery</span><strong>{_fmt(patch.get("recovery"))}</strong>'
                    f'<span>Patch KL</span><strong>{_fmt(patch.get("condition_kl_mean"))}</strong>'
                    '</div>'
                )
                parts.append(_response_effect_heatmap(
                    response_tokens,
                    patch.get("per_response_kl_recovered", []),
                    label="Per-token counterfactual KL recovered by native patch",
                    hover_details=_native_likelihood_hover(source_native, target_native, patch),
                ))
            if control:
                parts.append(
                    '<div class="metric-row">'
                    f'<span>Equal-size random patch recovery</span><strong>{_fmt(control.get("recovery"))}</strong>'
                    f'<span>Random patch KL</span><strong>{_fmt(control.get("condition_kl_mean"))}</strong>'
                    '</div>'
                )
                parts.append(_response_effect_heatmap(
                    response_tokens,
                    control.get("per_response_kl_recovered", []),
                    label="Per-token counterfactual KL recovered by equal-size random aligned-token patch",
                    hover_details=_native_likelihood_hover(source_native, target_native, control),
                ))

            mass = payload.get("mass_shift", {}).get(site, {})
            top_positions = mass.get("top_positions", [])[:3]
            if top_positions:
                lexical_items = []
                for pos in top_positions:
                    gains = ", ".join(
                        f"{_display_token(str(row.get('token_text','')))} ({_fmt(row.get('canonical_delta_probability'),3)})"
                        for row in pos.get("top_gains", [])[:4]
                    )
                    losses = ", ".join(
                        f"{_display_token(str(row.get('token_text','')))} ({_fmt(row.get('canonical_delta_probability'),3)})"
                        for row in pos.get("top_losses", [])[:4]
                    )
                    lexical_items.append(
                        f"<li><strong>t={int(pos.get('response_index',0))}</strong> "
                        f"{html.escape(str(pos.get('context','')))}<br>"
                        f"gains: {html.escape(gains)}<br>losses: {html.escape(losses)}</li>"
                    )
                parts.append(
                    '<details><summary>Largest literal token-probability movements for this patch</summary>'
                    f'<ul>{"".join(lexical_items)}</ul></details>'
                )
            site_sections.append("".join(parts))

        examples.append(
            '<article class="example">'
            f'<h3>{html.escape(str(payload.get("source_id")))} · {html.escape(str(payload.get("direction")))}</h3>'
            f'{"".join(site_sections)}'
            '</article>'
        )

    css = """
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#fafafa;color:#202124}
main{max-width:1200px;margin:auto;padding:24px}h1,h2,h3,h4{line-height:1.2}section,.example{background:white;border:1px solid #ddd;border-radius:10px;padding:18px;margin:18px 0}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid #eee;padding:7px;text-align:left;vertical-align:top}.muted{color:#777}details{margin:10px 0}.metric-row{display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;margin:8px 0}.metric-row span{color:#666}.metric-row strong{margin-right:14px}.effect-heatmap{line-height:2.1;padding:8px;border:1px solid #eee;border-radius:6px}.effect-token{padding:3px 1px;white-space:pre-wrap}.heatmap-label{font-size:12px;color:#666;margin:6px 0}pre{white-space:pre-wrap;word-break:break-word;background:#f6f7f8;padding:10px;border-radius:6px;max-height:420px;overflow:auto}img.figure{max-width:760px;width:100%;height:auto}
"""
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native user-aware reasoning analysis</title><style>{css}</style></head><body><main>
<h1>Native user-aware reasoning analysis</h1>
<p>The quantitative analysis tests information carried at the configured filler sites and in the representation of the full question. Gemma filler sites are response newlines and structural line-prefix whitespace, excluding prompt positions and special delimiters; DeepSeek uses its separate newline/punctuation rule. Native activation patching reports the fraction of source-to-target conditional KL recovered under the same fixed prefix, with equal-size random aligned-token controls. Token-level probability shifts are qualitative checks on the meaning of those native effects.</p>
<section><h2>Native activation patching</h2>{(f'<img class="figure" src="{html.escape(figure_paths["native_patching"].name)}" alt="native activation patching effects">' if "native_patching" in figure_paths else '')}
<div class="table-wrap"><table><thead><tr><th>Site</th><th>Native patch recovery</th><th>95% CI</th><th>Random patch recovery</th><th>95% CI</th></tr></thead><tbody>{''.join(patch_rows)}</tbody></table></div></section>
<section><h2>Representative examples</h2>{''.join(examples)}</section>
</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")

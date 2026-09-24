from __future__ import annotations

"""Internal html implementation for :mod:`behavior_pipeline.counterfactual_report`."""

import html
import json
import math
import re
from pathlib import Path

import numpy as np

from .counterfactual_runtime import (
    qualitative_record_key,
    select_qualitative_record_keys,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "–"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(value):
        return "–"
    return f"{value:.{digits}g}"


def _percent(value, digits: int = 1) -> str:
    if value is None:
        return "–"
    return f"{100.0 * float(value):.{digits}f}%"


def _method_label(name: str) -> str:
    mapping = {
        "fixed_steering": "Fixed vector",
        "directional_linear": "Linear",
        "edit_module_base_mlp": "MLP",
        "base_mlp": "MLP",
    }
    text = str(name)
    return mapping.get(text, text.replace("_", " "))


def _page(title: str, body: str, *, relative_root: str = "") -> str:
    nav = (
        f'<a href="{relative_root}index.html">Summary</a>'
        f'<a href="{relative_root}causal_validation.html">Attribution validation</a>'
        f'<a href="{relative_root}exhaustive_validation.html">Exhaustive validation</a>'
        f'<a href="{relative_root}sentence_browser.html">Sentence browser</a>'
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><link rel="stylesheet" href="{relative_root}assets/report.css">
</head><body><header class="top"><strong>Counterfactual edit analysis</strong><nav>{nav}</nav></header>
<main><h1>{html.escape(title)}</h1>{body}</main>
<script src="{relative_root}assets/report.js"></script></body></html>"""

_ZWSP_ENTITY = re.compile(r"(?i)&(?:amp;)*#(?:x0*200b|8203);?")


def _display_token_text(raw: str) -> tuple[str, bool]:
    """Preserve decoded whitespace while making line breaks legible.

    Spaces and tabs are rendered exactly as decoded. Each newline keeps a visible
    ``↵`` marker *and* emits the real line break, so one- and two-newline tokens
    create the same vertical structure as the underlying text. The raw decode is
    still preserved in the tooltip whenever zero-width cleanup or newline markup
    changes the visible string.
    """
    raw = str(raw)
    cleaned = _ZWSP_ENTITY.sub("", raw).replace("\u200b", "").replace("▁", " ")
    changed = cleaned != raw
    if not cleaned and raw:
        return "", True
    if "\n" in cleaned or "\r" in cleaned:
        normalized = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        return normalized.replace("\n", "↵\n"), True
    return cleaned, changed


def _heat_color(value: float, scale: float) -> str:
    if scale <= 0.0 or value == 0.0:
        return "transparent"
    alpha = min(0.78, max(0.025, abs(float(value)) / scale * 0.72))
    return (
        f"rgba(214,95,95,{alpha:.3f})"
        if value > 0
        else f"rgba(76,120,168,{alpha:.3f})"
    )


def _callout_token_text(value: object) -> str:
    return str(value or "").replace("▁", " ").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "↵")


def _callout_row(
    label: str,
    alternatives: list[dict],
    value_key: str,
    *,
    kind: str,
    limit: int = 3,
) -> dict:
    values = [
        {
            "token": _callout_token_text(row.get("token_text", "")),
            "value": float(row.get(value_key, 0.0) or 0.0),
        }
        for row in alternatives
        if isinstance(row, dict) and math.isfinite(float(row.get(value_key, 0.0) or 0.0))
    ]
    values.sort(key=lambda row: row["value"], reverse=True)
    return {"label": label, "kind": kind, "items": values[: int(limit)]}


def _support_probability_callout(alternatives: object) -> dict | None:
    if not isinstance(alternatives, list) or not alternatives:
        return None
    return {
        "rows": [
            _callout_row("Source", alternatives, "source_probability", kind="source"),
            _callout_row("Target", alternatives, "target_probability", kind="target"),
            _callout_row("Edit", alternatives, "edited_probability", kind="edit"),
        ]
    }


def _token_viewer(
    record: dict,
    values: list[float],
    *,
    label: str,
    response_only: bool = False,
    selected_index: int | None = None,
    selected_indices: list[int] | set[int] | tuple[int, ...] | None = None,
    legend_positive: str = "positive",
    legend_negative: str = "negative",
    hover_details: list[str] | None = None,
    hover_callouts: list[dict | None] | None = None,
    include_default_support_changes: bool = True,
) -> str:
    prompt_len = int(record.get("prompt_tokens", 0))
    tokens = list(record.get("tokens", []))
    start = prompt_len if response_only else 0
    shown = tokens[start:]
    vals = [float(v) for v in values]
    if response_only:
        if len(vals) == len(tokens):
            vals = vals[prompt_len:]
        elif len(vals) != len(shown):
            vals = (vals + [0.0] * len(shown))[: len(shown)]
    else:
        vals = (vals + [0.0] * len(tokens))[: len(tokens)]
    finite_abs = np.asarray(
        [abs(v) for v in vals if math.isfinite(v) and abs(v) > 0.0],
        dtype=np.float64,
    )
    full_scale = float(finite_abs.max()) if finite_abs.size else 1.0
    clipped_scale = float(np.percentile(finite_abs, 95.0)) if finite_abs.size else 1.0
    full_scale = max(full_scale, 1e-15)
    clipped_scale = max(clipped_scale, 1e-15)
    spans = []
    support_changes = record.get("support_changes", []) if include_default_support_changes else []
    selected_set = {int(value) for value in (selected_indices or [])}
    if selected_index is not None:
        selected_set.add(int(selected_index))
    for local_index, token in enumerate(shown):
        global_index = start + local_index
        value = vals[local_index]
        raw = str(token.get("token_text", ""))
        display, normalized = _display_token_text(raw)
        details = [f"token {global_index}", f"{label}: {value:.6g}"]
        if normalized:
            details.append(f"raw decoded token: {raw!r}")
        response_index = global_index - prompt_len
        if hover_details is not None and 0 <= response_index < len(hover_details):
            extra = str(hover_details[response_index]).strip()
            if extra:
                details.append(extra)
        callout = None
        if hover_callouts is not None and 0 <= response_index < len(hover_callouts):
            candidate = hover_callouts[response_index]
            if isinstance(candidate, dict):
                callout = dict(candidate)
        if callout is None and 0 <= response_index < len(support_changes):
            callout = _support_probability_callout(support_changes[response_index])
        if callout is None:
            callout = {"rows": []}
        heading_prefix = str(callout.pop("heading_prefix", "Token"))
        callout.update({
            "heading": f"{heading_prefix}: {_callout_token_text(raw)!r}",
            "metric_label": label,
            "metric_value": value,
        })
        if hover_details is not None and 0 <= response_index < len(hover_details):
            note = str(hover_details[response_index]).strip()
            if note and not callout.get("rows"):
                callout["note"] = note
        classes = ["tok"]
        if global_index < prompt_len:
            classes.append("prompt")
        if global_index in selected_set:
            classes.append("selected")
        tooltip = html.escape(json.dumps(callout, ensure_ascii=False, separators=(",", ":")), quote=True)
        aria = html.escape("; ".join(details[:2]), quote=True)
        text = html.escape(display)
        spans.append(
            f'<span class="{" ".join(classes)}" style="background:{_heat_color(value, clipped_scale)}" '
            f'data-value="{value:.17g}" data-prompt="{1 if global_index < prompt_len else 0}" '
            f'data-callout="{tooltip}" tabindex="0" aria-label="{aria}">{text}</span>'
        )
    viewer_id = f"viewer-{abs(hash((str(record.get('record_id')), label, response_only, tuple(sorted(selected_set))))) % 10**12}"
    return (
        f'<div class="viewer" id="{viewer_id}" data-scale-full="{full_scale:.17g}" data-scale-p95="{clipped_scale:.17g}">'
        '<div class="viewer-toolbar">'
        f'<div class="legend"><span class="neg-swatch"></span>{html.escape(legend_negative)}'
        '<span class="legend-gradient"></span>'
        f'{html.escape(legend_positive)}<span class="pos-swatch"></span></div>'
        '<label class="scale-note">Scale '
        '<select class="attribution-scale-mode">'
        '<option value="p95" selected>95th percentile |magnitude|</option>'
        '<option value="full">Full range</option>'
        '</select></label>'
        f'<span class="scale-note scale-value">95th percentile |scale| = {_number(clipped_scale)}</span>'
        '<div class="viewer-actions">'
        f'<button type="button" class="expand" data-viewer="{viewer_id}">Expand</button>'
        '</div></div>'
        f'<div class="viewer-text">{"".join(spans)}</div>'
        '</div>'
    )


def _effect_span_html(spans: dict) -> str:
    parts = []
    for sign, title in (("positive", "Largest KL increases"), ("negative", "Largest KL decreases")):
        rows = spans.get(sign, []) if isinstance(spans, dict) else []
        if not rows:
            continue
        items = "".join(
            f'<li><code>{html.escape(str(row.get("phrase", "")))}</code>'
            f'<span>{_number(row.get("score_sum"))}</span></li>'
            for row in rows[:8]
        )
        parts.append(f'<h5>{title}</h5><ul class="compact-list">{items}</ul>')
    if not parts:
        return '<p class="muted">No nonzero response spans were extracted.</p>'
    return "".join(parts)


def _case_records(records: list[dict], fraction: float, seed: int) -> list[dict]:
    selected_keys = select_qualitative_record_keys(
        records, fraction=float(fraction), seed=int(seed)
    )
    selected = [record for record in records if qualitative_record_key(record) in selected_keys]
    return sorted(
        selected,
        key=lambda record: (
            str(record.get("source_id", "")),
            str(record.get("response_id", "")),
            str(record.get("direction", "")),
        ),
    )


def _report_primary_method(config: dict, methods: list[str]) -> str:
    method = str(
        config.get("report", {}).get(
            "primary_attribution_method",
            config.get("attribution", {}).get("primary_method", methods[0]),
        )
    )
    return method if method in methods else methods[0]


def _report_visible_methods(config: dict, methods: list[str]) -> list[str]:
    if str(config.get("report", {}).get("attribution_view", "comparison")) == "primary":
        return [_report_primary_method(config, methods)]
    return list(methods)


def _ablation_probability_hover_details(row: dict) -> list[str]:
    """Build per-response likelihood tooltips from qualitative ablation rescoring."""
    table = row.get("qualitative_probability_table", [])
    if not isinstance(table, list) or not table:
        return []
    max_index = max((int(item.get("response_index", -1)) for item in table if isinstance(item, dict)), default=-1)
    details = [""] * (max_index + 1)
    for item in table:
        if not isinstance(item, dict):
            continue
        response_index = int(item.get("response_index", -1))
        if not 0 <= response_index < len(details):
            continue
        alternatives = item.get("alternatives", [])
        lines = []
        for alt in alternatives[:5] if isinstance(alternatives, list) else []:
            token_text = str(alt.get("token_text", "")).replace("▁", " ").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "↵\n")
            lines.append(
                f"{token_text!r}: src {float(alt.get('source', 0.0)):.3g}, "
                f"tgt {float(alt.get('target', 0.0)):.3g}, "
                f"edit {float(alt.get('edited', 0.0)):.3g}, "
                f"ablated {float(alt.get('edit_ablated', 0.0)):.3g}"
            )
        details[response_index] = "\n".join(lines)
    return details


def _ablation_probability_hover_callouts(row: dict) -> list[dict | None]:
    """Build callout-style Source/Target/Effect rows for each response token."""
    table = row.get("qualitative_probability_table", [])
    if not isinstance(table, list) or not table:
        return []
    max_index = max(
        (int(item.get("response_index", -1)) for item in table if isinstance(item, dict)),
        default=-1,
    )
    callouts: list[dict | None] = [None] * (max_index + 1)
    for item in table:
        if not isinstance(item, dict):
            continue
        response_index = int(item.get("response_index", -1))
        if not 0 <= response_index < len(callouts):
            continue
        alternatives = item.get("alternatives", [])
        if not isinstance(alternatives, list) or not alternatives:
            continue
        effect_rows = []
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            effect = dict(alternative)
            effect["effect"] = float(alternative.get("edited", 0.0) or 0.0) - float(
                alternative.get("edit_ablated", 0.0) or 0.0
            )
            effect_rows.append(effect)
        callouts[response_index] = {
            "heading_prefix": "Sink",
            "rows": [
                _callout_row("Source", alternatives, "source", kind="source"),
                _callout_row("Target", alternatives, "target", kind="target"),
                _callout_row("Change", effect_rows, "effect", kind="effect"),
            ]
        }
    return callouts


def _causal_event_profile_view(record: dict, ablation_rows: list[dict], *, method: str = "ig", limit: int = 5) -> str:
    """Render top total-response attribution events with their exact causal profiles."""
    rows = [
        row for row in ablation_rows
        if any(
            str(source) == f"{method}_signed_top"
            for source in row.get("selection_sources", [])
        )
    ]
    rows.sort(
        key=lambda row: float(row.get(f"{method}_total_attribution_credit", 0.0) or 0.0),
        reverse=True,
    )
    panels = []
    for row in rows[: int(limit)]:
        index = int(row.get("edit_index", -1))
        token_text = str(row.get("token_text", ""))
        credit = row.get(f"{method}_total_attribution_credit")
        effect = row.get("response_token_kl_increase", [])
        prompt_len = int(record.get("prompt_tokens", 0))
        full_effect = [0.0] * prompt_len + [float(value) for value in effect]
        causal_view = _token_viewer(
            record,
            full_effect,
            label="KL change after zeroing edit event",
            response_only=False,
            selected_index=index,
            legend_positive="removal worsens match",
            legend_negative="removal improves match",
            hover_details=_ablation_probability_hover_details(row),
            hover_callouts=_ablation_probability_hover_callouts(row),
            include_default_support_changes=False,
        )
        panels.append(
            '<section class="event-card">'
            '<div class="event-head">'
            f'<div><strong>Edit event at token {index}</strong> <code>{html.escape(token_text.replace(chr(10), "↵"))}</code></div>'
            f'<div><span class="pill">{html.escape(method.upper())} total credit {_number(credit)}</span></div>'
            '</div>'
            f'<div class="metric-row"><span>Mean KL increase</span><strong>{_number(row.get("mean_kl_increase"))}</strong>'
            f'<span>|edit|</span><strong>{_number(row.get("direct_edit_norm"))}</strong></div>'
            f'{causal_view}'
            f'<details><summary>Top affected phrases</summary>{_effect_span_html(row.get("downstream_effect_spans", {}))}</details>'
            '</section>'
        )
    if not panels:
        return '<p class="muted">No teacher-forced causal profiles were available for the selected IG events.</p>'
    return "".join(panels)


def _scatter(rows: list[dict], *, x_key: str, y_key: str, title: str, x_label: str = "attribution credit", selection_prefix: str | None = None) -> str:
    points = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None and math.isfinite(float(r[x_key])) and math.isfinite(float(r[y_key]))]
    if not points:
        return '<p class="muted">No paired observations.</p>'
    xs = np.asarray([float(r[x_key]) for r in points], dtype=np.float64)
    ys = np.asarray([float(r[y_key]) for r in points], dtype=np.float64)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmin == xmax: xmin, xmax = xmin - 1.0, xmax + 1.0
    if ymin == ymax: ymin, ymax = ymin - 1.0, ymax + 1.0
    width, height, left, top, right, bottom = 760, 440, 88, 34, 24, 66
    pw, ph = width-left-right, height-top-bottom
    px=lambda x:left+(x-xmin)/(xmax-xmin)*pw
    py=lambda y:top+(ymax-y)/(ymax-ymin)*ph
    ticks=[]
    for f in np.linspace(0,1,5):
        x=xmin+f*(xmax-xmin); y=ymin+f*(ymax-ymin)
        ticks.append(f'<text x="{px(x):.1f}" y="{height-34}" text-anchor="middle">{x:.3g}</text>')
        ticks.append(f'<text x="{left-10}" y="{py(y)+4:.1f}" text-anchor="end">{y:.3g}</text>')
    circles=[]
    for r in points:
        selected = (
            True if selection_prefix is None
            else any(str(source).startswith(selection_prefix) for source in r.get("selection_sources", []))
        )
        cls = "primary" if selected else "control"
        circles.append(f'<circle class="{cls}" cx="{px(float(r[x_key])):.2f}" cy="{py(float(r[y_key])):.2f}" r="3.2"/>')
    return f'''<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 {width} {height}" role="img">
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>
{''.join(ticks)}<g class="points">{''.join(circles)}</g><text x="{left+pw/2}" y="{height-5}" text-anchor="middle">{html.escape(x_label)}</text>
<text transform="translate(20 {top+ph/2}) rotate(-90)" text-anchor="middle">KL increase after zeroing edit event</text></svg>
<div class="plot-legend"><span class="dot primary"></span>top-attribution and random edit events</div></figure>'''


def _histogram(values: list[float], title: str) -> str:
    arr=np.asarray([float(v) for v in values if math.isfinite(float(v))],dtype=np.float64)
    if arr.size == 0: return '<p class="muted">No source-level values.</p>'
    counts, edges=np.histogram(arr,bins=min(18,max(6,int(np.sqrt(arr.size)))))
    width,height,left,top,right,bottom=760,300,60,30,20,50; pw=width-left-right; ph=height-top-bottom
    ymax=max(1,int(counts.max())); bars=[]
    for i,c in enumerate(counts):
        x=left+i/len(counts)*pw; w=pw/len(counts)-2; h=float(c)/ymax*ph
        bars.append(f'<rect x="{x+1:.1f}" y="{top+ph-h:.1f}" width="{max(1,w):.1f}" height="{h:.1f}"/>')
    return f'''<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 {width} {height}">
<line x1="{left}" y1="{top+ph}" x2="{width-right}" y2="{top+ph}"/><g class="hist">{''.join(bars)}</g>
<text x="{left}" y="{height-16}">{edges[0]:.2f}</text><text x="{width-right}" y="{height-16}" text-anchor="end">{edges[-1]:.2f}</text>
<text x="{left+pw/2}" y="{height-5}" text-anchor="middle">source-level mean within-record Spearman</text></svg></figure>'''

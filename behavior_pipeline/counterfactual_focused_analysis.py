from __future__ import annotations

"""Focused exhaustive validation and corpus text browsing."""

import html
import base64
import io
import math
import random
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .counterfactual_runtime_ablation import (
    _baseline_parity,
    _correlation_summary,
    _load_total_attribution,
    _per_response_correlation_summary,
    _record_with_matched_baseline,
    score_edits,
)
from .counterfactual_runtime_attribution import _attribution_filename
from .counterfactual_runtime_core import Runtime, read_json, write_json
from .topk_kl import mean_kl_increase
from .counterfactual_report_figures import (
    _paper_matplotlib, _add_scatter_headroom, _add_x_zero_line, _add_y_zero_line,
)
from .counterfactual_report_plot_style import PLOT_STYLE, SCATTER_COLORS


def select_random_record_paths(
    records_directory: Path,
    *,
    count: int,
    seed: int,
) -> list[Path]:
    """Select records independently of filesystem order and worker count."""
    paths = sorted(Path(records_directory).glob("*.pt"), key=lambda path: path.name)
    requested = max(0, int(count))
    if requested == 0 or requested >= len(paths):
        return paths
    return sorted(random.Random(int(seed)).sample(paths, requested), key=lambda path: path.name)


def partition_record_paths(paths: list[Path], count: int) -> list[list[Path]]:
    """Greedily balance selected records using sidecar token counts."""
    shard_count = max(1, int(count))
    groups: list[list[Path]] = [[] for _ in range(shard_count)]
    loads = [0 for _ in range(shard_count)]
    weighted = []
    for path in paths:
        metadata = read_json(path.with_suffix(".json"), {})
        cost = max(
            1,
            int(metadata.get("prompt_tokens", 0))
            + int(metadata.get("response_tokens", 0)),
        )
        weighted.append((cost, path))
    for cost, path in sorted(weighted, key=lambda item: (-item[0], item[1].name)):
        destination = min(range(shard_count), key=lambda index: (loads[index], index))
        groups[destination].append(path)
        loads[destination] += int(cost)
    return [sorted(group, key=lambda path: path.name) for group in groups]


def _validation_summary(rows: list[dict], methods: list[str]) -> dict:
    result = {}
    for method in methods:
        attribution_key = f"{method}_attribution_credit"
        result[method] = {
            "pooled": _correlation_summary(
                rows,
                predicted_key=attribution_key,
                observed_key="mean_kl_increase",
            ),
            "per_response": _per_response_correlation_summary(
                rows,
                predicted_key=attribution_key,
                observed_key="mean_kl_increase",
            ),
        }
    return result


def build_exhaustive_validation_artifact(
    rows: list[dict],
    *,
    methods: list[str],
    selected_records: list[dict],
    seed: int,
) -> dict:
    rows = sorted(
        rows,
        key=lambda row: (str(row["record_id"]), int(row["edit_index"])),
    )
    return {
        "analysis": "exhaustive_edit_event_attribution_ablation",
        "selection": "uniform_without_replacement_from_teacher_forced_records",
        "seed": int(seed),
        "records_selected": len(selected_records),
        "events_tested": len(rows),
        "effect_unit": "mean conditional KL increase per response token",
        "intervention": "Set one applied edit event z_t to zero while leaving all other edit events fixed.",
        "selected_records": selected_records,
        "validation_by_method": _validation_summary(rows, methods),
        "records": rows,
    }


def compute_exhaustive_edit_event_validation(
    runtime: Runtime,
    record_paths: list[Path],
    attribution_dir: Path,
    output_path: Path,
    config: dict,
) -> dict:
    """Ablate every edit event for a small predefined record sample."""
    methods = [str(value) for value in config["attribution"].get("methods", ["ig"])]
    total_maps = {
        method: _load_total_attribution(
            attribution_dir / _attribution_filename(method, "total_response_credit.json")
        )
        for method in methods
    }
    batch_size = int(config["exhaustive_validation"].get("ablation_batch_size", 16))
    support_only = bool(config["ablations"].get("support_only_projection", True))
    projection_chunk = int(config["ablations"].get("support_projection_chunk_size", 2048))
    rows: list[dict] = []
    selected_records: list[dict] = []

    for path in tqdm(record_paths, desc="exhaustive token ablations", unit="record"):
        record = torch.load(path, map_location="cpu")
        record_id = str(record["record_id"])
        attribution_rows = {method: total_maps[method].get(record_id) for method in methods}
        if any(row is None for row in attribution_rows.values()):
            missing = [method for method, row in attribution_rows.items() if row is None]
            raise RuntimeError(
                f"Missing total-response attribution for {record_id}: {', '.join(missing)}"
            )
        sequence = int(record["edit"].shape[1])
        attribution_values = {}
        for method, attribution_row in attribution_rows.items():
            values = torch.tensor(attribution_row["edit_event_credit"], dtype=torch.float32)
            if values.ndim != 1 or int(values.numel()) != sequence:
                raise RuntimeError(
                    f"{method} attribution length does not match record {record_id}: "
                    f"{int(values.numel())} != {sequence}"
                )
            attribution_values[method] = values

        baseline_edit = record["edit"].float()
        baseline_scores = score_edits(
            runtime,
            record,
            baseline_edit,
            support_only_projection=support_only,
            support_projection_chunk_size=projection_chunk,
        )
        baseline = {key: value[0] for key, value in baseline_scores.items()}
        matched_record = _record_with_matched_baseline(record, baseline)
        parity = _baseline_parity(record, matched_record)
        edit_norms = baseline_edit[0].norm(dim=-1)
        prompt_tokens = int(record["prompt_tokens"])

        selected_records.append(
            {
                "record_id": record_id,
                "source_id": str(record["source_id"]),
                "response_id": str(record["response_id"]),
                "direction": str(record["direction"]),
                "tokens": sequence,
            }
        )
        for start in range(0, sequence, batch_size):
            indices = list(range(start, min(sequence, start + batch_size)))
            edits = baseline_edit.expand(len(indices), -1, -1).clone()
            for local_index, edit_index in enumerate(indices):
                edits[local_index, edit_index, :] = 0.0
            scores = score_edits(
                runtime,
                record,
                edits,
                support_only_projection=support_only,
                support_projection_chunk_size=projection_chunk,
            )
            for local_index, edit_index in enumerate(indices):
                token = record["tokens"][edit_index]
                loss = scores["loss"][local_index].float()
                row = {
                    "record_id": record_id,
                    "source_id": str(record["source_id"]),
                    "response_id": str(record["response_id"]),
                    "direction": str(record["direction"]),
                    "edit_index": int(edit_index),
                    "region": "prompt" if edit_index < prompt_tokens else "response",
                    "region_index": (
                        int(edit_index)
                        if edit_index < prompt_tokens
                        else int(edit_index - prompt_tokens)
                    ),
                    "token_id": int(token["token_id"]),
                    "token_text": str(token["token_text"]),
                    "direct_edit_norm": float(edit_norms[edit_index]),
                    "mean_kl_increase": float(
                        mean_kl_increase(matched_record["edited_kl"], loss)
                    ),
                    **parity,
                }
                for method in methods:
                    row[f"{method}_attribution_credit"] = float(
                        attribution_values[method][edit_index]
                    )
                rows.append(row)

    artifact = build_exhaustive_validation_artifact(
        rows,
        methods=methods,
        selected_records=selected_records,
        seed=int(config["run"]["seed"]),
    )
    write_json(output_path, artifact)
    return artifact


def merge_exhaustive_validation_shards(
    shard_paths: list[Path],
    output_path: Path,
    config: dict,
) -> dict:
    methods = [str(value) for value in config["attribution"].get("methods", ["ig"])]
    rows: list[dict] = []
    selected: list[dict] = []
    for path in shard_paths:
        payload = read_json(path, {})
        rows.extend(payload.get("records", []))
        selected.extend(payload.get("selected_records", []))
    artifact = build_exhaustive_validation_artifact(
        rows,
        methods=methods,
        selected_records=selected,
        seed=int(config["run"]["seed"]),
    )
    write_json(output_path, artifact)
    return artifact


_SENTENCE_PATTERN = re.compile(
    r".+?(?:[.!?]+(?:[\"'\)\]]+)?(?=\s|$)|$)",
    flags=re.DOTALL,
)


def split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", str(text).replace("▁", " ")).strip()
    if not collapsed:
        return []
    return [match.group(0).strip() for match in _SENTENCE_PATTERN.finditer(collapsed) if match.group(0).strip()]


def build_sentence_collection(
    record_sidecars: list[Path],
    *,
    keywords: list[str],
    seed: int,
) -> dict:
    escaped = [re.escape(str(keyword)) for keyword in keywords]
    matcher = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
    rows = []
    sampled_records = []
    for example_number, path in enumerate(record_sidecars, 1):
        record = read_json(path, {})
        prompt_tokens = int(record.get("prompt_tokens", 0))
        tokens = list(record.get("tokens", []))
        response_text = "".join(
            str(token.get("token_text", "")) for token in tokens[prompt_tokens:]
        )
        sampled_records.append(
            {
                "example_number": int(example_number),
                "record_id": str(record.get("record_id", path.stem)),
                "source_id": str(record.get("source_id", "")),
                "direction": str(record.get("direction", "")),
            }
        )
        for sentence_index, sentence in enumerate(split_sentences(response_text)):
            matches = sorted({match.group(0).lower() for match in matcher.finditer(sentence)})
            if matches:
                rows.append(
                    {
                        "example_number": int(example_number),
                        "record_id": str(record.get("record_id", path.stem)),
                        "source_id": str(record.get("source_id", "")),
                        "response_id": str(record.get("response_id", "")),
                        "direction": str(record.get("direction", "")),
                        "sentence_index": int(sentence_index),
                        "keywords": matches,
                        "sentence": sentence,
                    }
                )
    return {
        "analysis": "keyword_sentence_browser",
        "selection": "uniform_without_replacement_from_teacher_forced_records",
        "seed": int(seed),
        "keywords": list(keywords),
        "records_sampled": len(record_sidecars),
        "sentences_matched": len(rows),
        "sampled_records": sampled_records,
        "sentences": rows,
    }


def _number(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.4f}"


def exhaustive_validation_plot_groups(exhaustive: dict) -> list[dict]:
    """Use the same finite event pairs for each plot and its own Spearman."""
    rows = list(exhaustive.get("records", []))
    record_ids = list(dict.fromkeys(
        [str(row["record_id"]) for row in exhaustive.get("selected_records", [])]
        + [str(row["record_id"]) for row in rows]
    ))
    groups = []
    for method in exhaustive.get("validation_by_method", {}):
        key = f"{method}_attribution_credit"
        populations = [("Pooled", rows)] + [
            (f"Example {index + 1} · {record_id}",
             [row for row in rows if str(row["record_id"]) == record_id])
            for index, record_id in enumerate(record_ids)
        ]
        for label, population in populations:
            points = [
                row for row in population
                if row.get(key) is not None and row.get("mean_kl_increase") is not None
                and math.isfinite(float(row[key]))
                and math.isfinite(float(row["mean_kl_increase"]))
            ]
            correlation = _correlation_summary(
                points, predicted_key=key, observed_key="mean_kl_increase"
            )
            groups.append({
                "method": method, "label": label, "points": points,
                "rho": correlation.get("spearman"), "x_key": key,
            })
    return groups


def _exhaustive_scatter_image(group: dict, model_label: str = "") -> str:
    plt, _ = _paper_matplotlib()
    if plt is None:
        return '<p class="muted">Plot unavailable: Matplotlib could not be loaded.</p>'
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    try:
        ax.scatter(
            [float(row[group["x_key"]]) for row in group["points"]],
            [float(row["mean_kl_increase"]) for row in group["points"]],
            s=PLOT_STYLE["scatter_size"], alpha=PLOT_STYLE["scatter_alpha"],
            color=SCATTER_COLORS["attribution"], linewidths=0, rasterized=True,
        )
        method = "IG" if group["method"] == "ig" else "RelP"
        ax.set_xlabel(f"{method} Attribution Score", fontsize=11)
        ax.set_ylabel("KL Change after Ablation", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        _add_scatter_headroom(ax)
        # Keep 3–4 readable major ticks inside the displayed range, without
        # allowing the locator or set_xticks to expand that range.
        from matplotlib.ticker import MaxNLocator, FixedLocator
        xmin, xmax = ax.get_xlim()
        ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(xmin, xmax)
        ticks = [float(value) for value in ticks if xmin <= value <= xmax]
        if not 3 <= len(ticks) <= 4:
            ticks = [xmin + (xmax - xmin) * index / 3 for index in range(4)]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        rho = group["rho"]
        value = f"{float(rho):.2f}" if rho is not None and math.isfinite(float(rho)) else "—"
        ax.text(
            0.96, 0.96, f"ρ = {value}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#AEB7C2", "alpha": 0.92}, zorder=5,
        )
        _add_x_zero_line(ax)
        _add_y_zero_line(ax)
        ax.set_title("Attribution vs. Causal Effect", fontsize=12, fontweight="bold")
        if model_label:
            from matplotlib.transforms import ScaledTranslation
            ax.set_title("Attribution vs. Causal Effect", fontsize=12, fontweight="bold", pad=26)
            ax.text(.5, 1, model_label, ha="center", va="bottom", fontsize=9,
                    transform=ax.transAxes + ScaledTranslation(0, 12/72, fig.dpi_scale_trans))
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="svg", dpi=PLOT_STYLE["dpi"])
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        label = html.escape(f'{method}: {group["label"]}', quote=True)
        return f'<img alt="{label}" src="data:image/svg+xml;base64,{encoded}">'
    finally:
        plt.close(fig)


def render_exhaustive_validation_html(exhaustive: dict, output_path: Path, *, model_label: str = "") -> None:
    if not model_label:
        location = str(output_path).lower()
        model_label = "DeepSeek" if "deepseek" in location else "Gemma" if "gemma" in location else ""
    model_label = {"Gemma": "Gemma-4-E4B-it", "DeepSeek": "DeepSeek-R1-0528-Qwen3-8B"}.get(model_label, model_label)
    figures = []
    for group in exhaustive_validation_plot_groups(exhaustive):
        label = html.escape(f'{str(group["method"]).upper()} · {group["label"]}')
        figures.append(
            f'<figure><figcaption>{label}</figcaption>'
            + _exhaustive_scatter_image(group, model_label) + '</figure>'
        )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exhaustive token validation</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fa;color:#17202a;font:14px/1.5 Inter,system-ui,sans-serif}}
main{{max-width:1300px;margin:32px auto;padding:0 24px 70px}}
.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:18px}}
figure{{margin:0;padding:14px;background:#fff;border:1px solid #d9e0e7;border-radius:8px}}
figcaption{{font-weight:700;overflow-wrap:anywhere}}img{{display:block;width:100%;height:auto}}.muted{{color:#667384}}
</style></head><body><main><h1>Exhaustive token validation</h1>
<p class="muted">Every edit event is causally ablated for a fixed random sample of {int(exhaustive.get('records_selected', 0))} responses. Each plot reports its own Spearman correlation.</p>
<h2>Attribution versus ablation</h2><div class="plots">{''.join(figures)}</div>
</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def render_sentence_browser_html(sentences: dict, output_path: Path) -> None:
    keywords = [str(value) for value in sentences.get("keywords", [])]
    keyword_matcher = re.compile(
        r"\b(?:" + "|".join(re.escape(value) for value in keywords) + r")\b",
        flags=re.IGNORECASE,
    )

    def highlighted(value: str) -> str:
        value = value.replace("▁", " ")
        pieces = []
        start = 0
        for match in keyword_matcher.finditer(value):
            pieces.append(html.escape(value[start : match.start()]))
            pieces.append("<mark>" + html.escape(match.group(0)) + "</mark>")
            start = match.end()
        pieces.append(html.escape(value[start:]))
        return "".join(pieces)

    sentence_rows = []
    for row in sentences.get("sentences", []):
        sentence_rows.append(
            '<article class="sentence" data-text="'
            + html.escape(str(row.get("sentence", "")).replace("▁", " ").lower(), quote=True)
            + '" data-direction="'
            + html.escape(str(row.get("direction", "")), quote=True)
            + '" data-keywords="'
            + html.escape(" ".join(row.get("keywords", [])), quote=True)
            + '"><div class="meta">Example '
            + str(int(row.get("example_number", 0)))
            + " · "
            + html.escape(str(row.get("direction", "")).replace("_to_", " → "))
            + " · sentence "
            + str(int(row.get("sentence_index", 0)) + 1)
            + '</div><p>'
            + highlighted(str(row.get("sentence", "")))
            + "</p></article>"
        )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Keyword sentence browser</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fa;color:#17202a;font:14px/1.5 Inter,system-ui,sans-serif}}
main{{max-width:1100px;margin:32px auto;padding:0 24px 70px}}h1{{font-size:30px}}.muted{{color:#667384}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}}input,select{{border:1px solid #b9c3ce;border-radius:6px;padding:8px 10px;background:#fff;font:inherit}}input{{min-width:280px;flex:1}}
.sentence{{background:#fff;border:1px solid #d9e0e7;border-radius:8px;padding:14px;margin:8px 0}}.sentence p{{font-size:15px;margin:5px 0 0}}.meta{{font-size:12px;color:#667384}}mark{{background:#ffe59a;color:inherit;border-radius:2px;padding:0 1px}}
@media(max-width:600px){{main{{padding:0 12px 50px}}input{{min-width:100%}}}}
</style></head><body><main><h1>Keyword sentence browser</h1>
<p class="muted">Every matching decoded response sentence from {int(sentences.get('records_sampled', 0))} randomly selected {'unique intervention-arm responses' if sentences.get('response_arm') == 'intervention' else 'teacher-forced records'}.</p>
<div class="controls"><input id="search" type="search" placeholder="Search matching sentences"><select id="direction"><option value="">Both directions</option><option value="c_to_i">Control → intervention</option><option value="i_to_c">Intervention → control</option></select><select id="keyword"><option value="">All keywords</option>{''.join(f'<option>{html.escape(keyword)}</option>' for keyword in sentences.get('keywords', []))}</select></div>
<p class="muted"><span id="visible-count">{len(sentence_rows)}</span> of {len(sentence_rows)} matching sentences</p><section>{''.join(sentence_rows)}</section>
<script>const search=document.getElementById('search'),direction=document.getElementById('direction'),keyword=document.getElementById('keyword'),items=[...document.querySelectorAll('.sentence')];function filter(){{const q=search.value.toLowerCase(),dir=direction.value,key=keyword.value.toLowerCase();let count=0;items.forEach(item=>{{const show=(!q||item.dataset.text.includes(q))&&(!dir||item.dataset.direction===dir)&&(!key||item.dataset.keywords.split(' ').includes(key));item.hidden=!show;if(show)count++}});document.getElementById('visible-count').textContent=count}}[search,direction,keyword].forEach(node=>node.addEventListener('input',filter));</script>
</main></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def build_keyword_sentence_browser(
    records_directory: Path,
    output_json: Path,
    *,
    count: int,
    keywords: list[str],
    seed: int,
) -> dict:
    sidecars = sorted(Path(records_directory).glob("*.json"), key=lambda path: path.name)
    # A response can appear under both edit directions; sample responses once.
    intervention_sidecars = []
    seen_responses = set()
    for path in sidecars:
        record = read_json(path, {})
        response_id = str(record.get("response_id", ""))
        arm = str(record.get("response_arm", ""))
        if not arm and ":intervention:" in response_id:
            arm = "intervention"
        if arm != "intervention":
            continue
        key = response_id or str(record.get("record_id", path.stem))
        if key in seen_responses:
            continue
        seen_responses.add(key)
        intervention_sidecars.append(path)
    sidecars = intervention_sidecars
    requested = max(0, int(count))
    if requested and requested < len(sidecars):
        sidecars = random.Random(int(seed)).sample(sidecars, requested)
    sidecars = sorted(sidecars, key=lambda path: path.name)
    artifact = build_sentence_collection(sidecars, keywords=keywords, seed=seed)
    artifact["selection"] = "uniform_without_replacement_from_unique_intervention_responses"
    artifact["response_arm"] = "intervention"
    artifact["records_requested"] = requested
    artifact["records_available"] = len(intervention_sidecars)
    write_json(output_json, artifact)
    return artifact

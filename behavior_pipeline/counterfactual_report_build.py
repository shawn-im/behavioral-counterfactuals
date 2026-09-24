from __future__ import annotations

from .native_hypothesis_sites import PATCH_SITE_ORDER, PATCH_PLOT_LABELS

"""Internal build implementation for :mod:`behavior_pipeline.counterfactual_report`."""

import html
import math
import shutil
from collections import defaultdict
from pathlib import Path

from .counterfactual_runtime import (
    load_record_summaries,
    read_json,
    write_json,
)
from .counterfactual_report_html import _causal_event_profile_view, _case_records, _histogram, _method_label, _number, _page, _percent, _report_primary_method, _report_visible_methods, _scatter, _token_viewer, _write
from .counterfactual_report_presentation import example_table, direction_badge, edited_support_distribution, pooled_unedited_support_distribution
from .counterfactual_report_figures import (
    _activation_recovery_value,
    _answer_accuracy_summary,
    _recovery_audit,
    _write_answer_accuracy_figure,
    _write_attribution_ablation_figure,
    _write_edit_norm_ablation_figure,
    _write_objective_comparison_figures,
    _write_recovery_figure,
    _write_support_coverage_figure,
    _top_edit_attribution_tokens,
    _write_attribution_correlation_distribution_figure,
    _write_causal_effect_size_comparison_figure,
    _write_attribution_method_correlation_figure,
)
from .native_hypothesis_report import _metric_value, write_native_quantitative_figures


def _is_evaluation_event(row: dict, method: str) -> bool:
    evaluation_method = row.get("evaluation_method")
    if evaluation_method is not None:
        return str(evaluation_method) == str(method)
    metadata = row.get("selection_groups", {}).get(str(method), {})
    return metadata.get("evaluation_group", metadata.get("group")) in {"top", "random"}


def _native_patch_hover_details(source: dict, target: dict, condition: dict) -> list[str]:
    """Build per-response likelihood tooltips with the five most diagnostic alternatives."""
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
    rows = []
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
                token_text = str(alt.get("token_text", "")).replace("▁", " ").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "↵\n")
                parts.append(
                    f"{token_text!r}: src {float(alt.get('source', 0.0)):.3g}, "
                    f"tgt {float(alt.get('target', 0.0)):.3g}, "
                    f"patch {float(alt.get('native_patch', 0.0)):.3g}"
                )
        rows.append("\n".join(parts))
    return rows


def _native_patch_hover_callouts(condition: dict) -> list[dict | None]:
    """Build callout-style Source/Target/Native patch rows per response token."""
    table = condition.get("qualitative_probability_table", [])
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
        alternatives = item.get("alternatives", [])
        if not 0 <= response_index < len(callouts) or not isinstance(alternatives, list) or not alternatives:
            continue

        def row(label: str, key: str, kind: str) -> dict:
            values = [
                {
                    "token": str(alt.get("token_text", "")).replace("▁", " ").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "↵"),
                    "value": float(alt.get(key, 0.0) or 0.0),
                }
                for alt in alternatives
                if isinstance(alt, dict)
            ]
            values.sort(key=lambda value: value["value"], reverse=True)
            return {"label": label, "kind": kind, "items": values[:3]}

        callouts[response_index] = {
            "heading_prefix": "Sink",
            "rows": [
                row("Source", "source", "source"),
                row("Target", "target", "target"),
                row("Native patch", "native_patch", "patch"),
            ]
        }
    return callouts


def _top_token_table_text(text: str) -> str:
    return "".join(
        '<span class="top-token-whitespace">' + html.escape(char) + '</span>'
        if char in " \t" else html.escape(char).replace("\n", "↵")
        for char in text.replace("▁", " ").replace("\r\n", "\n").replace("\r", "\n")
    )


def build_report(config: dict, run_dir: Path) -> dict:
    layer = config.get("model", {}).get("layer")
    layer_label = f"layer-{int(layer)}" if layer is not None else "selected-layer"
    report=run_dir/"report"; assets=report/"assets"; examples=report/"examples"; data=report/"data"
    for directory in (assets, examples, data):
        if directory.exists():
            shutil.rmtree(directory)
    for d in (assets,examples,data): d.mkdir(parents=True,exist_ok=True)
    for path in report.glob("*.html"):
        path.unlink()
    _write(assets/"report.css", r'''
:root{color-scheme:light;--ink:#17202a;--muted:#667384;--line:#d9e0e7;--panel:#fff;--bg:#f5f7fa;--accent:#176b87;--pos:#D65F5F;--neg:#4C78A8;--control:#8B95A1}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.top{position:sticky;top:0;z-index:20;height:58px;background:rgba(255,255,255,.95);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:0 24px;display:flex;justify-content:space-between;align-items:center}.top nav{display:flex;gap:18px}.top a,a{color:var(--accent);text-decoration:none}main{max-width:1500px;margin:34px auto;padding:0 26px 80px;min-width:0}h1{font-size:32px;margin:0 0 22px}h2{font-size:22px;margin:38px 0 14px}h3{font-size:17px;margin:24px 0 10px}h4{font-size:14px;margin:12px 0 8px}h5{font-size:12px;margin:10px 0 4px}.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.card,.panel,.event-card,figure{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 1px 2px rgba(15,23,42,.035)}.card{padding:15px 17px}.card strong{display:block;font-size:23px;margin-top:5px}.pill{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:3px 8px;margin:2px 4px 2px 0;background:#fafbfd;white-space:nowrap}.method-status{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 18px}.method-status strong{font-size:13px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px;background:#fff;max-height:65vh}table{width:100%;border-collapse:collapse;min-width:640px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#eef3f6;position:sticky;top:0;z-index:1}.primary-row{font-weight:650;background:#fbfdff}.viewer{background:#fff;border:1px solid var(--line);border-radius:9px;min-width:0;overflow:hidden}.viewer-toolbar{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;padding:7px 10px;border-bottom:1px solid var(--line);background:#fafbfd}.viewer-text{padding:14px;max-height:390px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;word-break:normal;line-height:1.85;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}.tok{display:inline;white-space:inherit;border-radius:3px;box-decoration-break:clone;-webkit-box-decoration-break:clone;cursor:default}.tok.prompt{border-bottom:1px solid rgba(100,116,139,.28)}.tok.selected{outline:2px solid #111827;outline-offset:1px}.legend{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:11px}.legend-gradient{display:inline-block;width:90px;height:7px;border-radius:9px;background:linear-gradient(90deg,var(--neg),#fff,var(--pos));border:1px solid #e5e7eb}.pos-swatch,.neg-swatch{width:8px;height:8px;border-radius:50%}.pos-swatch{background:var(--pos)}.neg-swatch{background:var(--neg)}.scale-note{color:var(--muted);font-size:11px}.scale-note select{margin-left:4px;padding:3px 5px;border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--ink);font:inherit}.expand{border:1px solid var(--line);border-radius:7px;background:#fff;padding:4px 8px;cursor:pointer}.viewer.fullscreen{position:fixed;inset:12px;z-index:100;background:#fff;display:flex;flex-direction:column;box-shadow:0 20px 55px rgba(15,23,42,.25)}.viewer.fullscreen .viewer-text{max-height:none;flex:1;font-size:14px}.event-card{padding:15px;margin:14px 0;min-width:0}.event-head{display:flex;justify-content:space-between;align-items:start;gap:12px;flex-wrap:wrap}.explanation-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;min-width:0}.explanation-grid>div{min-width:0}.metric-row{display:flex;gap:8px 14px;align-items:baseline;flex-wrap:wrap;margin:8px 0;color:var(--muted)}.metric-row strong{color:var(--ink)}details{margin-top:8px}summary{cursor:pointer;color:var(--accent)}.compact-list{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 10px;padding:0;list-style:none}.compact-list li{display:contents}.compact-list code{overflow-wrap:anywhere}.empty-view{height:120px;border:1px dashed var(--line);border-radius:9px;padding:16px;color:var(--muted);background:#fafbfd}.sink-select{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.sink-select select{max-width:100%;padding:6px 9px;border:1px solid var(--line);border-radius:8px;background:#fff}figure{padding:14px;margin:16px 0;max-width:860px}figcaption{font-weight:700;margin-bottom:8px}.paper-figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}.paper-figures figure{margin:0;max-width:none}.paper-figure{width:100%;height:auto;display:block}svg{width:100%;height:auto}svg line{stroke:#4b5563;stroke-width:1}svg text{font-size:11px;fill:#4b5563}.points .primary{fill:#2F6B8A;opacity:.38}.points .control{fill:#8B95A1;opacity:.30}.hist rect{fill:#2F6B8A;opacity:.58}.plot-legend{font-size:11px;color:var(--muted);display:flex;gap:7px;align-items:center;flex-wrap:wrap}.dot{width:8px;height:8px;border-radius:50%;display:inline-block}.dot.primary{background:#2F6B8A}.dot.control{background:#8B95A1}.note{border-left:3px solid #d3a52b;background:#fffdf5;padding:10px 13px;margin:12px 0}.diagnostics{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.diag{background:#fff;border:1px solid var(--line);border-radius:9px;padding:14px}.diag h4{margin-top:0}.case-list{columns:2;column-gap:30px}.case-list li{break-inside:avoid;margin:5px 0}.phrase-cell{white-space:normal;overflow-wrap:anywhere}.tiny{font-size:11px;color:var(--muted)}.viewer-actions{display:flex;gap:6px}.expand{border:1px solid var(--line);border-radius:7px;background:#fff;padding:4px 8px;cursor:pointer}code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.nowrap{white-space:nowrap}@media(max-width:1050px){.workspace-layout{grid-template-columns:1fr}.workspace-sidebar{position:static;max-height:none}.workspace-filters{grid-template-columns:1fr 1fr}}@media(max-width:900px){.workspace-filters{grid-template-columns:1fr}.membership{grid-template-columns:1fr}.explanation-grid{grid-template-columns:1fr}.case-list{columns:1}main{padding:0 14px 60px}.top{padding:0 14px}.viewer-text{max-height:330px}.top nav{gap:10px;font-size:12px}}
''' + r'''
.tok,.tok.prompt,.tok *{border-bottom:0;text-decoration:none}
.top-token-text{white-space:pre-wrap}.top-token-whitespace{text-decoration:underline;text-decoration-skip-ink:none}
.token-callout{position:fixed;z-index:250;width:min(560px,calc(100vw - 24px));padding:12px 13px;background:#fff;border:2px solid #d6b631;border-radius:8px;box-shadow:0 14px 34px rgba(15,23,42,.2);pointer-events:none}.token-callout[hidden]{display:none}.token-callout-head{display:flex;justify-content:space-between;gap:14px;align-items:baseline;padding-bottom:8px;margin-bottom:8px;border-bottom:1px solid #e5e7eb}.token-callout-title{font:650 13px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}.token-callout-metric{color:var(--muted);font-size:11px;text-align:right;white-space:nowrap}.token-callout-grid{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:7px 10px;align-items:start}.token-callout-label{font-size:12px;font-weight:700;padding-top:4px}.token-callout-label.source{color:#35658f}.token-callout-label.target{color:#a44040}.token-callout-label.effect,.token-callout-label.patch,.token-callout-label.edit{color:#846b08}.token-callout-items{display:flex;gap:5px;flex-wrap:wrap;min-width:0}.token-callout-chip{display:inline-flex;gap:5px;align-items:baseline;max-width:100%;padding:3px 6px;border:1px solid #d9e0e7;border-radius:5px;background:#fafbfd;font:11.5px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.token-callout-chip-token{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:150px}.token-callout-chip-value{color:#596675;white-space:nowrap}.token-callout-note{white-space:pre-wrap;font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--muted)}.tok[data-callout]:hover,.tok[data-callout]:focus{outline:2px solid #d6b631;outline-offset:1px}.tok[data-callout]:focus{position:relative;z-index:1}@media(max-width:620px){.token-callout{width:calc(100vw - 16px);padding:10px}.token-callout-head{display:block}.token-callout-metric{text-align:left;margin-top:3px;white-space:normal}.token-callout-grid{grid-template-columns:1fr}.token-callout-label{padding-top:2px}}
''')
    _write(assets/"report.js", r'''
(()=>{
const hexRgb=h=>{const s=h.replace('#','');return [parseInt(s.slice(0,2),16),parseInt(s.slice(2,4),16),parseInt(s.slice(4,6),16)]};
const mix=(hex,a)=>{const [r,g,b]=hexRgb(hex);return `rgba(${r},${g},${b},${a})`};
function heatColor(value,scale){if(!Number.isFinite(value)||!Number.isFinite(scale)||scale<=0||value===0)return'transparent';const a=Math.min(.78,Math.max(.025,Math.abs(value)/scale*.72));return mix(value>0?'#be2d37':'#235fb4',a)}
function restyleViewer(viewer,mode){const scale=Number(mode==='full'?viewer.dataset.scaleFull:viewer.dataset.scaleP95)||1e-15;viewer.querySelectorAll('.viewer-text .tok').forEach(t=>t.style.background=heatColor(Number(t.dataset.value||0),scale));const note=viewer.querySelector('.scale-value');if(note)note.textContent=`${mode==='full'?'max':'95th percentile'} |scale| = ${scale.toPrecision(4)}`}
document.addEventListener('change',e=>{const select=e.target.closest('.attribution-scale-mode');if(!select)return;const viewer=select.closest('.viewer');if(viewer)restyleViewer(viewer,select.value)});
document.addEventListener('click',e=>{const b=e.target.closest('.expand');if(!b)return;const v=document.getElementById(b.dataset.viewer);if(!v)return;v.classList.toggle('fullscreen');b.textContent=v.classList.contains('fullscreen')?'Close':'Expand';document.body.style.overflow=v.classList.contains('fullscreen')?'hidden':''});
const sink=document.getElementById('sink-select');if(sink)sink.addEventListener('change',e=>document.querySelectorAll('.sink-panel').forEach(p=>p.style.display=p.dataset.sink===e.target.value?'block':'none'));
document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.querySelectorAll('.viewer.fullscreen').forEach(v=>v.classList.remove('fullscreen'));document.querySelectorAll('.expand').forEach(b=>b.textContent='Expand');document.body.style.overflow=''}});

const callout=document.createElement('div');callout.className='token-callout';callout.hidden=true;callout.setAttribute('role','tooltip');document.body.appendChild(callout);
let activeToken=null;
const valueText=(value,kind)=>{const number=Number(value);if(!Number.isFinite(number))return'–';if(kind==='effect')return`${number>=0?'+':''}${(100*number).toFixed(2)}%`;return`${(100*number).toFixed(2)}%`};
function renderCallout(token){let data;try{data=JSON.parse(token.dataset.callout)}catch(_error){return}callout.replaceChildren();const head=document.createElement('div');head.className='token-callout-head';const title=document.createElement('div');title.className='token-callout-title';title.textContent=data.heading||'Token';const metric=document.createElement('div');metric.className='token-callout-metric';const metricValue=Number(data.metric_value);metric.textContent=`${data.metric_label||'Value'}: ${Number.isFinite(metricValue)?metricValue.toPrecision(4):'–'}`;head.append(title,metric);callout.appendChild(head);const rows=Array.isArray(data.rows)?data.rows:[];if(rows.length){const grid=document.createElement('div');grid.className='token-callout-grid';rows.forEach(row=>{const label=document.createElement('div');label.className=`token-callout-label ${row.kind||''}`;label.textContent=row.label||'';const items=document.createElement('div');items.className='token-callout-items';(Array.isArray(row.items)?row.items:[]).forEach(item=>{const chip=document.createElement('span');chip.className='token-callout-chip';const word=document.createElement('span');word.className='token-callout-chip-token';word.textContent=item.token||'∅';const value=document.createElement('span');value.className='token-callout-chip-value';value.textContent=valueText(item.value,row.kind);chip.append(word,value);items.appendChild(chip)});grid.append(label,items)});callout.appendChild(grid)}else if(data.note){const note=document.createElement('div');note.className='token-callout-note';note.textContent=data.note;callout.appendChild(note)}callout.hidden=false;activeToken=token;positionCallout()}
function positionCallout(){if(!activeToken||callout.hidden)return;const anchor=activeToken.getBoundingClientRect();const gap=10;const margin=8;const box=callout.getBoundingClientRect();let left=anchor.right+gap;if(left+box.width>innerWidth-margin)left=anchor.left-box.width-gap;if(left<margin)left=Math.min(Math.max(margin,anchor.left),innerWidth-box.width-margin);let top=anchor.top+(anchor.height-box.height)/2;if(top<margin)top=margin;if(top+box.height>innerHeight-margin)top=Math.max(margin,innerHeight-box.height-margin);callout.style.left=`${left}px`;callout.style.top=`${top}px`}
function hideCallout(){callout.hidden=true;activeToken=null}
document.addEventListener('pointerover',e=>{const token=e.target.closest('.tok[data-callout]');if(token)renderCallout(token)});
document.addEventListener('pointerout',e=>{const token=e.target.closest('.tok[data-callout]');if(token&&!token.contains(e.relatedTarget))hideCallout()});
document.addEventListener('focusin',e=>{const token=e.target.closest('.tok[data-callout]');if(token)renderCallout(token)});
document.addEventListener('focusout',e=>{if(e.target.closest('.tok[data-callout]'))hideCallout()});
document.addEventListener('click',e=>{const token=e.target.closest('.tok[data-callout]');if(token){renderCallout(token);e.stopPropagation()}else hideCallout()});
addEventListener('resize',positionCallout);document.addEventListener('scroll',positionCallout,true);


})();
''')

    teacher=read_json(run_dir/"teacher_forced"/"summary.json",{})
    attribution_config=config.get("attribution",{})
    attribution_methods=[str(value) for value in attribution_config.get("methods",["ig"])]
    primary_attribution_method=_report_primary_method(config,attribution_methods)
    attribution_view=str(config.get("report",{}).get("attribution_view","primary"))
    visible_attribution_methods=_report_visible_methods(config,attribution_methods)

    def method_file(method: str, base: str) -> Path:
        return run_dir/"attribution"/(base if method=="ig" else f"{method}_{base}")

    attribution_by_method={method: read_json(method_file(method,"total_response_credit.json"),{}) for method in attribution_methods}
    ablation=read_json(run_dir/"ablations"/"edit_event_ablation.json",{})
    methods_root=Path(str(config["data"].get("static_runs_dir","")))
    comparison_path=methods_root/"final_evaluation"/"test_comparison.json"
    comparison=read_json(comparison_path,{}) if comparison_path.is_file() else {}
    audit=_recovery_audit(config,run_dir,teacher,comparison)
    write_json(data/"recovery_audit.json",audit)

    current_sha=audit.get("checkpoint_sha256")
    comparison_methods=comparison.get("methods",[])
    current_method=next((r for r in comparison_methods if str((r.get("checkpoint_identity") or {}).get("sha256",""))==current_sha and current_sha),None)
    canonical_recovery=audit.get("canonical_test_recovery") if audit.get("canonical_test_recovery") is not None else teacher.get("recovery")
    validation_by_method=ablation.get("validation_by_method",{})
    native_summary=read_json(run_dir/"native_analysis"/"summary.json",{})

    teacher_records=load_record_summaries(run_dir/"teacher_forced") if (run_dir/"teacher_forced"/"records").exists() else []
    records_by_id={str(record["record_id"]):record for record in teacher_records}
    cases=_case_records(
        teacher_records,
        float(config["attribution"].get("qualitative_fraction", 0.01)),
        int(config.get("run", {}).get("seed", 0)),
    )
    rendered_case_ids=[str(record["record_id"]) for record in cases]

    top_events = int(config.get("ablations", {}).get("top_events_per_method", 5))
    random_events = int(config.get("ablations", {}).get("random_events_per_method", 5))
    selection_description = (
        f"the top {top_events} events by attribution magnitude plus {random_events} random non-top events per response"
    )

    main_cards=[("Canonical test recovery",canonical_recovery)]
    for method in visible_attribution_methods:
        method_validation=validation_by_method.get(method,{})
        main_cards.append((f"{method.upper()} median Spearman",method_validation.get("per_response_validation",{}).get("median_spearman")))
    cards=''.join(f'<div class="card"><span class="muted">{html.escape(n)}</span><strong>{_number(v)}</strong></div>' for n,v in main_cards)

    recovery_figure = _write_recovery_figure(
        comparison_methods, assets / "recovery_by_method.png"
    )
    attribution_figures = {}
    edit_norm_figures = {}
    for method in visible_attribution_methods:
        figure = _write_attribution_ablation_figure(
            ablation.get("records", []),
            method=method,
            validation=validation_by_method.get(method, {}),
            path=assets / f"attribution_vs_ablation_{method}.png",
        )
        if figure is not None:
            attribution_figures[method] = figure
        norm_figure = _write_edit_norm_ablation_figure(
            ablation.get("records", []),
            method=method,
            validation=validation_by_method.get(method, {}),
            path=assets / f"edit_norm_vs_ablation_{method}.png",
        )
        if norm_figure is not None:
            edit_norm_figures[method] = norm_figure
    correlation_distribution_figures = {}
    causal_effect_comparison_figures = {}
    for method in visible_attribution_methods:
        correlation_path = _write_attribution_correlation_distribution_figure(
            ablation,
            method,
            assets / f"attribution_correlation_distribution_{method}.png",
        )
        if correlation_path is not None:
            correlation_distribution_figures[method] = correlation_path
        effect_path = _write_causal_effect_size_comparison_figure(
            validation_by_method.get(method, {}),
            method=method,
            path=assets / f"causal_effect_size_comparison_{method}.png",
        )
        if effect_path is not None:
            causal_effect_comparison_figures[method] = effect_path
    correlation_figure = _write_attribution_method_correlation_figure(
        validation_by_method, visible_attribution_methods, assets / "attribution_method_correlation.png"
    )
    answer_accuracy = _answer_accuracy_summary(config)
    write_json(data / "answer_accuracy.json", answer_accuracy)
    answer_figure = _write_answer_accuracy_figure(
        answer_accuracy, assets / "answer_accuracy.png"
    )
    objective_path = Path(str(config.get("report", {}).get("objective_comparison_path", "")))
    objective_comparison = (
        read_json(objective_path, {})
        if str(objective_path) not in {"", "."} and objective_path.is_file()
        else {}
    )
    objective_behavior_figure, objective_activation_figure = _write_objective_comparison_figures(
        objective_comparison,
        behavior_path=assets / "objective_behavior_recovery.png",
        activation_path=assets / "objective_activation_nmse.png",
    )
    if objective_comparison:
        write_json(data / "objective_comparison.json", objective_comparison)
    support_distribution = read_json(
        run_dir / "teacher_forced" / "support_mass_distribution.json", {}
    )
    if not support_distribution:
        support_distribution = read_json(
            run_dir / "attribution" / "support_mass_distribution.json", {}
        )
    support_distribution = pooled_unedited_support_distribution(support_distribution)
    support_stats = {key: support_distribution.get(key) for key in ("p10", "minimum")}
    write_json(data / "conditional_support_mass_distribution.json", support_distribution)
    support_coverage_figure = _write_support_coverage_figure(
        support_distribution, assets / "conditional_support_coverage.png"
    ) if support_distribution else None
    # current_method is matched to the analysis checkpoint by SHA-256 above.
    edited_support = edited_support_distribution(current_method)
    write_json(data / "edited_support_mass_distribution.json", edited_support)
    edited_support_figure = _write_support_coverage_figure(
        edited_support, assets / "edited_support_coverage.png"
    ) if edited_support else None
    native_site_figures = write_native_quantitative_figures(assets, native_summary) if native_summary else {}
    token_fraction = float(config.get("report", {}).get("top_token_fraction", 0.05))
    token_limit = int(config.get("report", {}).get("top_token_limit", 30))
    token_rankings = _top_edit_attribution_tokens(
        teacher_records,
        attribution_by_method.get(primary_attribution_method, {}).get("records", []),
        top_fraction=token_fraction,
        limit=token_limit,
    )
    write_json(data / "top_edit_attribution_tokens.json", token_rankings)

    rows=[]
    for r in comparison_methods:
        is_current=bool(current_sha and str((r.get("checkpoint_identity") or {}).get("sha256",""))==current_sha)
        cls=' class="primary-row"' if is_current else ''
        label=_method_label(r.get("method","")) + (' · primary' if is_current else '')
        rows.append(f'<tr{cls}><td>{html.escape(label)}</td><td>{_number(r.get("parameters"),8)}</td><td>{_number(r.get("test_source_kl_mean"))}</td><td>{_number(r.get("test_edited_kl_mean"))}</td><td>{_number(r.get("test_recovery"))}</td><td>{_number(r.get("c_to_i_recovery"))}</td><td>{_number(r.get("i_to_c_recovery"))}</td><td>{_number(r.get("student_support_mass_p10"))}</td><td>{_number(r.get("student_support_mass_min"))}</td><td>{_number(r.get("best_checkpoint_epoch"),8)}</td></tr>')
    architecture=(
        '<h2>Architecture ablation</h2>'
        f'<p class="muted">Frozen manifest: {int(comparison.get("source_prompts",0))} sources · {int(comparison.get("sequence_direction_records",0))} sequence-direction records. Support p10 is the global tenth percentile of the edited model probability mass retained by the shared conditional support.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Method</th><th>Parameters</th><th>Source KL</th><th>Edited KL</th><th>Recovery</th><th>ctrl → intv</th><th>intv → ctrl</th><th>Support p10</th><th>Support min</th><th>Epoch</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        if comparison_methods else '<p class="muted">Final-evaluation comparison not found.</p>'
    )

    objective_rows = []
    for row in objective_comparison.get("objectives", []):
        objective_rows.append(
            '<tr>'
            f'<td>{html.escape(str(row.get("label", row.get("objective", ""))))}</td>'
            f'<td>{_number(row.get("selected_lr"))}</td>'
            f'<td>{_number(row.get("validation_metric"))}</td>'
            f'<td>{_number(row.get("test_behavior_recovery"))}</td>'
            f'<td>{_number(row.get("test_activation_nmse"))}</td>'
            '</tr>'
        )
    objective_table = (
        '<h2>Training-objective ablation</h2>'
        '<p class="muted">The architecture, residual injection site, data, optimizer schedule, and LR sweep are held fixed. Behavior reconstruction selects by validation KL recovery; activation reconstruction selects by validation NMSE.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Training objective</th><th>Selected LR</th><th>Validation selection metric</th><th>Test KL recovery</th><th>Test activation NMSE</th></tr></thead><tbody>'
        + ''.join(objective_rows) + '</tbody></table></div>'
        if objective_rows else ''
    )

    def token_table(rows: list[dict]) -> str:
        return ''.join(
            '<tr>'
            f'<td><code class="top-token-text">{_top_token_table_text(str(row.get("token_text", "")))}</code></td>'
            f'<td>{int(row.get("token_id", -1))}</td>'
            f'<td>{int(row.get("top_fraction_count", 0))}</td>'
            f'<td>{int(row.get("occurrences", 0))}</td>'
            f'<td>{_percent(row.get("top_fraction_rate"))}</td>'
            f'<td>{_number(row.get("signed_sum"))}</td>'
            f'<td>{_number(row.get("absolute_sum"))}</td>'
            '</tr>'
            for row in rows
        )

    token_pct = 100.0 * float(token_rankings.get("top_fraction", token_fraction))
    token_rankings_html = (
        '<h2>Top tokens by downstream IG attribution to the edit module</h2>'
        f'<p class="muted">These rankings use the applied edit-event IG credit for the total-response utility; they use only downstream applied-edit attribution. '
        f'The count ranking asks whether an occurrence is in the top {token_pct:g}% by |IG credit| within its sequence-direction record.</p>'
        f'<h3>Most often in the top {token_pct:g}%</h3>'
        '<div class="table-wrap"><table><thead><tr><th>Token</th><th>ID</th><th>Top-count</th><th>Occurrences</th><th>Top rate</th><th>Signed ΣIG</th><th>Σ|IG|</th></tr></thead><tbody>'
        + token_table(token_rankings.get("by_top_fraction_count", [])) + '</tbody></table></div>'
    ) if token_rankings.get("total_token_occurrences", 0) else ''

    case_table = example_table((str(r["record_id"]), str(r["direction"])) for r in cases)
    ig_steps=int(attribution_config.get("ig_steps",16))
    figure_cards = []
    if recovery_figure is not None:
        figure_cards.append('<figure><figcaption>Architecture ablation: held-out recovery</figcaption><img class="paper-figure" src="assets/architecture_recovery.png" alt="held-out recovery by method"></figure>')
    for method in visible_attribution_methods:
        if method in attribution_figures:
            figure_cards.append(f'<figure><figcaption>{html.escape(method.upper())} attribution versus exact edit-event ablation</figcaption><img class="paper-figure" src="assets/attribution_vs_ablation_{html.escape(method)}.png" alt="{html.escape(method)} attribution versus exact ablation"></figure>')
        if method in edit_norm_figures:
            figure_cards.append(f'<figure><figcaption>Edit norm versus causal effect size on the {html.escape(method.upper())} sample</figcaption><img class="paper-figure" src="assets/edit_norm_vs_ablation_{html.escape(method)}.png" alt="edit norm versus causal effect size on the {html.escape(method)} sample"></figure>')
    if correlation_figure is not None:
        figure_cards.append('<figure><figcaption>Attribution and edit-norm correlation with exact ablation</figcaption><img class="paper-figure" src="assets/attribution_method_correlation.png" alt="attribution and edit-norm correlation with exact ablation by method"></figure>')
    for method in visible_attribution_methods:
        if method in correlation_distribution_figures:
            figure_cards.append(f'<figure><figcaption>{html.escape(method.upper())} per-sequence attribution–ablation correlation</figcaption><img class="paper-figure" src="assets/attribution_correlation_distribution_{html.escape(method)}.png" alt="{html.escape(method)} per-sequence attribution ablation Spearman distribution"></figure>')
        if method in causal_effect_comparison_figures:
            method_label = "IG" if method == "ig" else "RelP"
            figure_cards.append(f'<figure><figcaption>{method_label} Top Attribution vs. Random</figcaption><img class="paper-figure" src="assets/causal_effect_size_comparison_{html.escape(method)}.png" alt="{method_label} top attribution versus random causal effect size"></figure>')
    if answer_figure is not None:
        figure_cards.append('<figure><figcaption>Final-answer accuracy with and without the suggestion</figcaption><img class="paper-figure" src="assets/answer_accuracy.png" alt="answer accuracy before and after intervention"></figure>')
    if objective_behavior_figure is not None:
        figure_cards.append('<figure><figcaption>Training objective: behavioral recovery</figcaption><img class="paper-figure" src="assets/objective_behavior_recovery.png" alt="behavioral recovery by training objective"></figure>')
    if objective_activation_figure is not None:
        activation_caption = (
            "Activation Recovery" if objective_comparison.get("objectives") and all(
                math.isfinite(_activation_recovery_value(row)) for row in objective_comparison["objectives"]
            ) else "Training objective: activation reconstruction"
        )
        figure_cards.append(f'<figure><figcaption>{activation_caption}</figcaption><img class="paper-figure" src="assets/objective_activation_nmse.png" alt="{activation_caption}"></figure>')
    if support_coverage_figure is not None:
        support_caption = "Unedited support coverage pooled across source arms"
        support_p10 = support_stats.get("p10")
        support_minimum = support_stats.get("minimum")
        figure_cards.append(
            f'<figure><figcaption>{html.escape(support_caption)}</figcaption>'
            f'<p class="muted">Unedited source support mass pooled across response positions and edit directions. p10 {_number(support_p10)} · minimum {_number(support_minimum)}</p>'
            f'<img class="paper-figure" src="assets/conditional_support_coverage.png" alt="distribution of conditional support mass"></figure>'
        )
    else:
        figure_cards.append('<figure><figcaption>Unedited support coverage unavailable</figcaption><p>Saved source-mass histograms are missing. Rebuild stage 1 and then stage 7; minimum-across-arms histograms cannot substitute for pooled masses.</p></figure>')
    if edited_support_figure is not None:
        figure_cards.append(
            '<figure><figcaption>Post-training edited-model support coverage</figcaption>'
            '<p class="muted">Selected editor on held-out response positions, pooled across both directions. '
            'Full-vocabulary probability mass retained by the shared source/target support; '
            f'p10 {_number(edited_support["p10"])} · minimum {_number(edited_support["minimum"])}.</p>'
            '<img class="paper-figure" src="assets/edited_support_coverage.png" '
            'alt="Post-training edited-model support-mass distribution"></figure>'
        )
    elif not edited_support:
        figure_cards.append(
            '<figure><figcaption>Post-training edited-model support coverage</figcaption>'
            '<p class="muted">Unavailable: no saved support histogram from final evaluation '
            'matching the analysis checkpoint. Run the unified evaluator and rebuild the report.</p></figure>'
        )
    if "native_patching" in native_site_figures:
        figure_cards.append('<figure><figcaption>Native activation patching: KL recovery</figcaption><img class="paper-figure" src="assets/native_activation_patching.png" alt="native activation patching and equal-size random controls"></figure>')
    figures_html = '<h2>Quantitative results</h2><div class="paper-figures">' + ''.join(figure_cards) + '</div>' if figure_cards else ''
    model_name = str(config.get("model", {}).get("name", "")).lower()
    model_label = "DeepSeek-R1-0528-Qwen3-8B" if "deepseek" in model_name else "Gemma-4-E4B-it" if "gemma" in model_name else ""
    patch_sites = native_summary.get("native_patching", {}).get("sites", {})
    patch_controls = native_summary.get("native_patching", {}).get("random_controls", {})
    site_label = lambda site: PATCH_PLOT_LABELS.get(site, site.replace("_", " ").title())
    native_rows=[]
    for site in PATCH_SITE_ORDER:
        patch_metric=patch_sites.get(site,{})
        control_metric=patch_controls.get(site,{})
        patch_value=_metric_value(patch_metric, "patch_recovery", "value")
        control_value=_metric_value(control_metric, "patch_recovery", "value")
        if patch_value is None and control_value is None:
            continue
        patch_ci=patch_metric.get("bootstrap_95_ci") or [None,None]
        control_ci=control_metric.get("bootstrap_95_ci") or [None,None]
        native_rows.append(
            f'<tr><td>{html.escape(site_label(site))}</td>'
            f'<td>{_number(patch_value)}</td><td>[{_number(patch_ci[0])}, {_number(patch_ci[1])}]</td>'
            f'<td>{_number(control_value)}</td><td>[{_number(control_ci[0])}, {_number(control_ci[1])}]</td></tr>'
        )
    native_results_parts=[]
    if native_rows:
        native_results_parts.append(
            '<h2>Native activation patching</h2>'
            f'<p class="muted">Target-condition {layer_label} activations are transplanted into the source run at each semantic site. Equal-size random aligned-token controls patch the same number of positions outside all semantic sites.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Site</th><th>Native patch recovery</th><th>95% CI</th><th>Random patch recovery</th><th>95% CI</th></tr></thead><tbody>'
            + ''.join(native_rows) + '</tbody></table></div>'
        )
    native_results_html=''.join(native_results_parts)

    native_link = (
        '<p><a class="pill" href="../native_analysis/index.html">Open stage-4 native activation-patching examples</a></p>'
        if (run_dir / "native_analysis" / "index.html").is_file() else ''
    )
    focused_links = []
    focused_outputs = (
        (
            run_dir / "exhaustive_validation" / "index.html",
            report / "exhaustive_validation.html",
            "Open exhaustive token validation",
            run_dir / "exhaustive_validation" / "exhaustive_token_validation.json",
            data / "exhaustive_token_validation.json",
        ),
        (
            run_dir / "sentence_browser" / "index.html",
            report / "sentence_browser.html",
            "Open keyword sentence browser",
            run_dir / "sentence_browser" / "keyword_sentences.json",
            data / "keyword_sentences.json",
        ),
    )
    for page_source, page_target, label, data_source, data_target in focused_outputs:
        if page_source.is_file() or data_source.is_file():
            if data_source.name == "exhaustive_token_validation.json" and data_source.is_file():
                from .counterfactual_focused_analysis import render_exhaustive_validation_html
                render_exhaustive_validation_html(read_json(data_source), page_target, model_label=model_label)
            elif data_source.is_file():
                from .counterfactual_focused_analysis import render_sentence_browser_html
                render_sentence_browser_html(read_json(data_source), page_target)
            else:
                shutil.copyfile(page_source, page_target)
            focused_links.append(
                f'<a class="pill" href="{html.escape(page_target.name)}">{html.escape(label)}</a>'
            )
            if data_source.is_file():
                shutil.copyfile(data_source, data_target)
        else:
            _write(
                page_target,
                _page(label, '<div class="note">This analysis artifact is unavailable.</div>'),
            )
    focused_link = f'<p>{"".join(focused_links)}</p>' if focused_links else ''
    model_heading = f'<p class="muted">{html.escape(model_label)}</p>' if model_label else ''
    index_body=(model_heading+f'<div class="cards">{cards}</div>'+figures_html+native_results_html+native_link+architecture+objective_table+token_rankings_html+
        focused_link+'<h2>Qualitative examples</h2>'+case_table)
    _write(report/"index.html",_page("Summary",index_body))

    causal_rows=ablation.get("records",[])
    method_sections=[]
    for method in visible_attribution_methods:
        method_validation=validation_by_method.get(method,{})
        aggregate=method_validation.get("aggregate",{})
        validation=method_validation.get("source_clustered_validation",{})
        norm_validation=method_validation.get("edit_norm_source_clustered_validation",{})
        per_response=method_validation.get("per_response_validation",{})
        norm_per_response=method_validation.get("edit_norm_per_response_validation",{})
        agg=aggregate.get("attribution_vs_ablation",{})
        normagg=aggregate.get("edit_norm_vs_ablation",{})
        ci=validation.get("median_spearman_bootstrap_95_ci") or [None,None]
        norm_ci=norm_validation.get("median_spearman_bootstrap_95_ci") or [None,None]
        validation_cards=[
            ("Tested source IDs",validation.get("rank_source_ids")),
            (f"{method.upper()} median per-response Spearman",per_response.get("median_spearman")),
            ("Median source Spearman",validation.get("median_source_level_within_record_spearman")),
            ("Edit-norm median per-response Spearman",norm_per_response.get("median_spearman")),
            ("Edit-norm median source Spearman",norm_validation.get("median_source_level_within_record_spearman")),
        ]
        vcards=''.join(f'<div class="card"><span class="muted">{html.escape(n)}</span><strong>{_percent(v) if pct else _number(v)}</strong></div>' for n,v,*rest in validation_cards for pct in [rest[0] if rest else False])
        metrics_table=f'''<div class="table-wrap"><table><thead><tr><th>Evaluation set</th><th>N</th><th>Pearson</th><th>Spearman</th><th>Sign accuracy</th></tr></thead><tbody>
<tr><td>{html.escape(method.upper())} attribution ({html.escape(selection_description)})</td><td>{int(agg.get("count",0))}</td><td>{_number(agg.get("pearson"))}</td><td>{_number(agg.get("spearman"))}</td><td>{_percent(agg.get("sign_accuracy"))}</td></tr>
<tr><td>Edit norm versus |mean KL increase| on the same events</td><td>{int(normagg.get("count",0))}</td><td>{_number(normagg.get("pearson"))}</td><td>{_number(normagg.get("spearman"))}</td><td>not applicable</td></tr>
</tbody></table></div>'''
        source_table=f'''<div class="table-wrap"><table><tbody>
<tr><th>Attribution: median source-level mean within-record Spearman</th><td>{_number(validation.get("median_source_level_within_record_spearman"))}</td><td>95% clustered bootstrap [{_number(ci[0])}, {_number(ci[1])}]</td></tr>
<tr><th>Attribution: mean source-level Spearman</th><td>{_number(validation.get("mean_source_level_within_record_spearman"))}</td><td>IQR [{_number(validation.get("source_level_spearman_p25"))}, {_number(validation.get("source_level_spearman_p75"))}]</td></tr>
<tr><th>Edit norm: median source-level mean within-record Spearman</th><td>{_number(norm_validation.get("median_source_level_within_record_spearman"))}</td><td>95% clustered bootstrap [{_number(norm_ci[0])}, {_number(norm_ci[1])}]</td></tr>
<tr><th>Edit norm: mean source-level Spearman</th><td>{_number(norm_validation.get("mean_source_level_within_record_spearman"))}</td><td>IQR [{_number(norm_validation.get("source_level_spearman_p25"))}, {_number(norm_validation.get("source_level_spearman_p75"))}]</td></tr>
</tbody></table></div>'''
        method_causal_rows=[
            row for row in causal_rows
            if _is_evaluation_event(row, method)
        ]
        method_sections.append(
            f'<h2>{html.escape(method.upper())}</h2>'
            f'<div class="cards">{vcards}</div>'
            f'<p class="muted">{html.escape(method.upper())} is evaluated independently using {html.escape(selection_description)}. Edit norm is evaluated on exactly those events.</p>'
            '<h3>Top-magnitude plus random correlations</h3>'+metrics_table+
            f'<p><strong>Median per-response Spearman:</strong> {_number(per_response.get("median_spearman"))} across {int(per_response.get("responses_with_spearman",0))} responses.</p>'+
            _scatter(method_causal_rows,x_key=f"{method}_total_attribution_credit",y_key="mean_kl_increase",title=f"{method.upper()} edit-event attribution versus causal effect",x_label=f"{method.upper()} attribution credit",selection_prefix=f"{method}_")+
            _scatter(method_causal_rows,x_key="direct_edit_norm",y_key="causal_effect_size",title=f"Edit norm versus causal effect size on the {method.upper()} sample",x_label="Applied edit norm",selection_prefix=f"{method}_")+
            '<h3>Source-clustered validation</h3>'+source_table+
            _histogram(validation.get("source_level_spearman_values",[]),f"{method.upper()} source-level rank agreement")+
            _histogram(norm_validation.get("source_level_spearman_values",[]),f"Edit-norm source-level rank agreement on {method.upper()} samples")
        )
    validation_intro=(
        f'<div class="note"><strong>Attribution method: {html.escape(primary_attribution_method.upper())}.</strong> Each observed value comes from setting one applied edit event to zero and rerunning the downstream Transformer.</div>'
    )
    validation_body=validation_intro+''.join(method_sections)
    _write(report/"causal_validation.html",_page("Attribution validation",validation_body))

    causal_by=defaultdict(list)
    for row in ablation.get("records",[]): causal_by[str(row["record_id"])].append(row)

    total_by_method={}
    for method in attribution_methods:
        total_by_method[method]={str(row["record_id"]):row for row in attribution_by_method[method].get("records",[]) if str(row.get("mode"))=="teacher_forced"}

    for example_index, record in enumerate(cases):
        rid=str(record["record_id"]); method_blocks=[]
        # Per-example interpretation stays deliberately focused on Integrated
        # Gradients. Display IG's independently selected signed-top events.
        method = "ig" if "ig" in attribution_methods else primary_attribution_method
        total=total_by_method.get(method,{}).get(rid,{})
        ig_causal_rows = causal_by.get(rid,[])
        method_blocks.append(
            f'<h2>Integrated Gradients attribution</h2>'
            +_token_viewer(record,total.get("edit_event_credit",[]),label="total-response IG edit-event credit",legend_positive="supports recovery",legend_negative="opposes recovery")
            +'<h3>Highest-attribution tested edit events</h3>'
            +_causal_event_profile_view(record,ig_causal_rows,method=method,limit=5)
        )
        native_payload = read_json(run_dir / "native_analysis" / "records" / f"{rid}.json", {})
        native_patch_block = ""
        if isinstance(native_payload, dict):
            fixed = native_payload.get("native_fixed_prefix", {})
            patches = fixed.get("patches", {}) if isinstance(fixed, dict) else {}
            controls = fixed.get("random_controls", {}) if isinstance(fixed, dict) else {}
            source_native = fixed.get("source", {}) if isinstance(fixed, dict) else {}
            target_native = fixed.get("target", {}) if isinstance(fixed, dict) else {}
            sections = []
            for site in PATCH_SITE_ORDER:
                patch = patches.get(site, {}) if isinstance(patches, dict) else {}
                control = controls.get(site, {}) if isinstance(controls, dict) else {}
                if not patch and not control:
                    continue
                label = site.replace("_", " ").title().replace("Question And Filler", "Question + filler")
                parts = [f'<h3>{html.escape(label)}</h3>']
                if patch:
                    parts.append(
                        f'<div class="metric-row"><span>Native patch recovery</span><strong>{_number(patch.get("recovery"))}</strong>'
                        f'<span>Patched KL</span><strong>{_number(patch.get("condition_kl_mean"))}</strong>'
                        '</div>'
                    )
                    prompt_len = int(record.get("prompt_tokens", 0))
                    patch_effect = [0.0] * prompt_len + [float(value) for value in patch.get("per_response_kl_recovered", [])]
                    parts.append(_token_viewer(
                        record, patch_effect,
                        label=f"{label}: source KL − native-patch KL", response_only=False,
                        selected_indices=patch.get("source_indices", []),
                        legend_positive="patch improves target match", legend_negative="patch worsens target match",
                        hover_details=_native_patch_hover_details(source_native, target_native, patch),
                        hover_callouts=_native_patch_hover_callouts(patch),
                        include_default_support_changes=False,
                    ))
                if control:
                    parts.append(
                        f'<h4>Equal-size random patch control</h4><div class="metric-row"><span>Control recovery</span><strong>{_number(control.get("recovery"))}</strong>'
                        f'<span>Control KL</span><strong>{_number(control.get("condition_kl_mean"))}</strong></div>'
                    )
                    prompt_len = int(record.get("prompt_tokens", 0))
                    control_effect = [0.0] * prompt_len + [float(value) for value in control.get("per_response_kl_recovered", [])]
                    parts.append(_token_viewer(
                        record, control_effect,
                        label=f"{label} random control: source KL − native-patch KL", response_only=False,
                        selected_indices=control.get("source_indices", []),
                        legend_positive="control patch improves target match", legend_negative="control patch worsens target match",
                        hover_details=_native_patch_hover_details(source_native, target_native, control),
                        hover_callouts=_native_patch_hover_callouts(control),
                        include_default_support_changes=False,
                    ))
                sections.append("".join(parts))
            if sections:
                native_patch_block = (
                    '<h2>Native activation patching: per-token KL recovery</h2>'
                    f'<p class="muted">Each heatmap shows source conditional KL minus patched conditional KL at each fixed-response position. Positive values mean the {layer_label} patch moves the native source computation toward the target condition. Hover a response token to compare the three highest-probability stored alternatives under Source, Target, and Native patch. Each semantic patch is paired with a deterministic equal-size random aligned-token control.</p>'
                    + "".join(sections)
                )
        body=(
            f'<p><span class="pill">Example {example_index}</span>{direction_badge(str(record["direction"]))}<span class="pill">analysis replay recovery {_number(record.get("recovery"))}</span></p>'
            + (
                '<p class="muted">Per-example attribution uses Integrated Gradients only; the displayed edit events are the five largest signed IG events and therefore all have exact teacher-forced causal profiles. This qualitative set is separate from the top-magnitude-plus-random validation sample.</p>'
            )
            + ''.join(method_blocks)
            + native_patch_block
        )
        direction_title = "Control to Intervention" if record["direction"] == "c_to_i" else "Intervention to Control"
        _write(examples/f'{rid}.html',_page(f'Example {example_index} · {direction_title}',body,relative_root='../'))

    manifest={
        "interpretation_object": "counterfactual_applied_edit_events",
        "attribution_methods": attribution_methods,
        "visible_attribution_methods": visible_attribution_methods,
        "primary_attribution_method": primary_attribution_method,
        "attribution_view": attribution_view,
        "examples": [str(r["record_id"]) for r in cases],
        "pages": [
            "index.html",
            "causal_validation.html",
            "exhaustive_validation.html",
            "sentence_browser.html",
        ],
        "recovery_audit": "data/recovery_audit.json",
        "top_edit_attribution_tokens": "data/top_edit_attribution_tokens.json",
        "objective_comparison": ("data/objective_comparison.json" if objective_comparison else None),
    }
    write_json(report/"report_manifest.json",manifest)
    return manifest

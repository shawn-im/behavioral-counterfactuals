from __future__ import annotations

"""Native-model hypothesis-analysis facade for counterfactual interface experiments.

This module provides a shared import entry point for patch-site selection,
native activation patching, aggregation, and HTML reporting.
"""

from .native_hypothesis_sites import _QUESTION_START, _QUESTION_END, load_native_runtime, canonical_direction_sign, _display_token, _is_newline_token, _question_text, _token_span_for_text, _terminal_newline_index, detect_patch_sites, select_native_record_summaries, partition_summaries_by_source, _site_metric_aggregate
from .native_hypothesis_patching import (
    _ActivationReplaceHook,
    _HiddenCapture,
    _response_slice,
    _summarize_response_logits,
    _native_response_head_context,
    _forward_source_or_target,
    _forward_with_patch,
    _conditional_kl_metrics,
    _context_for_position,
    _shift_details,
    runtime_decode_token,
    analyze_native_record,
    _native_patch_summary,
    compute_native_hypothesis_analysis,
    merge_native_hypothesis_shards,
    compute_native_hypothesis_analysis_shard,
)
from .native_hypothesis_report import (
    _fmt,
    _site_label,
    _response_effect_heatmap,
    write_native_quantitative_figures,
    _metric_value,
    _example_rank,
    build_native_hypothesis_report,
    _select_native_report_payloads,
)

from __future__ import annotations

"""Supported numerical API for the staged counterfactual edit analysis harness.

The implementation is split into focused internal modules; this facade exposes
the operations used by production scripts and their contract tests.
"""

from .counterfactual_runtime_core import (
    json_default,
    write_json,
    read_json,
    _signed_phrase_spans,
    is_static_mlp,
    _direction_names_from_signs,
    decode_applied_edit,
    load_edit_checkpoint,
    Runtime,
    load_runtime,
    _CaptureHook,
    _FixedEditHook,
    _InlineDirectionalEditHook,
    capture_hidden,
    trace_from_hidden,
    select_diverse_rows,
    truncate_cache_row,
    direction_sign,
    target_prompt_ids,
    _model_forward_parameters,
    _last_token_logits_kwarg,
)
from .counterfactual_runtime_tracing import (
    _TOKEN_TEXT_CACHE,
    token_texts,
    _SupportOnlyOutputHead,
    _BatchedResponseOnlyOutputHead,
    _BatchedSupportOnlyOutputHead,
    _response_support_logits,
    _response_logits,
    _conditional_probabilities,
    _top_support_probability_changes,
    _record_id,
    validate_support_row,
    trace_record,
    _summarize_records,
    save_records,
    trace_teacher_forced,
    merge_teacher_forced_shards,
    _right_pad_sequences,
    trace_records_batch,
    trace_records_adaptive,
    _record_paths,
    iter_records,
    load_records,
    load_record_summaries,
)
from .counterfactual_runtime_attribution import (
    _gauss_legendre_nodes,
    _scalar_integrated_gradient,
    _make_response_kl_utility,
    _cuda_oom,
    _adaptive_cuda_call,
    _attribution_filename,
    _write_attribution_artifacts,
    merge_attribution_shards,
    prepare_attribution_record_shards,
    compute_scalar_kl_attribution,
    qualitative_record_key,
    select_qualitative_record_keys,
)
from .counterfactual_runtime_ablation import (
    score_edits,
    _record_with_matched_baseline,
    _baseline_parity,
    _load_total_attribution,
    _correlation_summary,
    _source_clustered_causal_summary,
    _build_edit_event_ablation_artifact,
    _multi_method_ablation_candidates,
    prepare_ablation_record_shards,
    compute_edit_event_ablations,
    merge_edit_event_ablation_shards,
)

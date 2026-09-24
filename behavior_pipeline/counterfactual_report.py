from __future__ import annotations

"""Human-facing report facade for counterfactual applied-edit-event analysis.

This module provides a shared import entry point for report construction and
formatting helpers implemented in the focused modules below.
"""

from .counterfactual_report_html import _write, _number, _percent, _method_label, _page, _ZWSP_ENTITY, _display_token_text, _heat_color, _token_viewer, _effect_span_html, _case_records, _report_primary_method, _report_visible_methods, _causal_event_profile_view, _scatter, _histogram
from .counterfactual_report_figures import (
    _paper_matplotlib,
    _write_recovery_figure,
    _write_objective_comparison_figures,
    _write_attribution_ablation_figure,
    _raw_scruples_label_map,
    _answer_accuracy_summary,
    _write_answer_accuracy_figure,
    _write_support_coverage_figure,
    _top_edit_attribution_tokens,
    _recovery_audit,
)
from .counterfactual_report_build import (
    build_report,
)

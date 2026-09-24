"""Dependency-free presentation helpers for generated and pulled reports."""
import html
import math


def edited_support_distribution(method: dict | None) -> dict:
    """Validate saved edited-model support statistics and prepare report data.

    Return an empty dictionary when no populated histogram is available.
    Reject invalid populated histogram counts, bin edges, or summary statistics.
    Preserve the saved values and attach plot labels and checkpoint identity.
    """
    if not method or not method.get("student_support_mass_histogram"):
        return {}
    histogram = method["student_support_mass_histogram"]
    counts = list(histogram.get("counts", []))
    edges = list(histogram.get("edges", []))
    if not counts or not sum(counts):
        return {}
    if (len(edges) != len(counts) + 1
            or any(not math.isfinite(x) or x < 0 or int(x) != x for x in counts)
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))
            or histogram.get("positions", sum(counts)) != sum(counts)):
        raise ValueError("Invalid edited-model support-mass histogram")
    p10 = method.get("student_support_mass_p10")
    minimum = method.get("student_support_mass_min")
    if any(x is None or not math.isfinite(x) or not 0 <= x <= 1 for x in (p10, minimum)):
        raise ValueError("Invalid edited-model support-mass statistics")
    return {
        "population": "held_out_response_positions_both_edit_directions",
        "support_mass_definition": "edited_model_probability_mass_on_shared_source_target_support",
        "title": "Support Mass after Training",
        "x_label": "Edited-model Support Mass",
        "method": method.get("method"),
        "checkpoint_identity": method.get("checkpoint_identity"),
        "positions": sum(counts), "counts": counts, "edges": edges,
        "p10": p10, "minimum": minimum,
    }


def pooled_unedited_support_distribution(distribution: dict) -> dict:
    """Prepare report data from the saved pooled source-mass distribution.

    With both directions, source is control for c_to_i and intervention for
    i_to_c, so this already pools both arms. Adding target masses would duplicate
    that population. Reuse histogram validation from edited_support_distribution,
    then attach the unedited-model labels. Preserve the saved pooled quantile
    and return an empty dictionary if a populated source histogram is absent.
    """
    source = distribution.get("series", {}).get("source", {})
    result = edited_support_distribution({
        "student_support_mass_histogram": source.get("histogram"),
        "student_support_mass_p10": source.get("p10"),
        "student_support_mass_min": source.get("minimum"),
    })
    if not result:
        return {}
    result.pop("method")
    result.pop("checkpoint_identity")
    result.update(
        population="teacher_forced_response_positions_pooled_source_arms",
        support_mass_definition="unedited_source_probability_mass_on_shared_source_target_support",
        title="Support Mass before Training",
        x_label="Unedited-model Support Mass",
        series=distribution.get("series", {}),
    )
    return result


def direction_badge(direction: str) -> str:
    label, color, background = {
        "c_to_i": ("Control to Intervention", "#a63838", "#fbeeee"),
        "i_to_c": ("Intervention to Control", "#315b89", "#eef2f8"),
    }[direction]
    return (f'<span class="direction-badge" style="display:inline-block;white-space:nowrap;'
            f'border-radius:999px;padding:4px 12px;font-weight:600;color:{color};'
            f'background:{background};border:1px solid {color}">{label}</span>')


def example_table(records) -> str:
    rows = ''.join(
        f'<tr><td><a href="examples/{html.escape(record_id, quote=True)}.html">Example {number}</a></td>'
        f'<td>{direction_badge(direction)}</td></tr>'
        for number, (record_id, direction) in enumerate(records)
    )
    return ('<div class="table-wrap"><table class="example-table">'
            '<thead><tr><th>Example</th><th>Direction</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')

from __future__ import annotations

"""Shared publication-style defaults for all counterfactual report plots."""

PLOT_COLORS = {
    "primary": "#0072b2",
    "secondary": "#e69f00",
    "control": "#2ca02c",
    "ig": "#0072b2",
    "intgrad": "#0072b2",
    "relp": "#e69f00",
    "source": "#0072b2",
    "target": "#e69f00",
    "editor": "#2ca02c",
    "native_patch": "#0072b2",
    "behavior": "#0072b2",
    "activation": "#e69f00",
}

SCATTER_COLORS = {
    # Okabe-Ito colors remain legible on gray and accumulate cleanly when dense.
    "attribution": "#009e73",
    "edit_norm": "#cc79a7",
}

OKABE_ITO_COLORS = {
    "blue": "#0072b2",
    "orange": "#e69f00",
    "green": "#009e73",
}

ATTRIBUTION_METHOD_COLORS = {
    "ig": OKABE_ITO_COLORS["blue"],
    "relp": OKABE_ITO_COLORS["orange"],
}

PLOT_STYLE = {
    "figsize": (6.8, 4.4),
    "split_scatter_figsize": (10.6, 4.8),
    "dpi": 220,
    "font_size": 13.0,
    "title_size": 18.0,
    "label_size": 14.0,
    "tick_size": 12.0,
    "legend_size": 12.0,
    "bar_width": 0.66,
    "scatter_size": 13,
    "scatter_alpha": 0.30,
    "control_alpha": 0.30,
    "hist_alpha": 0.58,
    "line_width": 2.1,
}


def apply_report_plot_style(matplotlib, plt) -> None:
    """Apply consistent styling to static HTML-report figures."""
    palette = [
        PLOT_COLORS["primary"],
        PLOT_COLORS["secondary"],
        PLOT_COLORS["control"],
    ]
    plt.rcParams.update(
        {
            "font.family": matplotlib.rcParamsDefault["font.family"],
            "font.sans-serif": matplotlib.rcParamsDefault["font.sans-serif"],
            "font.size": PLOT_STYLE["font_size"],
            "axes.titlesize": PLOT_STYLE["title_size"],
            "axes.titleweight": "semibold",
            "axes.labelsize": PLOT_STYLE["label_size"],
            "axes.edgecolor": "#AEB7C2",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#D7DDE4",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.42,
            "xtick.labelsize": PLOT_STYLE["tick_size"],
            "ytick.labelsize": PLOT_STYLE["tick_size"],
            "xtick.color": "black",
            "ytick.color": "#4B5563",
            "legend.fontsize": PLOT_STYLE["legend_size"],
            "legend.frameon": False,
            "figure.facecolor": "#f4f4f4",
            "axes.facecolor": "#f4f4f4",
            "savefig.facecolor": "#f4f4f4",
            "savefig.dpi": PLOT_STYLE["dpi"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": matplotlib.cycler(color=palette),
        }
    )

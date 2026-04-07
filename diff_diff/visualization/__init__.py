"""
Visualization functions for difference-in-differences analysis.

Provides event study plots, diagnostic visualizations, and other plotting
utilities with support for matplotlib (default) and plotly backends.
"""

# Event study plots
# Continuous DiD plots
from diff_diff.visualization._continuous import (
    plot_dose_response,
)

# Diagnostic plots
from diff_diff.visualization._diagnostic import (
    plot_bacon,
    plot_sensitivity,
)
from diff_diff.visualization._event_study import (
    PlottableResults,
    _extract_plot_data,
    plot_event_study,
    plot_honest_event_study,
)

# Power analysis plots
from diff_diff.visualization._power import (
    plot_power_curve,
    plot_pretrends_power,
)

# Staggered DiD plots
from diff_diff.visualization._staggered import (
    plot_group_effects,
    plot_group_time_heatmap,
    plot_staircase,
)

# Synthetic control plots
from diff_diff.visualization._synthetic import (
    plot_synth_weights,
)

__all__ = [
    # Existing public functions
    "plot_event_study",
    "plot_honest_event_study",
    "plot_group_effects",
    "plot_sensitivity",
    "plot_bacon",
    "plot_power_curve",
    "plot_pretrends_power",
    # New public functions
    "plot_synth_weights",
    "plot_staircase",
    "plot_dose_response",
    "plot_group_time_heatmap",
    # Re-exported for backward compatibility (used in tests)
    "_extract_plot_data",
    "PlottableResults",
]

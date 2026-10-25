"""Shared lane definitions for the TWFE event-study decoupling oracle (4.0 PR 4c-i/4c-ii).

Single owner of the oracle PANELS and LANES used by BOTH
``tests/_capture_v4_merge_mpd_oracles.py`` (which wrote
``tests/fixtures/v4_twfe_event_study_oracles.json`` on the pre-refactor tree)
and ``tests/test_twfe_event_study_decoupling.py`` (which refits the lanes on
the current tree and compares). Keeping panel construction and lane kwargs
here means the tests rebuild exactly what the capture fitted; the fixture
carries each lane's ``echo()`` so an edited lane fails as "lane drift", not as
a numeric regression.

Why these lanes: each is a ``TwoWayFixedEffects.fit(..., event_study=True)``
design, pooled or within, covering every variance lane
``_fit_event_study_core`` offers (HC1 default, explicit cluster, hc2
carve-out, hc2_bm per-period BM df, cluster + hc2_bm CR2, Conley, survey TSL,
survey replicate, string labels, a non-suffix partition and a rank-deficient
NaN post period). The committed records were captured while the core still
had an independent twin (``MultiPeriodDiD``, removed in PR 4c-ii), so they
pin the TWFE surface against the pre-refactor numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from diff_diff import SurveyDesign, generate_did_data

PANEL_SEED = 20261024
NAN_PANEL_SEED = 42
JK_REPLICATES = 15
JK_COLUMNS: List[str] = [f"rep_{r}" for r in range(JK_REPLICATES)]


def build_panel() -> pd.DataFrame:
    """The main oracle panel: ``generate_did_data`` plus the derived columns every
    lane may need (coordinates for Conley, survey weights/strata/PSU, JK1
    replicate weights, a string copy of the period label)."""
    df = generate_did_data(
        n_units=60,
        n_periods=8,
        treatment_period=4,
        treatment_effect=2.0,
        seed=PANEL_SEED,
    ).copy()
    df["lat"] = 40.0 + 0.01 * df["unit"]
    df["lon"] = -100.0 + 0.01 * (df["unit"] % 7)
    rng = np.random.default_rng(1)
    df["w"] = rng.uniform(0.5, 2.0, size=len(df))  # per-row sampling weight
    df["stratum"] = df["unit"] % 4
    df["psu"] = df["unit"]
    # JK1 delete-a-group-of-PSUs replicate weights: 60 PSUs in 15 groups of 4
    # (the tests/test_replicate_weight_expansion.py recipe); retained rows are
    # rescaled by n_rep / (n_rep - 1).
    units = sorted(df["unit"].unique())
    group = len(units) // JK_REPLICATES
    for r, col in enumerate(JK_COLUMNS):
        deleted = list(units[r * group : (r + 1) * group])
        w_r = df["w"].to_numpy(copy=True)
        mask = df["unit"].isin(deleted).to_numpy()
        w_r[mask] = 0.0
        w_r[~mask] *= JK_REPLICATES / (JK_REPLICATES - 1)
        df[col] = w_r
    df["period_s"] = "p" + df["period"].astype(str)
    return df


def build_nan_post_panel() -> pd.DataFrame:
    """The rank-deficient NaN post-period design: 100 units x 6
    periods (treated = first half, effect 3.0 from period 3), then EVERY treated
    row in post period 3 dropped so the ``treated:period_3`` interaction is a
    zero column and that period's effect is NaN."""
    rng = np.random.RandomState(NAN_PANEL_SEED)
    n_units = 100
    rows = []
    for unit in range(n_units):
        is_treated = unit < n_units // 2
        unit_effect = rng.normal(0, 1)
        for period in range(6):
            y = 10.0 + unit_effect + period * 0.5
            if is_treated and period >= 3:
                y += 3.0
            y += rng.normal(0, 0.5)
            rows.append({"unit": unit, "period": period, "treated": int(is_treated), "outcome": y})
    df = pd.DataFrame(rows)
    keep = ~((df["treated"] == 1) & (df["period"] == 3))
    return pd.DataFrame(df.loc[keep]).reset_index(drop=True)


PANELS: Dict[str, Callable[[], pd.DataFrame]] = {
    "main": build_panel,
    "nan_post": build_nan_post_panel,
}


def build_survey(label: Optional[str]) -> Optional[SurveyDesign]:
    """Materialize a lane's survey design from its label (None -> no design)."""
    if label is None:
        return None
    if label == "tsl:w/stratum/psu":
        return SurveyDesign(weights="w", strata="stratum", psu="psu")
    if label == "jk1:w":
        return SurveyDesign(weights="w", replicate_weights=JK_COLUMNS, replicate_method="JK1")
    raise ValueError(f"unknown survey label {label!r}")


@dataclass(frozen=True)
class Lane:
    """One oracle lane: a TWFE event-study fit (ctor kwargs + fit kwargs).

    Fit kwargs never contain objects; the survey design is built from
    ``survey_label`` at fit time. ``panel`` names a ``PANELS`` entry.
    """

    panel: str
    twfe_ctor: Dict[str, Any]
    twfe_fit: Dict[str, Any]
    survey_label: Optional[str] = None
    notes: str = field(default="", compare=False)

    def echo(self) -> Dict[str, Any]:
        """JSON-safe record of everything that defines the lane (drift pin)."""
        return {
            "panel": self.panel,
            "twfe_ctor": dict(self.twfe_ctor),
            "twfe_fit": dict(self.twfe_fit),
            "survey_label": self.survey_label,
        }

    def survey(self) -> Optional[SurveyDesign]:
        return build_survey(self.survey_label)


_COMMON = {"outcome": "outcome", "treatment": "treated", "time": "period"}
_PART = {"post_periods": [4, 5, 6, 7], "reference_period": 3}
_CONLEY_CTOR = {
    "vcov_type": "conley",
    "conley_coords": ["lat", "lon"],
    "conley_cutoff_km": 50.0,
    "conley_lag_cutoff": 2,
}


def _pooled(ctor: Dict[str, Any], **fit_extra: Any) -> Lane:
    return Lane(
        panel="main",
        twfe_ctor=dict(ctor),
        twfe_fit={**_COMMON, **_PART, "spec": "pooled", **fit_extra},
    )


def _within(ctor: Dict[str, Any], survey_label: Optional[str] = None) -> Lane:
    return Lane(
        panel="main",
        twfe_ctor=dict(ctor),
        twfe_fit={**_COMMON, **_PART, "spec": "within", "unit": "unit"},
        survey_label=survey_label,
    )


LANES: Dict[str, Lane] = {
    "pooled_default": _pooled({}),
    "pooled_cluster_unit": _pooled({"cluster": "unit"}),
    "pooled_hc2_bm": _pooled({"vcov_type": "hc2_bm"}),
    "within_cluster_unit": _within({"cluster": "unit"}),
    "within_hc2": _within({"vcov_type": "hc2"}),
    # Cluster + hc2_bm: the CR2 Bell-McCaffrey bypass that no other lane
    # reaches; captured at d45d5316 (PR 4c-ii, before the core's post-average
    # block was removed) - see provenance.lane_additions.
    "pooled_cluster_hc2_bm": _pooled({"vcov_type": "hc2_bm", "cluster": "unit"}),
    "within_cluster_hc2_bm": _within({"vcov_type": "hc2_bm", "cluster": "unit"}),
    "within_conley": _within(_CONLEY_CTOR),
    "within_survey_tsl": _within({}, survey_label="tsl:w/stratum/psu"),
    "within_replicate_jk1": _within({}, survey_label="jk1:w"),
    "pooled_string_labels": Lane(
        panel="main",
        twfe_ctor={},
        twfe_fit={
            "outcome": "outcome",
            "treatment": "treated",
            "time": "period_s",
            "spec": "pooled",
            "post_periods": ["p4", "p5", "p6", "p7"],
            "reference_period": "p3",
        },
    ),
    "pooled_non_suffix": Lane(
        panel="main",
        twfe_ctor={},
        twfe_fit={**_COMMON, "spec": "pooled", "post_periods": [2, 5], "reference_period": 4},
    ),
    "pooled_nan_post_period": Lane(
        panel="nan_post",
        twfe_ctor={},
        twfe_fit={**_COMMON, "spec": "pooled", "post_periods": [3, 4, 5], "reference_period": 2},
    ),
}

#: Pooled lanes whose post-period average is finite (the NaN lane is excluded:
#: ``nan == nan`` is False; its contract is pinned NaN-aware separately).
FINITE_POOLED_LANES = [
    name
    for name, lane in LANES.items()
    if lane.twfe_fit.get("spec") == "pooled" and name != "pooled_nan_post_period"
]
WITHIN_LANES = [name for name, lane in LANES.items() if lane.twfe_fit.get("spec") == "within"]

"""Capture the pre-removal aggregation oracles for tests/test_aggregate_contract.py.

HISTORICAL since the 4.0 cut (PR 3b, completed by PR 3c): this script drives the
REMOVED fit-time ``aggregate=`` / ``balance_e=`` route on every adopter (and imports
the contract module's retired ``_boot_fit_time`` / ``_efficient_boot_fit_time``
helpers), so it is runnable only at the recorded
``provenance.captured_at_head`` of ``tests/fixtures/v4_aggregate_oracles.json``
(``8724fecd``, whose ``diff_diff/`` is byte-identical to the capture tree) - the
same convention as ``tests/_capture_v4_merge_ddd_oracles.py``. It is kept as the
fixture's provenance record; re-capturing means checking out that commit, ONCE
PER BACKEND::

    DIFF_DIFF_BACKEND=python python tests/_capture_v4_aggregate_oracles.py > numpy.json
    DIFF_DIFF_BACKEND=rust   python tests/_capture_v4_aggregate_oracles.py > rust.json

and merging the two payloads into ``tests/fixtures/v4_aggregate_oracles.json``
(``{"analytic": ..., "boot": {"numpy": ..., "rust": ...}, "provenance": ...}``).

Why. The 4.0 cut (PR 3b; Imputation/TwoStage/ContinuousDiD in PR 3c) deletes the
deprecated fit-time aggregation route that
``test_aggregate_contract.py`` used as its NUMERICAL REFERENCE (the ``*_fit_time``
fixtures and ``_boot_fit_time`` helpers): post-fit ``aggregate()`` was asserted
equal to ``fit(aggregate=...)``. Once that route is removed there is no independent
second implementation left to disagree with, so the reference becomes these
committed numbers, captured while the route still existed:

* ANALYTIC oracles - the ``fit(aggregate="all")`` (ContinuousDiD:
  ``"eventstudy"``) and ``balance_e=1`` surfaces on every design arm the contract
  file exercises. After the removal the kit recompute must reproduce them.
* BOOTSTRAPPED oracles, per weight backend - ``generate_bootstrap_weights_batch``
  draws a DIFFERENT seeded stream under the Rust backend than under NumPy, so
  the percentile surfaces are backend-specific. After the removal the CS/EDiD
  replays and the Imputation/TwoStage/ContinuousDiD relays must reproduce them,
  and a bootstrapped PLAIN fit's overall quintet must be unchanged (the
  always-on surface materialization draws no extra weights).

The panel builders are imported from the contract test module itself so the
oracle panels are, by construction, the test panels.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from typing import Any, Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "helpers"))

import test_aggregate_contract as T  # noqa: E402  (the contract module's panel builders)

from diff_diff import CallawaySantAnna, SurveyDesign  # noqa: E402
from diff_diff.bootstrap_chunking import effective_weight_backend  # noqa: E402


# --------------------------------------------------------------------------- #
# serialization helpers
# --------------------------------------------------------------------------- #
def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def _key(k: Any) -> Any:
    # JSON object keys must be strings; keep a lossless repr of the label.
    if isinstance(k, (np.integer, int)):
        return str(int(k))
    if isinstance(k, (np.floating, float)):
        return repr(float(k))
    return str(k)


def _row(d: Dict[str, Any]) -> Dict[str, Any]:
    ci = d.get("conf_int", (np.nan, np.nan))
    out = {
        "effect": _f(d.get("effect")),
        "se": _f(d.get("se")),
        "t_stat": _f(d.get("t_stat")),
        "p_value": _f(d.get("p_value")),
        "ci_lo": _f(ci[0]),
        "ci_hi": _f(ci[1]),
    }
    for k in ("n_obs", "n_groups", "n_units", "df_used"):
        if k in d:
            out[k] = _f(d[k])
    cb = d.get("cband_conf_int")
    if cb is not None:
        out["cband_lo"], out["cband_hi"] = _f(cb[0]), _f(cb[1])
    return out


def _dict_surface(surface: Optional[Dict[Any, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if surface is None:
        return None
    return {_key(k): _row(v) for k, v in sorted(surface.items(), key=lambda kv: float(kv[0]))}


def _overall(r: Any) -> Dict[str, Optional[float]]:
    ci = getattr(r, "overall_conf_int", None) or (np.nan, np.nan)
    return {
        "att": _f(r.att),
        "se": _f(r.se),
        "t_stat": _f(getattr(r, "overall_t_stat", np.nan)),
        "p_value": _f(r.p_value),
        "ci_lo": _f(ci[0]),
        "ci_hi": _f(ci[1]),
    }


def _matrix(m: Any) -> Optional[list]:
    if m is None:
        return None
    arr = np.asarray(m, dtype=float)
    return [[_f(x) for x in row] for row in arr]


def _ref_labels(r: Any) -> Optional[list]:
    """The event-time labels the post-fit container marks ``is_reference``.

    Stored so the oracle check can assert reference MEMBERSHIP independently of
    the container under test (a surface that wrongly marks an estimated horizon
    as reference must fail, not skip). Where the post-fit route is closed on
    the capture tree (bootstrapped Imputation/TwoStage/ContinuousDiD) the
    ``is_reference``-keyed rows (the 4.0 marking, row M-093) are the
    reference set; a surface with neither yields ``None`` (no membership pin).
    """
    try:
        es = _quiet(r.aggregate, "event_study")
    except NotImplementedError:
        legacy = r.event_study_effects or {}
        refs = [k_ for k_, v in ((_key(k), v) for k, v in legacy.items()) if v.get("is_reference")]
        return sorted(refs, key=float)
    labels = list(es.event_time)
    flags = np.asarray(es.is_reference, dtype=bool)
    return sorted((_key(lab) for lab, f in zip(labels, flags) if f), key=float)


def _es_bundle(r: Any) -> Dict[str, Any]:
    """Event-study surface + its provenance side fields where the class has them."""
    out: Dict[str, Any] = {"es": _dict_surface(r.event_study_effects)}
    out["ref_labels"] = None if r.event_study_effects is None else _ref_labels(r)
    for attr in ("event_study_vcov_index", "reference_event_times"):
        v = getattr(r, attr, None)
        out[attr] = None if v is None else [_f(x) for x in v]
    out["event_study_vcov"] = _matrix(getattr(r, "event_study_vcov", None))
    out["event_study_df"] = _f(getattr(r, "event_study_df", None))
    out["cband_crit_value"] = _f(getattr(r, "cband_crit_value", None))
    return out


def _all_bundle(r: Any) -> Dict[str, Any]:
    out = _es_bundle(r)
    out["group"] = _dict_surface(getattr(r, "group_effects", None))
    out["overall"] = _overall(r)
    return out


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# --------------------------------------------------------------------------- #
# analytic oracles (backend-independent in principle; the two runs are diffed)
# --------------------------------------------------------------------------- #
def capture_analytic() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    panel = T._panel()

    # ---- CallawaySantAnna --------------------------------------------------
    cs: Dict[str, Any] = {}
    cs["default_all"] = _all_bundle(
        _quiet(CallawaySantAnna().fit, panel, aggregate="all", **T.FIT_KW)
    )
    cs["default_balance_e1"] = _es_bundle(
        _quiet(CallawaySantAnna().fit, panel, aggregate="event_study", balance_e=1, **T.FIT_KW)
    )
    cs["rcs_all"] = _all_bundle(
        _quiet(CallawaySantAnna(panel=False).fit, T._rcs(), aggregate="all", **T.FIT_KW)
    )
    cs["anticipation1_all"] = _all_bundle(
        _quiet(CallawaySantAnna(anticipation=1).fit, panel, aggregate="all", **T.FIT_KW)
    )
    cs["universal_all"] = _all_bundle(
        _quiet(CallawaySantAnna(base_period="universal").fit, panel, aggregate="all", **T.FIT_KW)
    )
    cs["nyt_anticipation1_all"] = _all_bundle(
        _quiet(
            CallawaySantAnna(control_group="not_yet_treated", anticipation=1).fit,
            panel,
            aggregate="all",
            **T.FIT_KW,
        )
    )
    thinned = panel.drop(panel.index[::13]).reset_index(drop=True)
    cs["unbalanced_all"] = _all_bundle(
        _quiet(
            CallawaySantAnna(allow_unbalanced_panel=True).fit, thinned, aggregate="all", **T.FIT_KW
        )
    )
    cs["survey_all"] = _all_bundle(
        _quiet(
            CallawaySantAnna().fit,
            T._survey_panel(),
            survey_design=T._survey_design(),
            aggregate="all",
            **T.FIT_KW,
        )
    )
    out["CallawaySantAnna"] = cs

    # ---- EfficientDiD ------------------------------------------------------
    ed: Dict[str, Any] = {}
    ep = T._efficient_panel()
    ed["default_all"] = _all_bundle(T._fit_efficient(ep, aggregate="all"))
    ed["default_balance_e1"] = _es_bundle(
        T._fit_efficient(ep, aggregate="event_study", balance_e=1)
    )
    ed["fractional_balance_e1"] = _es_bundle(
        T._fit_efficient(T._efficient_fractional_panel(), aggregate="event_study", balance_e=1)
    )
    ed["clustered_all"] = _all_bundle(
        T._fit_efficient(T._efficient_clustered_panel(), est_kw={"cluster": "cl"}, aggregate="all")
    )
    d_sw, _ = T._efficient_survey_panel()
    ed["survey_weights_all"] = _all_bundle(
        T._fit_efficient(d_sw, survey_design=T._efficient_survey_design(), aggregate="all")
    )
    for tag, degenerate in (("healthy", None), ("dropped", "dropped"), ("undefined", "undefined")):
        d_rep, rep_cols = T._efficient_survey_panel(replicate=True, degenerate=degenerate)
        ed[f"replicate_{tag}_all"] = _all_bundle(
            T._fit_efficient(
                d_rep, survey_design=T._efficient_survey_design(rep_cols), aggregate="all"
            )
        )
    out["EfficientDiD"] = ed

    # ---- ImputationDiD -----------------------------------------------------
    im: Dict[str, Any] = {}
    ip = T._imputation_panel()
    im["default_all"] = _all_bundle(T._fit_imputation(ip, aggregate="all"))
    im["default_balance_e1"] = _es_bundle(
        T._fit_imputation(ip, aggregate="event_study", balance_e=1)
    )
    im["pretrends_all"] = _all_bundle(
        T._fit_imputation(ip, est_kw={"pretrends": True}, aggregate="all")
    )
    im["clustered_all"] = _all_bundle(
        T._fit_imputation(
            T._imputation_clustered_panel(), est_kw={"cluster": "cl"}, aggregate="all"
        )
    )
    d_sw, _ = T._imputation_survey_panel()
    im["survey_tsl_all"] = _all_bundle(
        T._fit_imputation(d_sw, survey_design=T._imputation_survey_design(), aggregate="all")
    )
    for degenerate in (None, "dropped", "undefined", "cohort_zero"):
        d_rep, rep_cols = T._imputation_survey_panel(replicate=True, degenerate=degenerate)
        sd = T._imputation_survey_design(rep_cols)
        tag = degenerate or "none"
        # LEVEL-MATCHED references: the joint replicate stack couples rows, so
        # fit(aggregate="event_study") and fit(aggregate="group") are separate.
        im[f"replicate_{tag}_event_study"] = _es_bundle(
            T._fit_imputation(d_rep, survey_design=sd, aggregate="event_study")
        )
        im[f"replicate_{tag}_group"] = _all_bundle(
            T._fit_imputation(d_rep, survey_design=sd, aggregate="group")
        )
    d_cov = T._imputation_panel(seed=23).copy()
    d_cov["x1"] = np.random.default_rng(23).normal(size=len(d_cov)) + 0.1 * d_cov["period"]
    im["covariates_all"] = _all_bundle(T._fit_imputation(d_cov, covariates=["x1"], aggregate="all"))
    out["ImputationDiD"] = im

    # ---- TwoStageDiD -------------------------------------------------------
    ts: Dict[str, Any] = {}
    tp = T._twostage_panel()
    ts["default_all"] = _all_bundle(T._fit_twostage(tp, aggregate="all"))
    ts["default_balance_e1"] = _es_bundle(T._fit_twostage(tp, aggregate="event_study", balance_e=1))
    ts["pretrends_all"] = _all_bundle(
        T._fit_twostage(tp, est_kw={"pretrends": True}, aggregate="all")
    )
    ts["prop5_all"] = _all_bundle(T._fit_twostage(T._twostage_prop5_panel(), aggregate="all"))
    ts["clustered_all"] = _all_bundle(
        T._fit_twostage(T._twostage_clustered_panel(), est_kw={"cluster": "cl"}, aggregate="all")
    )
    d_sw, _ = T._twostage_survey_panel()
    ts["survey_tsl_all"] = _all_bundle(
        T._fit_twostage(d_sw, survey_design=T._twostage_survey_design(), aggregate="all")
    )
    ts["survey_psu_all"] = _all_bundle(
        T._fit_twostage(d_sw, survey_design=T._twostage_survey_design(psu=True), aggregate="all")
    )
    ts["always_treated_survey_all"] = _all_bundle(
        T._fit_twostage(
            T._twostage_always_treated_panel(),
            survey_design=T._twostage_survey_design(),
            aggregate="all",
        )
    )
    for degenerate in (None, "dropped", "undefined", "cohort_zero"):
        d_rep, rep_cols = T._twostage_survey_panel(replicate=True, degenerate=degenerate)
        sd = T._twostage_survey_design(rep_cols)
        tag = degenerate or "none"
        ts[f"replicate_{tag}_event_study"] = _es_bundle(
            T._fit_twostage(d_rep, survey_design=sd, aggregate="event_study")
        )
        ts[f"replicate_{tag}_group"] = _all_bundle(
            T._fit_twostage(d_rep, survey_design=sd, aggregate="group")
        )
    ts["always_treated_survey_psu_all"] = _all_bundle(
        T._fit_twostage(
            T._twostage_always_treated_panel(),
            survey_design=T._twostage_survey_design(psu=True),
            aggregate="all",
        )
    )
    d_cov = T._twostage_panel(seed=23).copy()
    d_cov["x1"] = np.random.default_rng(23).normal(size=len(d_cov)) + 0.1 * d_cov["period"]
    ts["covariates_all"] = _all_bundle(T._fit_twostage(d_cov, covariates=["x1"], aggregate="all"))
    out["TwoStageDiD"] = ts

    # ---- ContinuousDiD (no group level; "eventstudy" is the only ES spelling) -
    co: Dict[str, Any] = {}
    cp = T._cont_panel()
    co["default_eventstudy"] = _all_bundle(T._fit_cont(cp, aggregate="eventstudy"))
    co["covariate_eventstudy"] = _all_bundle(
        T._fit_cont(T._cont_covariate_panel(), covariates=["x1"], aggregate="eventstudy")
    )
    co["anticipation1_eventstudy"] = _all_bundle(
        T._fit_cont(T._cont_panel(seed=47), est_kw={"anticipation": 1}, aggregate="eventstudy")
    )
    d_sw, _ = T._cont_survey_panel()
    co["survey_tsl_eventstudy"] = _all_bundle(
        T._fit_cont(d_sw, survey_design=T._cont_survey_design(), aggregate="eventstudy")
    )
    co["empty_post_eventstudy"] = _all_bundle(
        T._fit_cont(T._cont_empty_post_panel(), aggregate="eventstudy")
    )
    # The remaining TestContinuousAggregate.test_event_study_across_designs
    # arms (previously pinned only structurally); data built exactly as there.
    co["multi_cohort_eventstudy"] = _all_bundle(
        T._fit_cont(T._cont_panel(seed=43, cohort_periods=[3, 4, 5]), aggregate="eventstudy")
    )
    d_zd, _ = T._cont_survey_panel(zero_dose_treated=True)
    co["survey_zero_dose_drop_eventstudy"] = _all_bundle(
        T._fit_cont(d_zd, survey_design=T._cont_survey_design(), aggregate="eventstudy")
    )
    for tag, degenerate in (("healthy", None), ("dropped", "dropped"), ("undefined", "undefined")):
        d_rep, rep_cols = T._cont_survey_panel(replicate=True, degenerate=degenerate)
        co[f"replicate_{tag}_eventstudy"] = _all_bundle(
            T._fit_cont(
                d_rep,
                survey_design=T._cont_survey_design(rep_cols=rep_cols),
                aggregate="eventstudy",
            )
        )
    co["discrete_eventstudy"] = _all_bundle(
        T._fit_cont(
            T._cont_discrete_panel(), est_kw={"treatment_type": "discrete"}, aggregate="eventstudy"
        )
    )
    co["lowest_dose_eventstudy"] = _all_bundle(
        T._fit_cont(
            T._cont_lowest_dose_panel(),
            est_kw={"control_group": "lowest_dose"},
            aggregate="eventstudy",
        )
    )
    co["not_yet_treated_eventstudy"] = _all_bundle(
        T._fit_cont(
            T._cont_panel(seed=53, cohort_periods=[3, 5]),
            est_kw={"control_group": "not_yet_treated"},
            aggregate="eventstudy",
        )
    )
    co["nondefault_config_eventstudy"] = _all_bundle(
        T._fit_cont(
            T._cont_panel(seed=59),
            est_kw={"alpha": 0.10, "base_period": "universal"},
            aggregate="eventstudy",
        )
    )
    out["ContinuousDiD"] = co
    return out


# --------------------------------------------------------------------------- #
# bootstrapped oracles (backend-specific)
# --------------------------------------------------------------------------- #
def capture_boot() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    panel = T._panel()

    # ---- CallawaySantAnna: fit-time reference surfaces the replay must match -
    cs: Dict[str, Any] = {}
    cs["default_all"] = _all_bundle(T._boot_fit_time(panel))
    cs["default_plain_overall"] = _overall(T._boot_fit(panel))
    d = panel.copy()
    d["psu"] = d["unit"] % 7
    cs["psu_cluster_all"] = _all_bundle(T._boot_fit_time(d, cluster="psu"))
    d2 = panel.copy()
    rng = np.random.default_rng(5)
    wmap = {u: rng.uniform(0.5, 2.0) for u in d2["unit"].unique()}
    d2["w"] = d2["unit"].map(wmap)
    d2["psu"] = d2["unit"] % 7
    d2["stratum"] = d2["unit"] % 2
    sd = SurveyDesign(weights="w", strata="stratum", psu="psu", nest=True)
    cs["stratified_survey_all"] = _all_bundle(
        T._boot_fit_time(d2, fit_kwargs={"survey_design": sd})
    )
    cs["cband_all"] = _all_bundle(T._boot_fit_time(panel, cband=True))
    # The three design arms TestBootstrapReplayDesigns previously pinned only
    # structurally: non-census FPC survey, repeated cross-sections, unbalanced.
    d3 = panel.copy()
    rng3 = np.random.default_rng(6)
    wmap3 = {u: rng3.uniform(0.5, 2.0) for u in d3["unit"].unique()}
    d3["w"] = d3["unit"].map(wmap3)
    d3["psu"] = d3["unit"] % 7
    d3["fpc"] = 100
    sd3 = SurveyDesign(weights="w", psu="psu", fpc="fpc")
    cs["fpc_survey_all"] = _all_bundle(T._boot_fit_time(d3, fit_kwargs={"survey_design": sd3}))
    cs["rcs_all"] = _all_bundle(T._boot_fit_time(T._rcs(), panel=False))
    thinned = panel.drop(panel.index[::13]).reset_index(drop=True)
    cs["unbalanced_all"] = _all_bundle(T._boot_fit_time(thinned, allow_unbalanced_panel=True))
    out["CallawaySantAnna"] = cs

    # ---- EfficientDiD ------------------------------------------------------
    ed: Dict[str, Any] = {}
    ep = T._efficient_panel()
    ed["default_all"] = _all_bundle(T._efficient_boot_fit_time(ep))
    ed["default_plain_overall"] = _overall(T._efficient_boot_fit(ep))
    ed["clustered_all"] = _all_bundle(
        T._efficient_boot_fit_time(T._efficient_clustered_panel(), est_kw={"cluster": "cl"})
    )
    d_sw, _ = T._efficient_survey_panel()
    ed["survey_weights_all"] = _all_bundle(
        T._efficient_boot_fit_time(d_sw, survey_design=T._efficient_survey_design())
    )
    d_ps, sd_ps = T.TestEfficientBootstrapReplayDesigns._psu_survey_frame(strata=True)
    ed["stratified_psu_survey_all"] = _all_bundle(
        T._efficient_boot_fit_time(d_ps, survey_design=sd_ps)
    )
    d_fpc, sd_fpc = T.TestEfficientBootstrapReplayDesigns._psu_survey_frame(fpc=200.0)
    ed["fpc_survey_all"] = _all_bundle(T._efficient_boot_fit_time(d_fpc, survey_design=sd_fpc))
    ed["fractional_balance_e1_es"] = _es_bundle(
        T._efficient_boot_fit_time(
            T._efficient_fractional_panel(), aggregate="event_study", balance_e=1
        )
    )
    # Arms TestEfficientBootstrapReplay(Designs) previously pinned only
    # structurally: covariates, anticipation and the alpha/Mammen wiring.
    d_cov = ep.copy()
    rng_cov = np.random.default_rng(3)
    xmap = {u: rng_cov.normal() for u in d_cov["unit"].unique()}
    d_cov["x1"] = d_cov["unit"].map(xmap)
    ed["covariates_all"] = _all_bundle(T._efficient_boot_fit_time(d_cov, covariates=["x1"]))
    ed["anticipation1_all"] = _all_bundle(
        T._efficient_boot_fit_time(ep, est_kw={"anticipation": 1})
    )
    ed["alpha10_mammen_all"] = _all_bundle(
        T._efficient_boot_fit_time(ep, est_kw={"alpha": 0.10, "bootstrap_weights": "mammen"})
    )
    out["EfficientDiD"] = ed

    # ---- ImputationDiD: the fit-time route the relay must reproduce ---------
    im: Dict[str, Any] = {}
    ip = T._imputation_panel()
    boot_kw = {"n_bootstrap": 19, "seed": 1}
    im["default_all"] = _all_bundle(T._fit_imputation(ip, est_kw=boot_kw, aggregate="all"))
    im["default_plain_overall"] = _overall(T._fit_imputation(ip, est_kw=boot_kw))
    im["pretrends_all"] = _all_bundle(
        T._fit_imputation(ip, est_kw={**boot_kw, "pretrends": True}, aggregate="all")
    )
    im["pretrends_plain_overall"] = _overall(
        T._fit_imputation(ip, est_kw={**boot_kw, "pretrends": True})
    )
    d_sw, _ = T._imputation_survey_panel()
    im["survey_tsl_all"] = _all_bundle(
        T._fit_imputation(
            d_sw, est_kw=boot_kw, survey_design=T._imputation_survey_design(), aggregate="all"
        )
    )
    im["survey_tsl_plain_overall"] = _overall(
        T._fit_imputation(d_sw, est_kw=boot_kw, survey_design=T._imputation_survey_design())
    )
    out["ImputationDiD"] = im

    # ---- TwoStageDiD -------------------------------------------------------
    ts: Dict[str, Any] = {}
    tp = T._twostage_panel()
    ts["default_all"] = _all_bundle(T._fit_twostage(tp, est_kw=boot_kw, aggregate="all"))
    ts["default_plain_overall"] = _overall(T._fit_twostage(tp, est_kw=boot_kw))
    ts["pretrends_all"] = _all_bundle(
        T._fit_twostage(tp, est_kw={**boot_kw, "pretrends": True}, aggregate="all")
    )
    ts["prop5_all"] = _all_bundle(
        T._fit_twostage(T._twostage_prop5_panel(), est_kw=boot_kw, aggregate="all")
    )
    ts["prop5_plain_overall"] = _overall(T._fit_twostage(T._twostage_prop5_panel(), est_kw=boot_kw))
    out["TwoStageDiD"] = ts

    # ---- ContinuousDiD -----------------------------------------------------
    co: Dict[str, Any] = {}
    cp = T._cont_panel()
    cont_kw = {"n_bootstrap": 30, "seed": 3}

    def _cont_bundle(r: Any) -> Dict[str, Any]:
        b = _es_bundle(r)
        b["overall"] = _overall(r)
        b["overall_acrt"] = _f(getattr(r, "overall_acrt", np.nan))
        b["overall_acrt_se"] = _f(getattr(r, "overall_acrt_se", np.nan))
        return b

    co["default_eventstudy"] = _cont_bundle(T._fit_cont(cp, est_kw=cont_kw, aggregate="eventstudy"))
    co["default_plain"] = _cont_bundle(T._fit_cont(cp, est_kw=cont_kw))
    out["ContinuousDiD"] = co
    return out


def dump_fixture(obj: Any) -> str:
    """The fixture's canonical text: sorted keys, one indent level per nesting
    level, and every LEAF container (a dict or list of scalars - an oracle
    row, a label list, a vcov matrix row) on a single line. Diffable per row
    without the ~3x size of a fully indented dump."""

    def _fmt(o: Any, ind: int = 0) -> str:
        pad = " " * ind
        if isinstance(o, dict):
            if all(not isinstance(v, (dict, list)) for v in o.values()):
                return json.dumps(o, sort_keys=True)
            items = [f"{pad} {json.dumps(k)}: {_fmt(v, ind + 1)}" for k, v in sorted(o.items())]
            return "{\n" + ",\n".join(items) + f"\n{pad}}}"
        if isinstance(o, list):
            if all(not isinstance(v, (dict, list)) for v in o):
                return json.dumps(o)
            items = [f"{pad} {_fmt(v, ind + 1)}" for v in o]
            return "[\n" + ",\n".join(items) + f"\n{pad}]"
        return json.dumps(o)

    return _fmt(obj) + "\n"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    import diff_diff

    payload = {
        "provenance": {
            "captured_at_head": _git_head(),
            "diff_diff_version": diff_diff.__version__,
            "weight_backend": effective_weight_backend(),
            "numpy": np.__version__,
        },
        "analytic": capture_analytic(),
        "boot": capture_boot(),
    }
    sys.stdout.write(dump_fixture(payload))

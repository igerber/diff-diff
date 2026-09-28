"""Step 0: capture the pre-merge oracles for tests/test_v4_merge_ddd.py.

MUST run on the UNMODIFIED tree, under DIFF_DIFF_BACKEND=python.

Two oracles:
  A) 2x2x2 engine (TripleDifference.fit today) - guards the rewritten fit
     prologue/signature/dispatch.
  B) staggered engine (StaggeredTripleDifference.fit today) - guards the
     ~1700-line relocation into the private engine mixin. Nothing else in CI
     pins this in absolute terms (the SDDD suites carry no committed numeric
     pins; the R-golden lane skips when the gitignored CSVs are absent).

The +-cluster axis of oracle A uses generate_ddd_panel_data, NOT
generate_ddd_data: the cross-sectional generator emits `unit_id` incremented
once per ROW, so every cluster would be a singleton (and there is no `unit`
column at all). The panel generator has real repeated units - the same shape
tests/test_prep.py:1438 already fits with cluster="unit".

Oracle B covers every branch the relocation MOVED, not just the happy path:
the DR base config and its bootstrap, the three nuisance models under
covariates (_compute_pscore / _compute_or), the never-treated comparison fork
(_is_never_treated) and the survey-pweight path. Re-running this file on a
tree where the engine has already moved reproduces the same literals - that is
the point of the gate, but it is NOT a substitute for capturing pre-move: once
relocated there is no independent path left to disagree with.
"""

import json

import numpy as np

from diff_diff import StaggeredTripleDifference, TripleDifference
from diff_diff.prep_dgp import (
    generate_ddd_data,
    generate_ddd_panel_data,
    generate_staggered_ddd_data,
)
from diff_diff.survey import SurveyDesign

CROSS_KW = dict(
    n_per_cell=200,
    treatment_effect=2.0,
    group_effect=1.0,
    partition_effect=0.5,
    time_effect=0.7,
    noise_sd=1.0,
    add_covariates=True,
    seed=42,
)
PANEL_KW = dict(n_units=80, n_periods=4, treatment_period=2, noise_sd=1.0, seed=42)
STAG_KW = dict(n_units=96, n_periods=6, cohort_periods=[3, 4], seed=42)
# add_covariates=True feeds x1/x2 into the OUTCOME, so the covariate lanes are a
# different DGP draw than STAG_KW - they get their own fixture and their own
# oracle keys rather than being compared against the plain-data literals.
STAG_COV_KW = dict(STAG_KW, add_covariates=True)

COVS = ["age", "education"]
STAG_COVS = ["x1", "x2"]


def _f(x):
    """JSON-safe float (NaN -> None sentinel string handled by the caller)."""
    if x is None:
        return None
    x = float(x)
    return "NaN" if np.isnan(x) else x


def quintet(r):
    ci = getattr(r, "conf_int", None)
    return {
        "att": _f(r.att),
        "se": _f(r.se),
        "t_stat": _f(r.t_stat),
        "p_value": _f(r.p_value),
        "conf_int_lower": _f(ci[0]) if ci is not None else None,
        "conf_int_upper": _f(ci[1]) if ci is not None else None,
    }


def capture_2x2x2():
    out = {}
    df = generate_ddd_data(**CROSS_KW)
    for method in ("dr", "reg", "ipw"):
        r = TripleDifference(estimation_method=method).fit(
            df, outcome="outcome", group="group", partition="partition", post="time"
        )
        out[f"cross_{method}"] = dict(
            quintet(r),
            n_obs=int(r.n_obs),
            n_treated_eligible=int(r.n_treated_eligible),
            n_treated_ineligible=int(r.n_treated_ineligible),
            n_control_eligible=int(r.n_control_eligible),
            n_control_ineligible=int(r.n_control_ineligible),
            vcov_type=r.vcov_type,
            cluster_name=r.cluster_name,
            n_clusters=r.n_clusters,
        )
    r = TripleDifference(estimation_method="dr").fit(
        df,
        outcome="outcome",
        group="group",
        partition="partition",
        post="time",
        covariates=COVS,
    )
    out["cross_dr_cov"] = dict(quintet(r), n_obs=int(r.n_obs))

    # survey pweight lane (cross-sectional)
    dfw = df.copy()
    rng = np.random.default_rng(7)
    dfw["w"] = rng.uniform(0.5, 2.0, size=len(dfw))
    r = TripleDifference(estimation_method="reg").fit(
        dfw,
        outcome="outcome",
        group="group",
        partition="partition",
        post="time",
        survey_design=SurveyDesign(weights="w"),
    )
    out["cross_survey_pweight_reg"] = dict(quintet(r), n_obs=int(r.n_obs))

    # cluster lane MUST use the panel generator (see module docstring)
    pdf = generate_ddd_panel_data(**PANEL_KW)
    r = TripleDifference(estimation_method="dr", cluster="unit").fit(
        pdf, outcome="outcome", group="group", partition="partition", post="post"
    )
    out["panel_dr_cluster_unit"] = dict(
        quintet(r),
        n_obs=int(r.n_obs),
        cluster_name=r.cluster_name,
        n_clusters=int(r.n_clusters) if r.n_clusters is not None else None,
    )
    return out


def _gt_table(r):
    """Sorted (g,t) -> effect/se, key-stringified for JSON."""
    return {
        f"{g}|{t}": {"effect": _f(v["effect"]), "se": _f(v["se"])}
        for (g, t), v in sorted(r.group_time_effects.items())
    }


def capture_staggered():
    out = {}
    df = generate_staggered_ddd_data(**STAG_KW)
    fit_cols = dict(
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        eligibility="eligibility",
    )
    base = StaggeredTripleDifference(estimation_method="dr").fit(df, aggregate="all", **fit_cols)
    out["stag_dr_all"] = {
        "overall": quintet(base),
        "overall_att_es": _f(base.overall_att_es),
        "overall_se_es": _f(base.overall_se_es),
        "n_obs": int(base.n_obs),
        "n_treated_units": int(base.n_treated_units),
        "n_never_enabled": int(base.n_never_enabled),
        "group_time": _gt_table(base),
    }
    boot = StaggeredTripleDifference(
        estimation_method="dr", n_bootstrap=49, seed=7, cband=True
    ).fit(df, aggregate="all", **fit_cols)
    out["stag_dr_all_boot49"] = {
        "overall": quintet(boot),
        "cband_crit_value": _f(boot.cband_crit_value),
        "group_time": _gt_table(boot),
    }

    # --- Branch coverage for the RELOCATED nuisance/comparison/survey code ----
    # The two lanes above ride DR-without-covariates only. The mixin also moved
    # _compute_pscore, _compute_or and the never-treated comparison branch, and
    # those are otherwise pinned only by comparing two callers of the SAME moved
    # code - a shared transcription slip would keep that parity green. Each lane
    # below is an ABSOLUTE pin on one of the moved branches.

    # NOTE: dr/ipw/reg are numerically IDENTICAL on covariate-free data (the
    # propensity score is constant and the outcome regression is a bare mean, so
    # all three collapse to the same simple DiD - verified at capture time).
    # Committing three identical literal blocks would look like nuisance-model
    # coverage while providing none, so the no-covariate lanes are NOT captured
    # per method; the test asserts that convergence live instead, and the
    # discriminating pins are the *_cov lanes below.

    # never-treated comparison branch (the _is_never_treated fork). The compact
    # spelling is the DYING class's vocabulary; the merged surface passes
    # "never_treated" and must land on the same numbers.
    r = StaggeredTripleDifference(estimation_method="dr", control_group="nevertreated").fit(
        df, aggregate="all", **fit_cols
    )
    out["stag_dr_nevertreated"] = {
        "overall": quintet(r),
        "overall_att_es": _f(r.overall_att_es),
        "overall_se_es": _f(r.overall_se_es),
        "group_time": _gt_table(r),
    }

    # covariate lanes - the DGP differs (x1/x2 enter the outcome), so these pin
    # against their own fixture, not against stag_dr_all.
    dfc = generate_staggered_ddd_data(**STAG_COV_KW)
    for method in ("dr", "ipw", "reg"):
        r = StaggeredTripleDifference(estimation_method=method).fit(
            dfc, aggregate="all", covariates=STAG_COVS, **fit_cols
        )
        out[f"stag_{method}_cov"] = {
            "overall": quintet(r),
            "overall_att_es": _f(r.overall_att_es),
            "overall_se_es": _f(r.overall_se_es),
            "group_time": _gt_table(r),
        }

    # survey pweight lane through the staggered engine.
    dfw = df.copy()
    rng = np.random.default_rng(11)
    per_unit = {
        u: w
        for u, w in zip(
            sorted(dfw["unit"].unique()), rng.uniform(0.5, 2.0, size=dfw["unit"].nunique())
        )
    }
    dfw["w"] = dfw["unit"].map(per_unit)
    r = StaggeredTripleDifference(estimation_method="dr").fit(
        dfw, aggregate="all", survey_design=SurveyDesign(weights="w"), **fit_cols
    )
    out["stag_dr_survey_pweight"] = {
        "overall": quintet(r),
        "overall_att_es": _f(r.overall_att_es),
        "overall_se_es": _f(r.overall_se_es),
        "group_time": _gt_table(r),
    }
    return out


if __name__ == "__main__":
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = {"2x2x2": capture_2x2x2(), "staggered": capture_staggered()}
    print(json.dumps(payload, indent=1, sort_keys=True))

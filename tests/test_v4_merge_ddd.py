"""Phase 3(b) merge gates: TripleDifference absorbs StaggeredTripleDifference.

Ledger rows under test: M-013 (the class merge), M-064 (the SDDD alias warning
riding the class), M-140/M-141 (fit-time aggregate=/balance_e= carried on the
surviving class), M-142 (the pscore_trim validation tightening).

TOLERANCE DOCTRINE
------------------
Every merge-PARITY gate is BIT-EXACT (``assert_array_equal``): both sides run
the same engine in the same process, so a needed tolerance IS the finding.

Two deliberate exceptions, both comparing against literals captured in a
DIFFERENT process:

* the committed ORACLES below, and
* the R-parity lane (that suite's own tolerances).

The oracles are NOT portable across backends - ``TripleDifference`` routes OLS
through ``solve_ols`` (``diff_diff/triple_diff.py``), which dispatches to Rust
or NumPy, and the repo's own equivalence claim is only ``decimal=8``
(``tests/test_rust_backend.py``), while CI runs the suite under BOTH
``DIFF_DIFF_BACKEND=rust`` and ``=python`` (``.github/workflows/rust-test.yml``).
So oracle assertions use ``assert_allclose(rtol=1e-9, atol=1e-12)`` AND are
guarded to the python backend.

ORACLE PROVENANCE
-----------------
Captured on the UNMODIFIED pre-merge tree, before any source edit::

    DIFF_DIFF_BACKEND=python python3 capture_oracles.py

commit be694f493ac4e1640cdcccb5bdf7997a5b1f24dd.

Why two oracles, not one. ORACLE_2X2X2 guards the rewritten fit prologue,
signature and dispatch. ORACLE_STAGGERED guards the ~1700-line relocation of
the staggered engine into ``diff_diff/_staggered_triple_diff_engine.py``:
nothing else in CI pins that engine in ABSOLUTE terms, because the parity gates
compare two callers of the same relocated core, the four SDDD suites carry no
committed numeric pins, and the R-golden lane skips whenever the gitignored
CSVs are absent (``.gitignore``). Without it a transcription slip in the move
would be invisible.

Two capture-config notes worth keeping:

* The +-cluster oracle lane uses ``generate_ddd_panel_data``, NOT
  ``generate_ddd_data``: the cross-sectional generator emits ``unit_id``
  incremented once per ROW (and no ``unit`` column at all), so every cluster
  would be a singleton. The panel generator has real repeated units - the shape
  ``tests/test_prep.py`` already fits with ``cluster="unit"``.
* ``cross_dr``/``cross_reg``/``cross_ipw`` agree to ~1e-13: on a saturated DDD
  with no covariates all three estimators collapse to the same cell-mean
  contrast. Those three lanes pin that the prologue still ROUTES correctly;
  they are not independent evidence about method-specific math (the
  ``_cov``/survey/cluster lanes carry that).
* The same collapse applies to the STAGGERED engine, which is why there is no
  per-method covariate-free staggered oracle: three identical literal blocks
  would look like nuisance-model coverage while providing none. The convergence
  is asserted live (``test_methods_converge_without_covariates``) and the
  discriminating pins are the ``stag_{dr,ipw,reg}_cov`` lanes, whose mutual
  distinctness is itself asserted so the parametrization cannot quietly stop
  discriminating.

ORACLE_STAGGERED covers, in absolute terms, every branch the relocation moved:
the DR base config and its bootstrap, the three nuisance models under
covariates (``_compute_pscore`` / ``_compute_or``), the never-treated
comparison fork (``_is_never_treated``), and the survey-pweight path. The
covariate lanes use ``STAG_COV_KW`` (``add_covariates=True``), a DIFFERENT draw
than ``STAG_KW`` because x1/x2 enter the outcome - they pin against their own
fixture, never against ``stag_dr_all``.
"""

import os
import re
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from diff_diff import SDDD, StaggeredTripleDifference, TripleDifference
from diff_diff.prep_dgp import (
    generate_ddd_data,
    generate_ddd_panel_data,
    generate_staggered_ddd_data,
)
from diff_diff.staggered_triple_diff import _SDDD_DEPRECATION_MSG
from diff_diff.staggered_triple_diff_results import StaggeredTripleDiffResults
from diff_diff.survey import SurveyDesign
from diff_diff.triple_diff import TripleDifferenceResults

# Oracle assertions compare literals captured in a DIFFERENT process, so they
# are tolerance-based AND python-backend-guarded (see the tolerance doctrine).
ORACLE_RTOL = 1e-9
ORACLE_ATOL = 1e-12
_BACKEND_IS_PYTHON = os.environ.get("DIFF_DIFF_BACKEND", "").lower() == "python"
requires_python_backend = pytest.mark.skipif(
    not _BACKEND_IS_PYTHON,
    reason="committed oracle literals were captured under DIFF_DIFF_BACKEND=python; "
    "solve_ols dispatches to Rust or NumPy and the repo's cross-backend claim is "
    "only decimal=8, looser than this gate's tolerance",
)

# captured at be694f493ac4e1640cdcccb5bdf7997a5b1f24dd
ORACLE_2X2X2 = {
    "cross_dr": {
        "att": 2.0192439829287876,
        "cluster_name": None,
        "conf_int_lower": 1.3227242700327153,
        "conf_int_upper": 2.71576369582486,
        "n_clusters": None,
        "n_control_eligible": 400,
        "n_control_ineligible": 400,
        "n_obs": 1600,
        "n_treated_eligible": 400,
        "n_treated_ineligible": 400,
        "p_value": 1.5412648419886862e-08,
        "se": 0.35510355075673344,
        "t_stat": 5.686352554418942,
        "vcov_type": "hc1",
    },
    "cross_dr_cov": {
        "att": 2.0051590625940046,
        "conf_int_lower": 1.6213078544981028,
        "conf_int_upper": 2.3890102706899063,
        "n_obs": 1600,
        "p_value": 6.681912366997255e-24,
        "se": 0.19569696877741513,
        "t_stat": 10.246244870939538,
    },
    "cross_ipw": {
        "att": 2.0192439829287636,
        "cluster_name": None,
        "conf_int_lower": 1.3227242700326913,
        "conf_int_upper": 2.715763695824836,
        "n_clusters": None,
        "n_control_eligible": 400,
        "n_control_ineligible": 400,
        "n_obs": 1600,
        "n_treated_eligible": 400,
        "n_treated_ineligible": 400,
        "p_value": 1.5412648419892834e-08,
        "se": 0.35510355075673344,
        "t_stat": 5.686352554418875,
        "vcov_type": "hc1",
    },
    "cross_reg": {
        "att": 2.0192439829288134,
        "cluster_name": None,
        "conf_int_lower": 1.322724270032741,
        "conf_int_upper": 2.7157636958248856,
        "n_clusters": None,
        "n_control_eligible": 400,
        "n_control_ineligible": 400,
        "n_obs": 1600,
        "n_treated_eligible": 400,
        "n_treated_ineligible": 400,
        "p_value": 1.5412648419880546e-08,
        "se": 0.35510355075673344,
        "t_stat": 5.686352554419014,
        "vcov_type": "hc1",
    },
    "cross_survey_pweight_reg": {
        "att": 2.1916660209559,
        "conf_int_lower": 1.324875700947568,
        "conf_int_upper": 3.0584563409642316,
        "n_obs": 1600,
        "p_value": 7.820014665174102e-07,
        "se": 0.4419133298063041,
        "t_stat": 4.9594928985657285,
    },
    "panel_dr_cluster_unit": {
        "att": 1.9176442049480977,
        "cluster_name": "unit",
        "conf_int_lower": 1.0160086827192953,
        "conf_int_upper": 2.8192797271769,
        "n_clusters": 80,
        "n_obs": 320,
        "p_value": 3.7155532583036706e-05,
        "se": 0.4582420854203778,
        "t_stat": 4.184784126034402,
    },
}

ORACLE_STAGGERED = {
    "stag_dr_all": {
        "group_time": {
            "3.0|2": {"effect": -0.027719840746928036, "se": 0.2811010312694456},
            "3.0|3": {"effect": 2.762999666832223, "se": 0.2838203199927689},
            "3.0|4": {"effect": 2.946059062614397, "se": 0.395161527831989},
            "3.0|5": {"effect": 3.0227256449202673, "se": 0.3155992489776067},
            "3.0|6": {"effect": 3.426518594109598, "se": 0.3635995672681256},
            "4.0|2": {"effect": -0.0027171020705352477, "se": 0.32263986406339495},
            "4.0|3": {"effect": -0.544147723760115, "se": 0.3356048332466622},
            "4.0|4": {"effect": 3.5587114889457214, "se": 0.3634683141995581},
            "4.0|5": {"effect": 3.597702813259514, "se": 0.34627545455921666},
            "4.0|6": {"effect": 3.7459834980563302, "se": 0.39030264674197895},
        },
        "n_never_enabled": 24,
        "n_obs": 576,
        "n_treated_units": 36,
        "overall": {
            "att": 3.2943858241054356,
            "conf_int_lower": 2.850906376698114,
            "conf_int_upper": 3.737865271512757,
            "p_value": 5.076360843907034e-48,
            "se": 0.22626918193672466,
            "t_stat": 14.559586930520208,
        },
        "overall_att_es": 3.3109024203559563,
        "overall_se_es": 0.23391014483341283,
    },
    "stag_dr_all_boot49": {
        "cband_crit_value": 2.7121511569356116,
        "group_time": {
            "3.0|2": {"effect": -0.027719840746928036, "se": 0.2722181669173392},
            "3.0|3": {"effect": 2.762999666832223, "se": 0.28424292019694336},
            "3.0|4": {"effect": 2.946059062614397, "se": 0.3731732550535591},
            "3.0|5": {"effect": 3.0227256449202673, "se": 0.3110908492941959},
            "3.0|6": {"effect": 3.426518594109598, "se": 0.33836191268742344},
            "4.0|2": {"effect": -0.0027171020705352477, "se": 0.31692583310553385},
            "4.0|3": {"effect": -0.544147723760115, "se": 0.3762241748319726},
            "4.0|4": {"effect": 3.5587114889457214, "se": 0.3607135604501477},
            "4.0|5": {"effect": 3.597702813259514, "se": 0.4049317945954622},
            "4.0|6": {"effect": 3.7459834980563302, "se": 0.39946787358947394},
        },
        "overall": {
            "att": 3.2943858241054356,
            "conf_int_lower": 2.973695470025734,
            "conf_int_upper": 3.82236144438262,
            "p_value": 0.02,
            "se": 0.23513162889644051,
            "t_stat": 14.010815301910695,
        },
    },
    "stag_dr_nevertreated": {
        "group_time": {
            "3.0|2": {"effect": -0.036712929377447884, "se": 0.3156231893405813},
            "3.0|3": {"effect": 2.4505616566073596, "se": 0.3418289853541229},
            "3.0|4": {"effect": 2.946059062614397, "se": 0.395161527831989},
            "3.0|5": {"effect": 3.0227256449202673, "se": 0.3155992489776067},
            "3.0|6": {"effect": 3.426518594109598, "se": 0.3635995672681256},
            "4.0|2": {"effect": -0.02225820203848114, "se": 0.36226342694022373},
            "4.0|3": {"effect": -0.544147723760115, "se": 0.3356048332466622},
            "4.0|4": {"effect": 3.5587114889457214, "se": 0.3634683141995581},
            "4.0|5": {"effect": 3.597702813259514, "se": 0.34627545455921666},
            "4.0|6": {"effect": 3.7459834980563302, "se": 0.39030264674197895},
        },
        "overall": {
            "att": 3.249751822644741,
            "conf_int_lower": 2.8069280007214155,
            "conf_int_upper": 3.6925756445680666,
            "p_value": 6.560096242110517e-47,
            "se": 0.22593467299208733,
            "t_stat": 14.383590529101983,
        },
        "overall_att_es": 3.2718476690778484,
        "overall_se_es": 0.23479955028064609,
    },
    "stag_dr_cov": {
        "group_time": {
            "3.0|2": {"effect": 0.30104528952593945, "se": 0.3288203170665867},
            "3.0|3": {"effect": 3.0785075003936813, "se": 0.2880034245519164},
            "3.0|4": {"effect": 2.7415492276631688, "se": 0.3403067398792659},
            "3.0|5": {"effect": 2.701920247082487, "se": 0.32408769717572766},
            "3.0|6": {"effect": 3.1354001973198953, "se": 0.3767773891775244},
            "4.0|2": {"effect": 0.09285109959308976, "se": 0.3024820435924664},
            "4.0|3": {"effect": 0.2869519248812908, "se": 0.323592049203466},
            "4.0|4": {"effect": 2.7543305469828474, "se": 0.3709804006699708},
            "4.0|5": {"effect": 2.7349820440287345, "se": 0.2930394677448108},
            "4.0|6": {"effect": 2.6886356893894936, "se": 0.3633732146038028},
        },
        "overall": {
            "att": 2.8336179218371864,
            "conf_int_lower": 2.4550853638028727,
            "conf_int_upper": 3.2121504798715,
            "p_value": 9.758456975554044e-49,
            "se": 0.19313240499321932,
            "t_stat": 14.671892694220174,
        },
        "overall_att_es": 2.871340706272525,
        "overall_se_es": 0.20149333462640176,
    },
    "stag_ipw_cov": {
        "group_time": {
            "3.0|2": {"effect": 0.30697667162647824, "se": 0.3247400010524773},
            "3.0|3": {"effect": 3.1252746382126366, "se": 0.29781018754636845},
            "3.0|4": {"effect": 2.730696345860042, "se": 0.3223912049564754},
            "3.0|5": {"effect": 2.668735095865907, "se": 0.31800505390579015},
            "3.0|6": {"effect": 3.0764584177360605, "se": 0.3616484472028155},
            "4.0|2": {"effect": 0.10289546605785345, "se": 0.30357100920990315},
            "4.0|3": {"effect": 0.29457053449196857, "se": 0.3264538981435691},
            "4.0|4": {"effect": 2.725589827482862, "se": 0.4021132659327127},
            "4.0|5": {"effect": 2.70592308497532, "se": 0.3077614474407017},
            "4.0|6": {"effect": 2.6604495789311056, "se": 0.3767586528071246},
        },
        "overall": {
            "att": 2.8133038555805623,
            "conf_int_lower": 2.4218667691120874,
            "conf_int_upper": 3.204740942049037,
            "p_value": 4.59862408907509e-45,
            "se": 0.19971646905559512,
            "t_stat": 14.086489055629269,
        },
        "overall_att_es": 2.8461981758499997,
        "overall_se_es": 0.20425902876397498,
    },
    "stag_reg_cov": {
        "group_time": {
            "3.0|2": {"effect": 0.2776783805103399, "se": 0.33365048698863964},
            "3.0|3": {"effect": 3.076572394024985, "se": 0.28703632625261727},
            "3.0|4": {"effect": 2.649951806165864, "se": 0.37479998504995804},
            "3.0|5": {"effect": 2.7012789558536996, "se": 0.31653025014644537},
            "3.0|6": {"effect": 3.0420008720472342, "se": 0.3881434808314953},
            "4.0|2": {"effect": 0.08041142066229977, "se": 0.3049263704997503},
            "4.0|3": {"effect": 0.2858927581065889, "se": 0.3264180220652882},
            "4.0|4": {"effect": 2.769419984274427, "se": 0.36003067748880446},
            "4.0|5": {"effect": 2.733860817994398, "se": 0.2947015467495497},
            "4.0|6": {"effect": 2.691465318642148, "se": 0.36282711170283977},
        },
        "overall": {
            "att": 2.8092214498575365,
            "conf_int_lower": 2.4288798195291026,
            "conf_int_upper": 3.1895630801859705,
            "p_value": 1.7084282921997317e-47,
            "se": 0.1940554180222292,
            "t_stat": 14.476387613850276,
        },
        "overall_att_es": 2.8383188776312487,
        "overall_se_es": 0.20378953973090694,
    },
    "stag_dr_survey_pweight": {
        "group_time": {
            "3.0|2": {"effect": -0.15168667565630167, "se": 0.2957758221514469},
            "3.0|3": {"effect": 2.6028992505775195, "se": 0.32935262794723136},
            "3.0|4": {"effect": 2.6888855004876224, "se": 0.3918630493389082},
            "3.0|5": {"effect": 2.9347276471371493, "se": 0.3423394186475062},
            "3.0|6": {"effect": 3.400335285909308, "se": 0.4190520610387603},
            "4.0|2": {"effect": 0.15529987882584567, "se": 0.30936054767650695},
            "4.0|3": {"effect": -0.7550528967872525, "se": 0.32574440796417875},
            "4.0|4": {"effect": 3.494966537949268, "se": 0.3614205225792733},
            "4.0|5": {"effect": 3.717983004091062, "se": 0.3845593915382934},
            "4.0|6": {"effect": 3.8371645373489773, "se": 0.39151964978005016},
        },
        "overall": {
            "att": 3.2649376305989826,
            "conf_int_lower": 2.789054729251909,
            "conf_int_upper": 3.7408205319460563,
            "p_value": 4.606519827221944e-24,
            "se": 0.23970918564291063,
            "t_stat": 13.620411006955264,
        },
        "overall_att_es": 3.28298050259862,
        "overall_se_es": 0.25007641854869433,
    },
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

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
# add_covariates=True feeds x1/x2 into the OUTCOME, so this is a different draw
# than STAG_KW - the covariate oracles pin against it, not against stag_dr_all.
STAG_COV_KW = dict(STAG_KW, add_covariates=True)
COVS = ["age", "education"]
STAG_COVS = ["x1", "x2"]


# The dying class emits a FutureWarning per construction; the pyproject filter
# keeps the legacy suites quiet, but a test-local simplefilter RESETS the filter
# list, so build it through this helper inside any recording block.
def _sddd(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return StaggeredTripleDifference(**kwargs)


@pytest.fixture(scope="module")
def cross():
    return generate_ddd_data(**CROSS_KW)


@pytest.fixture(scope="module")
def panel():
    return generate_ddd_panel_data(**PANEL_KW)


@pytest.fixture(scope="module")
def stag():
    return generate_staggered_ddd_data(**STAG_KW)


@pytest.fixture(scope="module")
def stag_cov():
    return generate_staggered_ddd_data(**STAG_COV_KW)


@pytest.fixture(scope="module")
def stag_survey(stag):
    """Staggered panel with a per-UNIT pweight (weights must not vary within a
    unit - a within-unit-varying weight is a different design entirely)."""
    df = stag.copy()
    rng = np.random.default_rng(11)
    units = sorted(df["unit"].unique())
    per_unit = dict(zip(units, rng.uniform(0.5, 2.0, size=len(units))))
    df["w"] = df["unit"].map(per_unit)
    return df


C_COLS = dict(outcome="outcome", group="group", partition="partition", post="time")
S_NEW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")
S_OLD = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _fit_new(df, **kwargs):
    """Staggered fit through the MERGED surface."""
    return TripleDifference(**kwargs.pop("ctor", {})).fit(
        df, partition="eligibility", **S_NEW, **kwargs
    )


def _fit_old(df, **kwargs):
    """Staggered fit through the DEPRECATED surface."""
    return _sddd(**kwargs.pop("ctor", {})).fit(df, eligibility="eligibility", **S_OLD, **kwargs)


def _eq_with_nans(a, b):
    """NaN-mask equality first, then the finite subset - so a NaN-vs-0.0
    regression cannot be masked by nan_to_num-style comparison."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert_array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert_array_equal(a[m], b[m])


def _eq_dicts_with_nans(a, b, keys=("effect", "se")):
    """Identical key sets in identical ITERATION order, then values.

    Order matters: the bootstrap draw order depends on cell order, so a
    reordering would silently change seeded results.
    """
    assert list(a.keys()) == list(b.keys())
    for k in a:
        for f in keys:
            _eq_with_nans(a[k][f], b[k][f])


def _quintet(r):
    ci = r.conf_int
    return np.array([r.att, r.se, r.t_stat, r.p_value, ci[0], ci[1]], dtype=float)


# ---------------------------------------------------------------------------
# Gate A - the 2x2x2 engine is unchanged (committed pre-merge oracle)
# ---------------------------------------------------------------------------


@requires_python_backend
class TestGateA2x2x2Unchanged:
    """The rewritten fit prologue/signature/dispatch moved no 2x2x2 numbers.

    There is no surviving in-process pre-merge path to compare against: direct
    construction and the triple_difference() wrapper both route through the
    modified class, so a shared regression would be invisible to a
    self-comparison. Hence the committed literals.
    """

    @pytest.mark.parametrize("method", ["dr", "reg", "ipw"])
    def test_cross_sectional_lanes_match_oracle(self, cross, method):
        exp = ORACLE_2X2X2[f"cross_{method}"]
        r = TripleDifference(estimation_method=method).fit(cross, **C_COLS)
        assert_allclose(
            _quintet(r),
            [
                exp["att"],
                exp["se"],
                exp["t_stat"],
                exp["p_value"],
                exp["conf_int_lower"],
                exp["conf_int_upper"],
            ],
            rtol=ORACLE_RTOL,
            atol=ORACLE_ATOL,
        )
        assert r.n_obs == exp["n_obs"]
        assert r.n_treated_eligible == exp["n_treated_eligible"]
        assert r.n_control_ineligible == exp["n_control_ineligible"]
        assert r.vcov_type == exp["vcov_type"]
        assert r.cluster_name == exp["cluster_name"]
        assert r.n_clusters == exp["n_clusters"]

    def test_covariate_lane_matches_oracle(self, cross):
        exp = ORACLE_2X2X2["cross_dr_cov"]
        r = TripleDifference(estimation_method="dr").fit(cross, covariates=COVS, **C_COLS)
        assert_allclose(r.att, exp["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.se, exp["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)

    def test_survey_pweight_lane_matches_oracle(self, cross):
        exp = ORACLE_2X2X2["cross_survey_pweight_reg"]
        dfw = cross.copy()
        dfw["w"] = np.random.default_rng(7).uniform(0.5, 2.0, size=len(dfw))
        r = TripleDifference(estimation_method="reg").fit(
            dfw, survey_design=SurveyDesign(weights="w"), **C_COLS
        )
        assert_allclose(r.att, exp["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.se, exp["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)

    def test_cluster_lane_matches_oracle(self, panel):
        """The cluster axis runs on the PANEL generator deliberately: the
        cross-sectional one emits unit_id incremented once per ROW (and no
        `unit` column), so every cluster would be a singleton."""
        exp = ORACLE_2X2X2["panel_dr_cluster_unit"]
        r = TripleDifference(estimation_method="dr", cluster="unit").fit(
            panel, outcome="outcome", group="group", partition="partition", post="post"
        )
        assert_allclose(r.att, exp["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.se, exp["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert r.cluster_name == "unit"
        assert r.n_clusters == exp["n_clusters"] > 1

    def test_post_still_lands_in_slot_5(self, cross):
        """Positional slot 5 is `post`; the M-031 pin passes it positionally."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # positional canonical form must not warn
            positional = TripleDifference().fit(cross, "outcome", "group", "partition", "time")
        keyword = TripleDifference().fit(cross, **C_COLS)
        _eq_with_nans(_quintet(positional), _quintet(keyword))

    def test_time_alias_still_warns_and_routes_identically(self, cross):
        with pytest.warns(FutureWarning, match="calendar column only"):
            aliased = TripleDifference().fit(
                cross, outcome="outcome", group="group", partition="partition", time="time"
            )
        _eq_with_nans(_quintet(aliased), _quintet(TripleDifference().fit(cross, **C_COLS)))


# ---------------------------------------------------------------------------
# Gate B' - the RELOCATED staggered engine against its own committed oracle
# ---------------------------------------------------------------------------


@requires_python_backend
class TestGateBPrimeStaggeredOracle:
    """The engine that actually moved, pinned in ABSOLUTE terms.

    Gate B below compares two callers of the same relocated core, so it cannot
    see a transcription slip in the move itself; the SDDD suites carry no
    committed numeric pins; and the R-golden lane skips whenever the gitignored
    CSVs are absent. This gate is the one that would catch it.
    """

    def test_base_config_matches_oracle(self, stag):
        exp = ORACLE_STAGGERED["stag_dr_all"]
        r = _fit_new(stag, aggregate="all")
        assert_allclose(r.overall_att, exp["overall"]["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.overall_se, exp["overall"]["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.overall_att_es, exp["overall_att_es"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert r.n_obs == exp["n_obs"]
        assert r.n_never_enabled == exp["n_never_enabled"]
        got = {f"{g}|{t}": v for (g, t), v in sorted(r.group_time_effects.items())}
        assert set(got) == set(exp["group_time"])
        for k, v in exp["group_time"].items():
            assert_allclose(got[k]["effect"], v["effect"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(got[k]["se"], v["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)

    def test_bootstrap_config_matches_oracle(self, stag):
        exp = ORACLE_STAGGERED["stag_dr_all_boot49"]
        r = _fit_new(stag, aggregate="all", ctor=dict(n_bootstrap=49, seed=7, cband=True))
        assert_allclose(r.overall_att, exp["overall"]["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.overall_se, exp["overall"]["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(
            r.cband_crit_value, exp["cband_crit_value"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL
        )

    # -- branch coverage for the rest of the relocated engine -----------------
    # The two lanes above ride DR-without-covariates. _compute_pscore,
    # _compute_or, the never-treated comparison fork and the survey path all
    # moved too, and Gate B cannot see a slip in any of them (both of its sides
    # run the moved code). These are absolute pins on each.

    def test_never_treated_branch_matches_oracle(self, stag):
        """The _is_never_treated fork. The merged surface passes the UNDERSCORED
        spelling; the oracle was captured through the dying class's compact one,
        so this also pins the vocabulary bridge onto real numbers."""
        exp = ORACLE_STAGGERED["stag_dr_nevertreated"]
        r = _fit_new(stag, aggregate="all", ctor=dict(control_group="never_treated"))
        self._assert_against(r, exp)
        # ...and it must actually be a DIFFERENT comparison group, else a bridge
        # that collapsed every spelling to one branch would still pass above.
        base = ORACLE_STAGGERED["stag_dr_all"]["overall"]["att"]
        assert exp["overall"]["att"] != base

    @pytest.mark.parametrize("method", ["dr", "ipw", "reg"])
    def test_covariate_lanes_match_oracle(self, stag_cov, method):
        """Covariates are what separate the three nuisance models, so this is
        the lane that would actually catch a _compute_pscore/_compute_or slip."""
        exp = ORACLE_STAGGERED[f"stag_{method}_cov"]
        r = _fit_new(
            stag_cov,
            aggregate="all",
            covariates=STAG_COVS,
            ctor=dict(estimation_method=method),
        )
        self._assert_against(r, exp)

    def test_covariate_lanes_are_mutually_distinct(self):
        """Guard on the guard: if the three covariate oracles ever coincide, the
        parametrized test above silently stops discriminating between the
        nuisance models even while passing."""
        atts = {ORACLE_STAGGERED[f"stag_{m}_cov"]["overall"]["att"] for m in ("dr", "ipw", "reg")}
        assert len(atts) == 3

    def test_methods_converge_without_covariates(self, stag):
        """Why there is no per-method no-covariate oracle: with no covariates the
        propensity score is constant and the outcome regression is a bare mean,
        so dr/ipw/reg collapse to the same estimator. Asserted live rather than
        committed as three identical literal blocks that would LOOK like
        nuisance-model coverage while providing none."""
        fits = [
            _fit_new(stag, aggregate="all", ctor=dict(estimation_method=m))
            for m in ("dr", "ipw", "reg")
        ]
        for r in fits[1:]:
            assert_allclose(r.overall_att, fits[0].overall_att, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(r.overall_se, fits[0].overall_se, rtol=ORACLE_RTOL, atol=ORACLE_ATOL)

    def test_survey_pweight_matches_oracle(self, stag_survey):
        exp = ORACLE_STAGGERED["stag_dr_survey_pweight"]
        r = _fit_new(stag_survey, aggregate="all", survey_design=SurveyDesign(weights="w"))
        self._assert_against(r, exp)
        # the weights must actually bite
        assert exp["overall"]["att"] != ORACLE_STAGGERED["stag_dr_all"]["overall"]["att"]

    @staticmethod
    def _assert_against(r, exp):
        assert_allclose(r.overall_att, exp["overall"]["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.overall_se, exp["overall"]["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        assert_allclose(r.overall_att_es, exp["overall_att_es"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
        got = {f"{g}|{t}": v for (g, t), v in sorted(r.group_time_effects.items())}
        assert set(got) == set(exp["group_time"])
        for k, v in exp["group_time"].items():
            assert_allclose(got[k]["effect"], v["effect"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)
            assert_allclose(got[k]["se"], v["se"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)


# ---------------------------------------------------------------------------
# Gate B - merged staggered mode == deprecated class, bit-exact
# ---------------------------------------------------------------------------

# ONE AXIS AT A TIME against a fixed base, not a Cartesian product: the full
# cross-product is ~5.8k configs x two fits x bootstrap draws, which is not
# runnable. Named interaction cells cover the couplings that matter.
_BASE = dict(estimation_method="dr")
_AXES = [
    ("method_ipw", dict(estimation_method="ipw"), {}),
    ("method_reg", dict(estimation_method="reg"), {}),
    ("base_period_universal", dict(base_period="universal"), {}),
    ("anticipation_1", dict(anticipation=1), {}),
    ("cband_off", dict(n_bootstrap=49, seed=7, cband=False), {}),
    ("weights_mammen", dict(n_bootstrap=49, seed=7, bootstrap_weights="mammen"), {}),
    ("weights_webb", dict(n_bootstrap=49, seed=7, bootstrap_weights="webb"), {}),
    ("aggregate_none", {}, dict(aggregate=None)),
    ("aggregate_event_study", {}, dict(aggregate="event_study")),
    ("aggregate_group", {}, dict(aggregate="group")),
    ("aggregate_simple", {}, dict(aggregate="simple")),
    ("balance_e", {}, dict(aggregate="event_study", balance_e=1)),
    ("covariates", {}, dict(covariates=["x1"])),
    # named interaction cells
    ("boot_x_cband", dict(n_bootstrap=49, seed=7, cband=True), dict(aggregate="all")),
    ("universal_x_anticipation", dict(base_period="universal", anticipation=1), {}),
]

_STAG_FIELDS = [
    "overall_att",
    "overall_se",
    "overall_t_stat",
    "overall_p_value",
    "overall_att_es",
    "overall_se_es",
    "overall_t_stat_es",
    "overall_p_value_es",
    "n_obs",
    "n_treated_units",
    "n_control_units",
    "n_never_enabled",
    "n_eligible",
    "n_ineligible",
    "cband_crit_value",
]


def _assert_staggered_parity(old, new):
    assert type(new) is type(old) is StaggeredTripleDiffResults
    for f in _STAG_FIELDS:
        a, b = getattr(old, f), getattr(new, f)
        if a is None or b is None:
            assert a is b, f"{f}: {a!r} vs {b!r}"
        else:
            _eq_with_nans(a, b)
    _eq_with_nans(
        np.asarray(old.overall_conf_int, dtype=float), np.asarray(new.overall_conf_int, dtype=float)
    )
    _eq_dicts_with_nans(old.group_time_effects, new.group_time_effects)
    for attr in ("event_study_effects", "group_effects"):
        a, b = getattr(old, attr), getattr(new, attr)
        assert (a is None) == (b is None)
        if a is not None:
            _eq_dicts_with_nans(a, b, keys=("effect", "se"))


class TestGateBStaggeredParity:
    """Every lane: merged staggered mode is BIT-EXACT vs the dying class."""

    @pytest.mark.parametrize("label,ctor,fit_kw", _AXES, ids=[a[0] for a in _AXES])
    def test_axis_parity(self, stag, label, ctor, fit_kw):
        full_ctor = {**_BASE, **ctor}
        base_fit = dict(aggregate="all")
        base_fit.update(fit_kw)
        if "covariates" in fit_kw:
            df = generate_staggered_ddd_data(**STAG_KW, add_covariates=True)
            base_fit["covariates"] = ["x1"]
        else:
            df = stag
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = _fit_old(df, ctor=dict(full_ctor), **base_fit)
            new = _fit_new(df, ctor=dict(full_ctor), **base_fit)
        _assert_staggered_parity(old, new)

    @pytest.mark.parametrize(
        "new_value,old_value",
        [("not_yet_treated", "notyettreated"), ("never_treated", "nevertreated")],
    )
    def test_control_group_vocabulary_parity(self, stag, new_value, old_value):
        """Each class takes only ITS OWN spelling, so a parity pair is the
        underscored value on the new surface vs the compact one on the old."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = _fit_old(stag, ctor=dict(control_group=old_value), aggregate="all")
            new = _fit_new(stag, ctor=dict(control_group=new_value), aggregate="all")
        _assert_staggered_parity(old, new)

    def test_control_group_values_are_SEMANTICALLY_different(self, stag):
        """Guards a _is_never_treated that collapses both spellings.

        Parity alone cannot catch it: both surfaces route through the same
        helper, so a collapsing helper keeps every parity gate green while
        silently switching the comparison-group semantics. Nothing else in the
        repo differentiates the two values (the R-golden lane skips in CI).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nyt = _fit_new(stag, ctor=dict(control_group="not_yet_treated"), aggregate="all")
            nev = _fit_new(stag, ctor=dict(control_group="never_treated"), aggregate="all")
        assert nyt.overall_att != nev.overall_att, (
            "not_yet_treated and never_treated produced identical ATTs - the "
            "comparison-group branch is not reading control_group"
        )

    def test_first_treat_inf_recode_warning_parity(self, stag):
        df = stag.copy()
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["first_treat"] == 0, "first_treat"] = np.inf
        msgs = []
        for fit in (_fit_old, _fit_new):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fit(df, aggregate="simple")
            msgs.append([str(x.message) for x in w if "first_treat=inf" in str(x.message)])
        assert msgs[0] == msgs[1] and msgs[0], msgs

    def test_bootstrap_x_replicate_survey_raises_on_BOTH_surfaces(self, stag):
        """Excluded from the parity matrix by construction: this combination
        raises before any parity field exists, so it is exception-parity."""
        df = stag.copy()
        rng = np.random.default_rng(3)
        df["w"] = 1.0
        # Replicate weights must be UNIT-constant for a panel estimator, so
        # draw per unit and map onto rows.
        units = df["unit"].unique()
        for i in range(1, 4):
            per_unit = dict(zip(units, rng.uniform(0.5, 1.5, size=len(units))))
            df[f"rw{i}"] = df["unit"].map(per_unit)
        design = SurveyDesign(
            weights="w",
            replicate_weights=[f"rw{i}" for i in range(1, 4)],
            replicate_method="JK1",
        )
        for fit, name in ((_fit_old, "StaggeredTripleDifference"), (_fit_new, "TripleDifference")):
            with pytest.raises(NotImplementedError) as exc:
                fit(df, ctor=dict(n_bootstrap=49, seed=7), survey_design=design)
            assert name in str(exc.value), f"{name} not named: {exc.value}"


# ---------------------------------------------------------------------------
# Gate C - the rejection matrix, both directions
# ---------------------------------------------------------------------------


class TestGateCRejectionMatrix:
    def test_staggered_rejects_2x2x2_fit_params(self, stag):
        for kw in ({"post": "period"}, {"group": "eligibility"}):
            with pytest.raises(ValueError, match=r"belong\(s\) to the 2x2x2 mode"):
                _fit_new(stag, **kw)

    @pytest.mark.parametrize("kw", [{"unit": "unit"}, {"aggregate": "all"}, {"balance_e": 1}])
    def test_2x2x2_rejects_staggered_fit_params(self, cross, kw):
        with pytest.raises(ValueError, match=r"require\(s\) first_treat="):
            TripleDifference().fit(cross, **C_COLS, **kw)

    @pytest.mark.parametrize(
        "param,value",
        [
            ("control_group", "never_treated"),
            ("anticipation", 1),
            ("base_period", "universal"),
            ("n_bootstrap", 9),
        ],
    )
    def test_2x2x2_rejects_staggered_ctor_params(self, cross, param, value):
        with pytest.raises(ValueError, match="applies to the staggered DDD mode only"):
            TripleDifference(**{param: value}).fit(cross, **C_COLS)

    def test_staggered_rejects_robust(self, stag):
        with pytest.warns(FutureWarning, match="robust"):
            est = TripleDifference(robust=True)
        with pytest.raises(ValueError, match="applies to the 2x2x2 mode only"):
            est.fit(stag, partition="eligibility", **S_NEW)

    def test_staggered_rejects_mutated_vcov_type(self, stag):
        """vcov_type is validated eagerly in __init__ and by set_params' probe
        re-init, so its ONLY live route is direct attribute mutation - the
        bypass __init__'s comment documents. This arm is that guard."""
        est = TripleDifference()
        est.vcov_type = "classical"
        with pytest.raises(ValueError, match="applies to the 2x2x2 mode only"):
            est.fit(stag, partition="eligibility", **S_NEW)

    def test_staggered_rejects_cluster(self, stag):
        with pytest.raises(ValueError, match="not supported in staggered DDD mode"):
            TripleDifference(cluster="unit").fit(stag, partition="eligibility", **S_NEW)

    @pytest.mark.parametrize(
        "param,value",
        [
            ("bootstrap_weights", "webb"),
            ("seed", 7),
            ("cband", False),
            ("alpha", 0.10),
            ("pscore_trim", 0.02),
            ("rank_deficient_action", "silent"),
            ("epv_threshold", 5),
            ("pscore_fallback", "unconditional"),
        ],
    )
    def test_shared_and_unreachable_params_pass_silently_in_2x2x2(self, cross, param, value):
        """The six shared params must stay silent, and the three
        bootstrap satellites are unreachable-by-construction (n_bootstrap > 0
        is itself rejected) rather than silently ignored."""
        r = TripleDifference(**{param: value}).fit(cross, **C_COLS)
        assert np.isfinite(r.att)

    def test_bad_aggregate_VALUE_precedes_the_mode_error(self, cross):
        """Deliberate ordering (3(a)'s `spec` precedent): a bad value is
        reported as a bad value even in the wrong mode."""
        with pytest.raises(ValueError, match="aggregate must be"):
            TripleDifference().fit(cross, aggregate="bogus", **C_COLS)

    @pytest.mark.parametrize("missing", ["group", "partition", "post"])
    def test_2x2x2_required_args_named_individually(self, cross, missing):
        kw = dict(C_COLS)
        kw.pop(missing)
        with pytest.raises(TypeError, match=f"missing required argument: '{missing}'"):
            TripleDifference().fit(cross, **kw)

    def test_missing_group_raises_without_the_rename_warning(self, cross):
        """require_arg for group/partition runs BEFORE the M-031 shim: after
        it, this call would newly emit the rename FutureWarning before raising,
        and under -W error the surfaced exception type would flip."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError, match="missing required argument: 'group'"):
                TripleDifference().fit(cross, outcome="outcome", time="time")

    @pytest.mark.parametrize("missing", ["unit", "time", "partition"])
    def test_staggered_required_args_named_individually(self, stag, missing):
        kw = dict(partition="eligibility", **S_NEW)
        kw.pop(missing)
        with pytest.raises(TypeError, match=f"missing required argument: '{missing}'"):
            TripleDifference().fit(stag, **kw)

    def test_staggered_params_are_keyword_only(self, stag):
        """A positionally-written staggered call must be a clean TypeError, not
        a silent bind into the 2x2x2 slots."""
        with pytest.raises(TypeError):
            TripleDifference().fit(
                stag, "outcome", "unit", "period", "first_treat", "eligibility", "extra", "x", "y"
            )


# ---------------------------------------------------------------------------
# Gate D - time= semantics across the two modes
# ---------------------------------------------------------------------------


class TestGateDTimeSemantics:
    def test_staggered_time_emits_no_rename_warning(self, stag):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _fit_new(stag, aggregate="simple")
        assert not [x for x in w if "calendar column only" in str(x.message)]

    def test_staggered_time_is_bit_exact_vs_the_dying_surface(self, stag):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_staggered_parity(
                _fit_old(stag, aggregate="all"), _fit_new(stag, aggregate="all")
            )

    def test_2x2x2_both_post_and_time_raises(self, cross):
        with pytest.raises(ValueError, match="pass only post="):
            TripleDifference().fit(
                cross,
                outcome="outcome",
                group="group",
                partition="partition",
                post="time",
                time="time",
            )

    def test_m085_is_not_live_in_3_9(self, cross):
        """2x2x2 time= WARNS; it does not raise. The 4.0 flip is M-085's job."""
        with pytest.warns(FutureWarning):
            r = TripleDifference().fit(
                cross, outcome="outcome", group="group", partition="partition", time="time"
            )
        assert np.isfinite(r.att)


# ---------------------------------------------------------------------------
# Gate E - deprecation choreography (M-013 / M-064)
# ---------------------------------------------------------------------------


class TestGateEDeprecation:
    def test_construction_warns_with_the_pinned_message(self):
        with pytest.warns(FutureWarning, match=re.escape(_SDDD_DEPRECATION_MSG)):
            StaggeredTripleDifference()

    def test_sddd_alias_is_the_same_class_and_warns_once(self):
        assert SDDD is StaggeredTripleDifference
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SDDD()
        fired = [
            x
            for x in w
            if x.category is FutureWarning
            and "StaggeredTripleDifference is deprecated" in str(x.message)
        ]
        assert len(fired) == 1, [str(x.message) for x in w]
        assert "SDDD alias is deprecated with it" in str(fired[0].message)

    def test_warns_and_still_works(self, stag):
        with pytest.warns(FutureWarning):
            est = StaggeredTripleDifference()
        r = est.fit(stag, eligibility="eligibility", **S_OLD, aggregate="simple")
        assert np.isfinite(r.overall_att)

    def test_successor_does_not_warn(self, stag):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TripleDifference()
        assert not [x for x in w if x.category is FutureWarning]

    def test_deprecation_attributes_to_the_caller(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StaggeredTripleDifference()
        assert w[0].filename.endswith("test_v4_merge_ddd.py")

    def test_set_params_re_emits(self):
        """Documented side effect of BaseEstimator's transactional probe
        re-init, shared with MultiPeriodDiD's shim."""
        est = _sddd()
        with pytest.warns(FutureWarning, match="StaggeredTripleDifference is deprecated"):
            est.set_params(alpha=0.10)


# ---------------------------------------------------------------------------
# Gate F - introspection, the new validation contracts, cross-mode refit
# ---------------------------------------------------------------------------

_MERGED_PARAMS = {
    "estimation_method",
    "robust",
    "cluster",
    "vcov_type",
    "alpha",
    "pscore_trim",
    "rank_deficient_action",
    "epv_threshold",
    "pscore_fallback",
    "control_group",
    "anticipation",
    "base_period",
    "n_bootstrap",
    "bootstrap_weights",
    "seed",
    "cband",
}


class TestGateFIntrospectionAndValidation:
    def test_merged_param_set_is_pinned(self):
        assert set(TripleDifference().get_params()) == _MERGED_PARAMS

    def test_dying_class_param_set_is_frozen(self):
        assert set(_sddd().get_params()) == {
            "estimation_method",
            "control_group",
            "alpha",
            "anticipation",
            "base_period",
            "n_bootstrap",
            "bootstrap_weights",
            "seed",
            "cband",
            "pscore_trim",
            "cluster",
            "rank_deficient_action",
            "epv_threshold",
            "pscore_fallback",
        }

    def test_round_trip_both_classes(self):
        est = TripleDifference(control_group="never_treated", n_bootstrap=9)
        assert TripleDifference(**est.get_params()).get_params() == est.get_params()
        old = _sddd(control_group="nevertreated")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            assert StaggeredTripleDifference(**old.get_params()).get_params() == old.get_params()

    def test_set_params_round_trip(self):
        est = TripleDifference()
        assert est.set_params(alpha=0.10) is est
        assert est.get_params()["alpha"] == 0.10

    @pytest.mark.parametrize("bad", [0, 0.5, -0.1, 0.9, float("nan"), float("inf")])
    def test_pscore_trim_boundaries_rejected(self, bad):
        with pytest.raises(ValueError, match=r"pscore_trim must be in \(0, 0.5\)"):
            TripleDifference(pscore_trim=bad)

    @pytest.mark.parametrize(
        "bad", [None, "0.01", 1j, [0.01], (0.01,), np.array([0.01]), True, False]
    )
    def test_pscore_trim_non_scalar_rejected(self, bad):
        """The TYPE guard, not just the range one. A bare `0 < x < 0.5` raises an
        incidental TypeError on None/str/complex/list, an ambiguous-truth error on
        a multi-element array, and ACCEPTS a 1-element array - storing an ndarray
        as the parameter, which then rides into np.clip(pscore, trim, 1 - trim)."""
        with pytest.raises(ValueError, match="pscore_trim must be"):
            TripleDifference(pscore_trim=bad)

    def test_pscore_trim_multi_element_array_rejected(self):
        with pytest.raises(ValueError, match="pscore_trim must be"):
            TripleDifference(pscore_trim=np.array([0.01, 0.02]))

    @pytest.mark.parametrize("good", [0.01, 0.49, 0.001, np.float64(0.02), np.float32(0.02)])
    def test_pscore_trim_interior_accepted(self, good):
        assert TripleDifference(pscore_trim=good).pscore_trim == good

    def test_pscore_trim_coerced_to_builtin_float(self):
        """The shared utils.validate_pscore_trim helper coerces on store."""
        assert type(TripleDifference(pscore_trim=np.float32(0.02)).pscore_trim) is float

    def test_set_params_pscore_trim_is_transactional(self):
        est = TripleDifference(pscore_trim=0.01)
        with pytest.raises(ValueError):
            est.set_params(pscore_trim=0)
        assert est.pscore_trim == 0.01, "probe re-init must not mutate on failure"

    # -- anticipation domain (silent-estimand guard) --------------------------

    @pytest.mark.parametrize("bad", [-1, -5, 1.5, 0.5, "1", None, True, False, np.float64(2.0)])
    def test_anticipation_domain_rejected(self, bad):
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            TripleDifference(anticipation=bad)

    @pytest.mark.parametrize("good", [0, 1, 3, np.int64(2)])
    def test_anticipation_domain_accepted(self, good):
        assert TripleDifference(anticipation=good).anticipation == good

    def test_set_params_anticipation_is_transactional(self):
        est = TripleDifference(anticipation=1)
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            est.set_params(anticipation=-1)
        assert est.anticipation == 1, "probe re-init must not mutate on failure"

    def test_negative_anticipation_cannot_reach_the_engine(self, stag):
        """The behavioral point of the guard, not just the raise: a negative
        window would make the universal base period `g` - an ALREADY-TREATED
        period - and relax the not-yet-treated threshold to max(t, base) - 1,
        admitting cohorts treated at the evaluation period as clean controls.
        Neither is visible in the output, so the constructor is the only place
        it can be stopped."""
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            TripleDifference(anticipation=-1, base_period="universal")
        # the guard is EAGER: it fires at construction, so no fit can be reached
        # with an out-of-domain window even via the staggered branch.
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            TripleDifference(anticipation=-1).fit(
                stag, partition="eligibility", **S_NEW, aggregate="simple"
            )

    @pytest.mark.parametrize("bad", [-1, 1.5, True])
    def test_deprecated_sibling_also_fails_closed_on_anticipation(self, stag, bad):
        """The dying class's frozen 3.x API SHAPE (param names, `eligibility=`,
        the compact control_group vocabulary, accepted-then-ignored `cluster=`)
        was never a licence to emit silently-biased numbers: it runs the SAME
        engine, where a negative window selects an already-treated base period
        and admits contaminated controls. So the guard lives in the engine and
        BOTH surfaces fail closed.

        The split is deliberate and pinned below: CONSTRUCTION still succeeds on
        the deprecated class (its signature contract is untouched), while FIT
        raises - the check is an identification guard, not an API change."""
        est = _sddd(anticipation=bad)
        assert est.anticipation == bad, "the dying class's constructor contract is unchanged"
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            est.fit(stag, eligibility="eligibility", **S_OLD, aggregate="simple")

    # -- first_treat cohort encoding (silent-population guard) ----------------

    @pytest.mark.parametrize("sentinel", [-1, -np.inf, -0.5])
    @pytest.mark.parametrize("surface", ["merged", "deprecated"])
    def test_negative_cohorts_fail_closed_on_both_surfaces(self, stag, sentinel, surface):
        """_precompute_structures builds the treated set from `g > 0` and the
        never-enabled set from `g == 0`, so a unit encoded with the common `-1`
        never-treated convention belonged to NEITHER: still counted in n_obs,
        contributing to no ATT comparison, and the fit returned a plausible
        finite estimate for a DIFFERENT population (measured before the guard:
        overall_att 3.29438582 -> 2.99470938, n_never_enabled 24 -> 0, silently).

        The +inf branch already defended this same input axis with an explicit
        recode-and-warn, so silence on negatives was a hole in an established
        contract. Raising rather than recoding: -1-as-never is a convention, not
        an unambiguous limit like +inf, and guessing would BE the silent sample
        change."""
        df = stag.copy()
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["first_treat"] == 0, "first_treat"] = sentinel
        fit = _fit_new if surface == "merged" else _fit_old
        with pytest.raises(ValueError, match="negative cohort value"):
            fit(df, aggregate="simple")

    # -- degenerate enabling cohort (honest-reporting guard) ------------------

    @staticmethod
    def _strip_eligibility_from_cohort(df, cohort):
        out = df.copy()
        units = out.loc[out["first_treat"] == cohort, "unit"].unique()
        out.loc[out["unit"].isin(units), "eligibility"] = 0
        return out

    @pytest.mark.parametrize("surface", ["merged", "deprecated"])
    def test_degenerate_cohort_is_named_and_dropped_from_groups(self, stag, surface):
        """A positive cohort whose units are ALL partition==0 cannot identify
        ATT(g,t), so it contributes to no aggregate - but it was still advertised
        in `groups`/`n_groups`, and no warning named it (the ones mentioning it
        describe its role as a COMPARISON cohort for other g). The estimate is
        deliberately unchanged: it is valid for the cohorts that do identify."""
        df = self._strip_eligibility_from_cohort(stag, 4)
        fit = _fit_new if surface == "merged" else _fit_old
        with pytest.warns(UserWarning, match=r"cohort g=4\.0 has no eligible treated units"):
            r = fit(df, aggregate="simple")
        assert list(r.groups) == [3.0], "groups must not advertise a non-contributing cohort"
        assert r.to_dict()["n_groups"] == 1
        assert {g for g, _ in r.group_time_effects} == {3.0}

    def test_degenerate_cohort_does_not_move_the_estimate(self, stag):
        """The fix is reporting-only - pinned so a later 'improvement' cannot
        quietly turn it into an estimand change."""
        df = self._strip_eligibility_from_cohort(stag, 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new, old = _fit_new(df, aggregate="simple"), _fit_old(df, aggregate="simple")
        _eq_with_nans([new.overall_att, new.overall_se], [old.overall_att, old.overall_se])
        assert list(new.groups) == list(old.groups)

    def test_healthy_cohorts_are_untouched(self, stag):
        """Negative pin: no warning and no groups change on a well-posed panel."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = _fit_new(stag, aggregate="simple")
        assert not [x for x in w if "no eligible treated units" in str(x.message)]
        assert list(r.groups) == [3.0, 4.0]

    def test_degenerate_cohort_under_survey_weights(self, stag_survey):
        """Survey counterpart: rows exist for the cohort but carry no eligible
        treated mass."""
        df = self._strip_eligibility_from_cohort(stag_survey, 4)
        with pytest.warns(UserWarning, match=r"cohort g=4\.0 has no eligible treated units"):
            r = _fit_new(df, aggregate="simple", survey_design=SurveyDesign(weights="w"))
        assert list(r.groups) == [3.0]

    def test_positive_inf_still_recodes_with_a_warning(self, stag):
        """The negative guard must not disturb the +inf lane it sits beside."""
        df = stag.copy()
        df["first_treat"] = df["first_treat"].astype(float)
        df.loc[df["first_treat"] == 0, "first_treat"] = np.inf
        with pytest.warns(UserWarning, match="recoding to 0"):
            r = _fit_new(df, aggregate="simple")
        plain = _fit_new(stag, aggregate="simple")
        _eq_with_nans([r.overall_att, r.overall_se], [plain.overall_att, plain.overall_se])
        assert r.n_never_enabled == plain.n_never_enabled

    def test_valid_cohorts_are_untouched_by_the_guard(self, stag):
        """Regression pin: the guard changes no accepted fit."""
        r = _fit_new(stag, aggregate="all")
        exp = ORACLE_STAGGERED["stag_dr_all"]
        assert_allclose(r.overall_att, exp["overall"]["att"], rtol=ORACLE_RTOL, atol=ORACLE_ATOL)

    def test_anticipation_guard_survives_attribute_mutation(self, stag):
        """The engine-level guard also covers the __init__/set_params bypass."""
        est = TripleDifference()
        est.anticipation = -1  # neither validator sees this
        with pytest.raises(ValueError, match="anticipation must be a non-negative integer"):
            est.fit(stag, partition="eligibility", **S_NEW, aggregate="simple")

    @pytest.mark.parametrize("compact", ["notyettreated", "nevertreated"])
    def test_merged_class_rejects_the_compact_vocabulary(self, compact):
        """The boundary Design 3's bridge and Gate B's pairing rest on."""
        with pytest.raises(ValueError, match="control_group must be"):
            TripleDifference(control_group=compact)

    @pytest.mark.parametrize("underscored", ["not_yet_treated", "never_treated"])
    def test_dying_class_rejects_the_underscored_vocabulary(self, underscored):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(ValueError, match="control_group must be"):
                StaggeredTripleDifference(control_group=underscored)

    def test_base_period_and_bootstrap_weights_contracts(self):
        with pytest.raises(ValueError, match="base_period must be"):
            TripleDifference(base_period="bogus")
        with pytest.raises(ValueError, match="bootstrap_weights must be"):
            TripleDifference(bootstrap_weights="bogus")

    def test_n_bootstrap_type_contract(self):
        for bad in (True, 2.5, -1, None):
            with pytest.raises(ValueError, match="n_bootstrap"):
                TripleDifference(n_bootstrap=bad)

    def test_cross_mode_refit_on_one_instance(self, cross, stag):
        """fit() writes mode-specific state (results_ changes TYPE, is_fitted_,
        _replicate_n_valid is reset only in the 2x2x2 prologue), so a refit
        across modes must match a fresh instance in both directions."""
        est = TripleDifference()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stag_first = est.fit(stag, partition="eligibility", **S_NEW, aggregate="all")
            assert isinstance(stag_first, StaggeredTripleDiffResults)
            then_cross = est.fit(cross, **C_COLS)
            assert isinstance(then_cross, TripleDifferenceResults)
            fresh_cross = TripleDifference().fit(cross, **C_COLS)
            _eq_with_nans(_quintet(then_cross), _quintet(fresh_cross))

            back_to_stag = est.fit(stag, partition="eligibility", **S_NEW, aggregate="all")
            _assert_staggered_parity(stag_first, back_to_stag)
        assert est.is_fitted_

    def test_frame_offset_is_restored_after_a_RAISING_staggered_fit(self, cross, stag):
        """The try/finally must restore the offset, or every later warning on
        the instance attributes to the wrong frame."""
        est = TripleDifference()
        bad = stag.copy()
        bad.loc[bad.index[0], "eligibility"] = 1 - bad.loc[bad.index[0], "eligibility"]
        with pytest.raises(ValueError):
            est.fit(bad, partition="eligibility", **S_NEW)
        assert est._warn_frame_offset == 0


# ---------------------------------------------------------------------------
# Gate G - warning attribution, message threading, the R lane
# ---------------------------------------------------------------------------


class TestGateGAttributionAndThreading:
    def test_user_attributed_warnings_land_on_the_CALLER_on_both_surfaces(self, stag):
        """The facade adds a frame; _frame_offset must absorb it. Uses the
        base-period-outside-panel warning, which fires from the fit body."""
        df = generate_staggered_ddd_data(**{**STAG_KW, "cohort_periods": [1, 3]})
        for fit in (_fit_old, _fit_new):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fit(df, aggregate="simple")
            fired = [x for x in w if "outside the observed panel" in str(x.message)]
            assert fired, "expected the base-period warning"
            assert all(x.filename.endswith("test_v4_merge_ddd.py") for x in fired), [
                x.filename for x in fired
            ]

    def test_low_bootstrap_warning_attributes_to_the_caller_on_both_surfaces(self, stag):
        """Shared with CallawaySantAnna via the bootstrap mixin, so it reads the
        offset through getattr rather than taking it as an argument."""
        for fit in (_fit_old, _fit_new):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                fit(stag, ctor=dict(n_bootstrap=9, seed=1), aggregate="simple")
            fired = [x for x in w if "is low" in str(x.message)]
            assert fired and all(x.filename.endswith("test_v4_merge_ddd.py") for x in fired), [
                (x.filename, str(x.message)[:40]) for x in fired
            ]

    def test_callawaysantanna_attribution_is_unchanged(self):
        """CS never sets _warn_frame_offset, so the shared site's getattr
        default keeps its attribution bit-identical to 3.x."""
        from diff_diff import CallawaySantAnna
        from diff_diff.prep_dgp import generate_staggered_data

        df = generate_staggered_data(n_units=60, n_periods=5, seed=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CallawaySantAnna(n_bootstrap=9, seed=1).fit(
                df, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
            )
        fired = [x for x in w if "is low" in str(x.message)]
        assert fired and all(x.filename.endswith("test_v4_merge_ddd.py") for x in fired)

    def test_survey_pweight_error_names_the_surface_that_was_fit(self, stag):
        df = stag.copy()
        df["w"] = 1.0
        design = SurveyDesign(weights="w", weight_type="fweight")
        for fit, name in ((_fit_old, "StaggeredTripleDifference"), (_fit_new, "TripleDifference")):
            with pytest.raises(ValueError) as exc:
                fit(df, survey_design=design)
            assert str(exc.value).startswith(f"{name} survey support requires")

    def test_partition_vocabulary_never_leaks_on_the_merged_surface(self, stag):
        """BOTH sentences of the time-invariance error are parameterized - the
        tail 'varying eligibility' would otherwise say the wrong word."""
        df = stag.copy()
        df.loc[df.index[0], "eligibility"] = 1 - df.loc[df.index[0], "eligibility"]
        with pytest.raises(ValueError) as exc:
            _fit_new(df)
        msg = str(exc.value)
        assert "Partition must be time-invariant" in msg
        assert "eligibility" not in msg, msg
        with pytest.raises(ValueError) as exc_old:
            _fit_old(df)
        assert "Eligibility must be time-invariant" in str(exc_old.value)
        assert "varying eligibility" in str(exc_old.value)

    def test_no_valid_group_time_message_uses_the_right_vocabulary(self, stag):
        """An anticipation window wider than the panel skips every cell, which
        is the cheapest route to this message. (Filtering the partition instead
        would trip the Q-notation validator first - that message is paper
        vocabulary and is deliberately NOT parameterized.)"""
        for fit, word, wrong in (
            (_fit_new, "partition", "eligibility"),
            (_fit_old, "eligibility", "partition"),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match="No valid group-time effects") as exc:
                    fit(stag, ctor=dict(anticipation=10))
            assert word in str(exc.value)
            assert wrong not in str(exc.value)


# ---------------------------------------------------------------------------
# Gate H - downstream consumers
# ---------------------------------------------------------------------------


class TestGateHConsumers:
    def test_event_study_surface_identical_across_surfaces(self, stag):
        from diff_diff.results_base import build_event_study_surface

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = build_event_study_surface(_fit_old(stag, aggregate="event_study"))
            new = build_event_study_surface(_fit_new(stag, aggregate="event_study"))
        _eq_with_nans(old.att, new.att)
        _eq_with_nans(old.se, new.se)
        assert old.source == new.source

    def test_business_report_handles_both_control_group_vocabularies(self, stag):
        from diff_diff.business_report import BusinessReport

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new = BusinessReport(
                _fit_new(stag, ctor=dict(control_group="not_yet_treated"), aggregate="simple")
            ).to_dict()
            old = BusinessReport(
                _fit_old(stag, ctor=dict(control_group="notyettreated"), aggregate="simple")
            ).to_dict()
        assert new["sample"]["n_never_enabled"] == old["sample"]["n_never_enabled"]

    def test_summary_renders_in_both_modes(self, cross, stag):
        """summary()/print_summary() delegate to results_, so the merged class
        renders the staggered container in staggered mode - new public
        behavior that follows for free but should be pinned."""
        est = TripleDifference()
        est.fit(cross, **C_COLS)
        assert "Triple Difference" in est.summary()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(stag, partition="eligibility", **S_NEW, aggregate="all")
        text = est.summary()
        assert "ATT" in text and len(text.splitlines()) > 10

    def test_describe_target_parameter_unchanged(self, stag):
        from diff_diff._reporting_helpers import describe_target_parameter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert describe_target_parameter(
                _fit_new(stag, aggregate="simple")
            ) == describe_target_parameter(_fit_old(stag, aggregate="simple"))

    def test_power_rejects_staggered_config_at_all_three_entry_points(self):
        from diff_diff.power import simulate_mde, simulate_power, simulate_sample_size

        for fn in (simulate_power, simulate_mde, simulate_sample_size):
            with pytest.raises(ValueError, match="staggered DDD mode"):
                fn(TripleDifference(control_group="never_treated"), n_simulations=1)
            with pytest.raises(ValueError, match="staggered DDD mode"):
                fn(TripleDifference(), n_simulations=1, estimator_kwargs={"first_treat": "g"})

    def test_power_still_works_on_defaults(self):
        from diff_diff.power import simulate_power

        r = simulate_power(TripleDifference(), n_units=64, n_simulations=2, seed=1)
        assert r is not None

    @pytest.mark.parametrize(
        "inert",
        [dict(seed=7), dict(bootstrap_weights="mammen"), dict(cband=False)],
    )
    def test_power_accepts_the_inert_bootstrap_satellites(self, inert):
        """The three params that fit() accepts in 2x2x2 mode as inert must not be
        rejected by power. They take effect only through n_bootstrap > 0, which
        2x2x2 mode already rejects, so a non-default value cannot signal a
        staggered configuration - and refusing them made the same estimator legal
        to fit and illegal to simulate. `seed` is the one that bites in practice:
        users set it habitually, and simulate_* take their own separate seed=."""
        from diff_diff.power import simulate_mde, simulate_power, simulate_sample_size

        # the fit really does accept it (the premise of this gate)
        cross = generate_ddd_data(**CROSS_KW)
        assert TripleDifference(**inert).fit(cross, **C_COLS) is not None

        assert (
            simulate_power(TripleDifference(**inert), n_units=64, n_simulations=2, seed=1)
            is not None
        )
        assert simulate_mde(
            TripleDifference(**inert),
            n_units=64,
            n_simulations=2,
            seed=1,
            max_steps=2,
            progress=False,
        )
        assert simulate_sample_size(
            TripleDifference(**inert),
            n_simulations=2,
            seed=1,
            max_steps=2,
            progress=False,
        )

    def test_deprecated_wrapper_stays_2x2x2_only(self, cross):
        """`triple_difference()` is deliberately NOT extended to the staggered
        mode. It is itself deprecated (row M-075, removed at 4.0), phase 3(a)
        extended no wrapper with its new mode, and adding params to a dying
        surface would mint fresh 4.0 removal obligations. The migration steer -
        'construct the estimator instead' - is the intended pressure. Pinned so
        the omission reads as a decision."""
        import inspect

        from diff_diff import triple_difference

        params = set(inspect.signature(triple_difference).parameters)
        staggered_only = {
            "unit",
            "first_treat",
            "aggregate",
            "balance_e",
            "control_group",
            "anticipation",
            "base_period",
            "n_bootstrap",
            "bootstrap_weights",
            "seed",
            "cband",
        }
        assert not (params & staggered_only), (
            "the deprecated wrapper grew a staggered-mode param; it is slated for "
            "4.0 removal and must not widen"
        )

        # ...and the 2x2x2 route it DOES own still works, warning exactly once
        with pytest.warns(FutureWarning, match="triple_difference.. is deprecated"):
            r = triple_difference(cross, "outcome", "group", "partition", "time")
        direct = TripleDifference().fit(cross, **C_COLS)
        _eq_with_nans(_quintet(r), _quintet(direct))

    @pytest.mark.parametrize("fn_name", ["simulate_power", "simulate_mde", "simulate_sample_size"])
    def test_power_accepts_explicit_none_aggregate_and_balance_e(self, cross, fn_name):
        """`aggregate`/`balance_e` default to None and fit() rejects only a
        NON-None value, so keying the power guard on KEY PRESENCE rejected
        `estimator_kwargs={"aggregate": None}` - a config fit() accepts. That
        broke the boundary the guard exists to uphold: legal to fit implies
        legal to simulate."""
        import diff_diff.power as power_mod

        kw = {"aggregate": None, "balance_e": None}
        assert TripleDifference().fit(cross, **C_COLS, **kw) is not None

        fn = getattr(power_mod, fn_name)
        extra = {"n_units": 48} if fn_name != "simulate_sample_size" else {}
        if fn_name != "simulate_power":
            extra.update(max_steps=2, progress=False)
        assert fn(TripleDifference(), n_simulations=2, seed=1, estimator_kwargs=kw, **extra)

    @pytest.mark.parametrize("key,value", [("aggregate", "simple"), ("balance_e", 1)])
    def test_power_still_rejects_non_none_aggregate_and_balance_e(self, key, value):
        from diff_diff.power import simulate_power

        with pytest.raises(ValueError, match="staggered DDD mode"):
            simulate_power(
                TripleDifference(), n_units=48, n_simulations=1, estimator_kwargs={key: value}
            )

    @pytest.mark.parametrize("key", ["first_treat", "unit"])
    @pytest.mark.parametrize("value", ["g", None])
    def test_power_rejects_mode_selectors_by_presence(self, key, value):
        """The other half of the same rule: `first_treat`/`unit` are
        sentinel-defaulted in fit(), so SUPPLYING them at all is the signal -
        an explicit None still selects staggered mode and then fails on a
        missing column. Presence is the correct test for these two."""
        from diff_diff.power import simulate_power

        with pytest.raises(ValueError, match="staggered DDD mode"):
            simulate_power(
                TripleDifference(), n_units=48, n_simulations=1, estimator_kwargs={key: value}
            )

    @pytest.mark.parametrize("surface", ["merged", "deprecated"])
    def test_bootstrap_warning_names_the_estimator_that_was_fit(self, stag, surface):
        """The single-PSU/degenerate-design warning lives in the mixin shared by
        CallawaySantAnna and BOTH DDD classes, so the hard-coded literal named
        CallawaySantAnna no matter which surface was fit - a misleading diagnosis
        on a fit that is failing closed.

        Routed through a real degenerate fit rather than asserted against source
        text: a source pin cannot see whether the interpolated value actually
        reaches the user."""
        df = stag.copy()
        df["w"] = 1.0
        df["psu"] = 0  # collapse to one PSU -> bootstrap variance unidentified
        df["fpc_col"] = 10**6  # design-based variance so the PSU is retained
        design = SurveyDesign(weights="w", psu="psu", fpc="fpc_col")
        ctor = dict(estimation_method="reg", n_bootstrap=20, seed=7)
        fit = _fit_new if surface == "merged" else _fit_old
        expected = "TripleDifference" if surface == "merged" else "StaggeredTripleDifference"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit(df, aggregate="event_study", survey_design=design, ctor=ctor)
        hits = [
            str(w.message)
            for w in caught
            if "bootstrap with survey/cluster design" in str(w.message)
        ]
        assert hits, "expected the single-PSU bootstrap warning"
        assert all(m.startswith(f"{expected} bootstrap") for m in hits), hits
        assert not any(m.startswith("CallawaySantAnna") for m in hits)

    def test_bootstrap_label_is_declared_on_every_host(self):
        """CallawaySantAnna shares the mixin and its message was already correct,
        so its label must stay byte-identical."""
        from diff_diff.staggered import CallawaySantAnna

        assert CallawaySantAnna._BOOTSTRAP_LABEL == "CallawaySantAnna"
        assert TripleDifference._BOOTSTRAP_LABEL == "TripleDifference"
        assert StaggeredTripleDifference._BOOTSTRAP_LABEL == "StaggeredTripleDifference"

    @pytest.mark.parametrize(
        "staggered", ["control_group", "anticipation", "base_period", "n_bootstrap"]
    )
    def test_power_still_rejects_the_genuinely_staggered_four(self, staggered):
        """The other side of the same boundary - narrowing the roster must not
        have opened a hole for the params that DO select staggered behavior."""
        from diff_diff.power import simulate_power

        value = {
            "control_group": "never_treated",
            "anticipation": 1,
            "base_period": "universal",
            "n_bootstrap": 49,
        }[staggered]
        with pytest.raises(ValueError, match="staggered DDD mode"):
            simulate_power(TripleDifference(**{staggered: value}), n_simulations=1)

    def test_staggered_roster_reads_live_constructor_defaults(self):
        """The guard compares each staggered-only param against its DEFAULT. If
        those defaults were hard-coded, a future constructor-default change would
        silently reclassify a plain TripleDifference() as staggered-configured
        and reject a legitimate 2x2x2 power run."""
        import inspect

        from diff_diff.utils import STAGGERED_DDD_CTOR_PARAMS, staggered_ddd_ctor_defaults

        sig = inspect.signature(TripleDifference.__init__).parameters
        missing = [n for n in STAGGERED_DDD_CTOR_PARAMS if n not in sig]
        assert not missing, f"roster names no longer on the constructor: {missing}"

        derived = staggered_ddd_ctor_defaults(TripleDifference())
        assert set(derived) == set(STAGGERED_DDD_CTOR_PARAMS)
        for name, default in derived.items():
            assert default == sig[name].default

        # and a default-constructed estimator is never an offender
        from diff_diff.power import _reject_staggered_ddd_config

        _reject_staggered_ddd_config(TripleDifference(), {})

    def test_fit_and_power_share_one_staggered_boundary(self):
        """The defect this centralization fixes: fit() used to hardcode the four
        defaults while power derived them, so a future default change could make
        the two mode-detection boundaries disagree about what
        'staggered-configured' means. Both now read the same helper - asserted
        by behavior on every rostered param, not just by shared imports."""
        from diff_diff.power import simulate_power
        from diff_diff.utils import STAGGERED_DDD_CTOR_PARAMS

        cross = generate_ddd_data(**CROSS_KW)
        non_default = {
            "control_group": "never_treated",
            "anticipation": 1,
            "base_period": "universal",
            "n_bootstrap": 49,
        }
        assert set(non_default) == set(STAGGERED_DDD_CTOR_PARAMS)
        for name, value in non_default.items():
            est = TripleDifference(**{name: value})
            with pytest.raises(ValueError, match="staggered"):
                est.fit(cross, **C_COLS)
            with pytest.raises(ValueError, match="staggered DDD mode"):
                simulate_power(TripleDifference(**{name: value}), n_simulations=1)


# ---------------------------------------------------------------------------
# Gate G (R lane) - the golden, plus its always-running substitute
# ---------------------------------------------------------------------------


class TestGateGRParity:
    """The R-golden lane SKIPS in CI: the generated CSVs are gitignored
    (.gitignore benchmarks/data/synthetic/*.csv), so `_load_r_data` calls
    pytest.skip. It is therefore paired with a substitute that always runs and
    proves the routing is identity-preserving on the same config.
    """

    def test_r_golden_through_the_merged_surface(self):
        """Local-only. Reuses the methodology suite's loader and tolerances
        rather than duplicating them; maps the golden's compact control_group
        to the merged class's underscored vocabulary."""
        import json

        from tests.test_methodology_staggered_triple_diff import (
            ATT_ATOL,
            ATT_RTOL,
            TestStaggeredDDDAggregation,
            _load_r_data,
        )

        results_file = (
            __import__("pathlib").Path(__file__).parent.parent
            / "benchmarks"
            / "data"
            / "synthetic"
            / "staggered_ddd_r_results.json"
        )
        if not results_file.exists():
            pytest.skip("R golden results not present")
        golden = json.loads(results_file.read_text())
        key = "s42_dgp1_dr_nyt"
        if key not in golden:
            pytest.skip(f"scenario {key} absent from the golden file")
        scenario = golden[key]
        df = _load_r_data(42, 1)  # skips when the gitignored CSV is absent

        # No aggregate=: the default simple aggregation is what the golden's
        # overall_att_simple records, and it is how the methodology suite fits.
        r = TripleDifference(estimation_method="dr", control_group="not_yet_treated").fit(
            df, partition="eligibility", **S_NEW
        )
        # Tolerances mirror the methodology suite EXACTLY rather than inventing
        # new ones: per-(g,t) cells are pinned tightly, while the simple overall
        # carries that suite's documented looser envelope (a weighting deviation
        # on individual values - see TestStaggeredDDDAggregation's docstring).
        assert_allclose(
            r.overall_att,
            scenario["overall_att_simple"],
            rtol=TestStaggeredDDDAggregation.AGG_RTOL,
            atol=TestStaggeredDDDAggregation.AGG_ATOL,
        )
        # the per-(g,t) table too, so the golden pins more than one scalar
        py_gt = dict(sorted(r.group_time_effects.items()))
        r_gt = list(zip(scenario["gt_groups"], scenario["gt_periods"]))
        assert len(py_gt) == len(r_gt), f"{len(py_gt)} cells vs R's {len(r_gt)}"
        for i, key_gt in enumerate(r_gt):
            assert_allclose(
                py_gt[key_gt]["effect"], scenario["gt_att"][i], rtol=ATT_RTOL, atol=ATT_ATOL
            )

    def test_always_running_substitute_on_the_same_config(self, stag):
        """What the merge actually needs to establish: the routing is
        identity-preserving on the golden's config, even when R data is absent."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = _fit_old(
                stag,
                ctor=dict(estimation_method="dr", control_group="notyettreated"),
                aggregate="simple",
            )
            new = _fit_new(
                stag,
                ctor=dict(estimation_method="dr", control_group="not_yet_treated"),
                aggregate="simple",
            )
        _assert_staggered_parity(old, new)

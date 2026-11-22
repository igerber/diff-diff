"""Replicate-weight inference df: dropped replicates cap the design df.

The design df of a replicate-weight design is the QR rank of the analysis-weight
matrix minus 1 (``None`` when rank <= 1). A replicate is dropped by an all-zero
column or by a failed refit / solve. Every estimator used to REPLACE the design
df with ``n_valid - 1`` in that case, which

- RAISED the df above the design df on a rank-deficient replicate matrix
  (15 usable columns of rank 5: t(14) instead of t(4)), and
- produced t-inference on an UNDEFINED design df (rank 1 plus a dropped column:
  t(2) instead of NaN inference).

Invariant under test: on a replicate design the df given to ``safe_inference``
is never above the design df and never a t- or z-reference when the design df
is undefined; at the influence-function sites a statistic with fewer than two
valid replicates changes no other statistic's df.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import diff_diff.survey as survey_mod
from diff_diff import (
    CallawaySantAnna,
    ContinuousDiD,
    DifferenceInDifferences,
    EfficientDiD,
    ImputationDiD,
    StackedDiD,
    SunAbraham,
    SurveyDesign,
    TripleDifference,
    TwoStageDiD,
    TwoWayFixedEffects,
)
from diff_diff.linalg import LinearRegression
from diff_diff.prep_dgp import generate_staggered_ddd_data
from diff_diff.survey import (
    cap_replicate_df,
    compute_replicate_if_variance,
    reduced_replicate_df,
)
from diff_diff.utils import safe_inference

# ---------------------------------------------------------------------------
# Data and replicate designs
# ---------------------------------------------------------------------------

DESIGN_DF = 4  # rank 5 in every defined fixture below
KINDS_DEFINED = ("healthy", "dropped_full_rank", "dropped_rank_deficient")
# df the pre-fix code reported on each fixture (n_valid - 1).
REPLACED_DF = {"dropped_rank_deficient": 14, "undefined_dropped": 2}


def _panel(seed=3, n_units=60, n_periods=6):
    """Staggered panel: cohorts {3, 5, never} interleaved across units."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        ft = 3 if u % 3 == 0 else (5 if u % 3 == 1 else 0)
        a, w = rng.normal(), rng.uniform(0.5, 2.0)
        dose = rng.uniform(0.5, 2.0) if ft else 0.0
        for t in range(1, n_periods + 1):
            y = a + 0.2 * t + (1.5 if ft and t >= ft else 0.0) + rng.normal(0, 0.5)
            rows.append((u, t, ft, y, w, dose))
    d = pd.DataFrame(rows, columns=["unit", "time", "first_treat", "outcome", "w", "dose"])
    d["treated"] = (d["first_treat"] > 0).astype(int)
    d["post"] = (d["time"] >= 3).astype(int)
    return d


def _ddd_panel():
    d = generate_staggered_ddd_data(
        n_units=240, n_periods=4, cohort_periods=[3, 4], never_enabled_frac=0.4, seed=5
    )
    units = np.sort(d["unit"].unique())
    rng = np.random.default_rng(17)
    d["w"] = d["unit"].map(pd.Series(rng.uniform(0.5, 2.0, len(units)), index=units))
    return d


def _ddd_cross_section(seed=42, n=240):
    rng = np.random.default_rng(seed)
    d1 = np.repeat([0, 1], n // 2)
    d2 = np.tile([0, 1], n // 2)
    post = rng.integers(0, 2, n)
    y = 1.0 + 0.5 * d1 + 0.3 * d2 + 2.0 * d1 * d2 * post + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "row": np.arange(n),
            "y": y,
            "d1": d1,
            "d2": d2,
            "post": post,
            "w": rng.uniform(0.5, 2.0, n),
        }
    )


def _add_reps(d, kind, key="unit", seed=9):
    """Attach replicate columns of a known rank and return ``(data, design)``.

    Base columns are the full-sample weight times a strictly positive
    ``key``-constant factor, so no replicate zeroes out a unit, a cohort or the
    comparison group: every refit is estimable and the only dropped replicate
    is the explicit all-zero column. Random positive columns are full rank.
    """
    rng = np.random.default_rng(seed)
    ids = np.sort(d[key].unique())
    w = d["w"].to_numpy()

    def base_col():
        return d[key].map(dict(zip(ids, rng.uniform(0.5, 1.5, len(ids))))).to_numpy() * w

    zero = np.zeros(len(d))
    if kind == "healthy":  # R = 5, rank 5, n_valid 5
        cols = [base_col() for _ in range(5)]
    elif kind == "dropped_full_rank":  # R = 6, rank 5, n_valid 5
        cols = [base_col() for _ in range(5)] + [zero]
    elif kind == "dropped_rank_deficient":  # R = 16, rank 5, n_valid 15
        cols = [b for b in [base_col() for _ in range(5)] for _ in range(3)] + [zero]
    elif kind == "undefined_dropped":  # R = 4, rank 1, n_valid 3
        b = base_col()
        cols = [b, 2.0 * b, 3.0 * b, zero]
    elif kind == "full12":  # R = 12, rank 12, n_valid 12
        cols = [base_col() for _ in range(12)]
    else:  # pragma: no cover
        raise ValueError(kind)
    names = [f"rw{i}" for i in range(len(cols))]
    out = pd.concat([d, pd.DataFrame(dict(zip(names, cols)), index=d.index)], axis=1)
    # mse=True centers on the full-sample estimate. The undefined fixture needs
    # it for a positive variance: its three usable columns are proportional, and
    # a refit is scale-invariant, so mean-centering would give exactly 0.
    design = SurveyDesign(
        weights="w",
        replicate_weights=names,
        replicate_method="JK1",
        mse=(kind == "undefined_dropped"),
    )
    return out, design


def test_fixture_ranks():
    """The fixtures have the df / dropped-column structure the tests rely on."""
    expected = {
        "healthy": (4, 5),
        "dropped_full_rank": (4, 6),
        "dropped_rank_deficient": (4, 16),
        "undefined_dropped": (None, 4),
        "full12": (11, 12),
    }
    for kind, (df, n_rep) in expected.items():
        d, design = _add_reps(_panel(), kind)
        resolved = design.resolve(d)
        assert resolved.df_survey == df, kind
        assert resolved.n_replicates == n_rep, kind


# ---------------------------------------------------------------------------
# Statistic extraction
# ---------------------------------------------------------------------------


def _scalar(res, label):
    return [
        (
            label,
            float(res.att),
            float(res.se),
            float(res.t_stat),
            float(res.p_value),
            float(res.conf_int[0]),
            float(res.conf_int[1]),
        )
    ]


def _rows(container, label):
    """Estimable, non-reference rows of an aggregate container.

    Reference rows carry SE 0 or NaN by construction and are not statistics.
    """
    att = np.asarray(container.att, dtype=float)
    se = np.asarray(container.se, dtype=float)
    ref = np.asarray(getattr(container, "is_reference", np.zeros(len(att), dtype=bool)))
    out = []
    for i in range(len(att)):
        if ref[i] or not np.isfinite(att[i]) or not np.isfinite(se[i]) or se[i] <= 0:
            continue
        out.append(
            (
                f"{label}[{i}]",
                att[i],
                se[i],
                float(container.t_stat[i]),
                float(container.p_value[i]),
                float(container.conf_int_lower[i]),
                float(container.conf_int_upper[i]),
            )
        )
    return out


def _row_dfs(container):
    """Per-row df provenance of the estimable, non-reference rows."""
    att = np.asarray(container.att, dtype=float)
    se = np.asarray(container.se, dtype=float)
    ref = np.asarray(getattr(container, "is_reference", np.zeros(len(att), dtype=bool)))
    keep = ~ref & np.isfinite(att) & np.isfinite(se) & (se > 0)
    return np.asarray(container.df, dtype=float)[keep]


class _Out:
    """What one adapter reports for one fit."""

    def __init__(self):
        self.stats = []  # (label, effect, se, t, p, ci_lo, ci_hi)
        self.meta_df = "absent"  # survey_metadata.df_survey, when the result has one
        self.container_dfs = []  # EventStudyResults.df_survey scalars
        self.row_dfs = []  # per-row df arrays of aggregate containers

    def add_result(self, res, label):
        self.stats += _scalar(res, label)
        self.meta_df = res.survey_metadata.df_survey
        return self

    def add_container(self, container, label):
        self.stats += _rows(container, label)
        self.row_dfs.append(_row_dfs(container))
        if hasattr(container, "df_survey"):
            self.container_dfs.append(container.df_survey)
        return self


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


# ---------------------------------------------------------------------------
# Estimator adapters: (data, design) -> _Out
# ---------------------------------------------------------------------------


def _staggered_args(d, design):
    return (d, "outcome", "unit", "time", "first_treat"), {"survey_design": design}


def _fit_cs(d, design):
    a, k = _staggered_args(d, design)
    res = CallawaySantAnna(estimation_method="reg").fit(*a, **k)
    out = _Out().add_result(res, "cs")
    out.add_container(res.aggregate("event_study"), "cs.es")
    out.add_container(res.aggregate("group"), "cs.group")
    return out


def _fit_edid(d, design):
    a, k = _staggered_args(d, design)
    res = EfficientDiD().fit(*a, **k)
    out = _Out().add_result(res, "edid")
    out.add_container(res.aggregate("event_study"), "edid.es")
    out.add_container(res.aggregate("group"), "edid.group")
    return out


def _fit_continuous(d, design):
    res = ContinuousDiD(n_bootstrap=0, num_knots=2).fit(
        d, "outcome", "unit", "time", "first_treat", "dose", survey_design=design
    )
    out = _Out().add_result(res, "cont.att")
    out.stats.append(
        (
            "cont.acrt",
            float(res.overall_acrt),
            float(res.overall_acrt_se),
            float(res.overall_acrt_t_stat),
            float(res.overall_acrt_p_value),
            float(res.overall_acrt_conf_int[0]),
            float(res.overall_acrt_conf_int[1]),
        )
    )
    out.add_container(res.aggregate("event_study"), "cont.es")
    return out


def _fit_did_plain(d, design):
    res = DifferenceInDifferences().fit(
        d, outcome="outcome", treatment="treated", post="post", survey_design=design
    )
    return _Out().add_result(res, "did")


def _fit_did_absorb(d, design):
    res = DifferenceInDifferences().fit(
        d,
        outcome="outcome",
        treatment="treated",
        post="post",
        absorb=["unit"],
        survey_design=design,
    )
    return _Out().add_result(res, "did.absorb")


def _fit_twfe(d, design):
    res = TwoWayFixedEffects().fit(
        d, outcome="outcome", treatment="treated", post="post", unit="unit", survey_design=design
    )
    return _Out().add_result(res, "twfe")


def _fit_twfe_es(spec):
    def fit(d, design):
        kw = {"unit": "unit"} if spec == "within" else {}
        es = TwoWayFixedEffects().fit(
            d,
            outcome="outcome",
            treatment="treated",
            time="time",
            event_study=True,
            spec=spec,
            post_periods=[3, 4, 5, 6],
            survey_design=design,
            **kw,
        )
        # EventStudyResults: no survey_metadata, scalar df_survey instead.
        return _Out().add_container(es, f"twfe.es.{spec}")

    return fit


def _fit_sun_abraham(d, design):
    a, k = _staggered_args(d, design)
    res = SunAbraham(n_bootstrap=0).fit(*a, **k)
    out = _Out().add_result(res, "sa")
    for e, row in sorted(res.event_study_effects.items()):
        se = row["se"]
        if np.isfinite(row["effect"]) and np.isfinite(se) and se > 0:
            out.stats.append(
                (
                    f"sa.es[{e}]",
                    row["effect"],
                    se,
                    row["t_stat"],
                    row["p_value"],
                    row["conf_int"][0],
                    row["conf_int"][1],
                )
            )
    return out


def _fit_keyword_staggered(cls, label, levels, **est_kw):
    def fit(d, design):
        res = cls(**est_kw).fit(
            d,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=design,
        )
        out = _Out().add_result(res, label)
        for level in levels:
            out.add_container(res.aggregate(level), f"{label}.{level}")
        return out

    return fit


def _fit_ddd_staggered(d, design):
    res = TripleDifference().fit(
        d,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        partition="eligibility",
        survey_design=design,
    )
    out = _Out().add_result(res, "ddd.staggered")
    out.add_container(res.aggregate("event_study"), "ddd.staggered.es")
    return out


def _fit_ddd_2x2x2(d, design):
    res = TripleDifference(estimation_method="reg").fit(
        d, outcome="y", group="d1", partition="d2", post="post", survey_design=design
    )
    return _Out().add_result(res, "ddd.2x2x2")


# name -> (data builder, replicate key, adapter)
ADAPTERS = {
    "cs": (_panel, "unit", _fit_cs),
    "efficient_did": (_panel, "unit", _fit_edid),
    "continuous_did": (_panel, "unit", _fit_continuous),
    "did_plain": (_panel, "unit", _fit_did_plain),
    "did_absorb": (_panel, "unit", _fit_did_absorb),
    "twfe": (_panel, "unit", _fit_twfe),
    "twfe_es_within": (_panel, "unit", _fit_twfe_es("within")),
    "twfe_es_pooled": (_panel, "unit", _fit_twfe_es("pooled")),
    "sun_abraham": (_panel, "unit", _fit_sun_abraham),
    "stacked_did": (_panel, "unit", _fit_keyword_staggered(StackedDiD, "stacked", ["event_study"])),
    "imputation_did": (
        _panel,
        "unit",
        _fit_keyword_staggered(ImputationDiD, "imp", ["event_study", "group"], n_bootstrap=0),
    ),
    "two_stage_did": (
        _panel,
        "unit",
        _fit_keyword_staggered(TwoStageDiD, "ts", ["event_study", "group"], n_bootstrap=0),
    ),
    "ddd_staggered": (_ddd_panel, "unit", _fit_ddd_staggered),
    "ddd_2x2x2": (_ddd_cross_section, "row", _fit_ddd_2x2x2),
}

# Which replicate-variance routine prices each adapter's statistics.
IF_PATH = ("cs", "efficient_did", "continuous_did", "ddd_staggered", "ddd_2x2x2")
REFIT_PATH = (
    "did_absorb",
    "twfe",
    "twfe_es_within",
    "sun_abraham",
    "stacked_did",
    "imputation_did",
    "two_stage_did",
)
OLS_PATH = ("did_plain", "twfe_es_pooled")

_FIT_CACHE = {}


def _run(name, kind):
    """Fit ``name`` on fixture ``kind`` (cached: each fit is reused across tests)."""
    if (name, kind) not in _FIT_CACHE:
        build, key, adapter = ADAPTERS[name]
        d, design = _add_reps(build(), kind, key=key)
        _FIT_CACHE[(name, kind)] = _quiet(lambda: adapter(d, design))
    return _FIT_CACHE[(name, kind)]


def _assert_inference_at_df(out, df):
    assert out.stats, "adapter returned no estimable statistics"
    for label, effect, se, t, p, lo, hi in out.stats:
        exp_t, exp_p, exp_ci = safe_inference(effect, se, alpha=0.05, df=df)
        np.testing.assert_allclose(t, exp_t, rtol=1e-10, err_msg=label)
        np.testing.assert_allclose(p, exp_p, rtol=1e-8, atol=1e-300, err_msg=label)
        np.testing.assert_allclose([lo, hi], exp_ci, rtol=1e-10, err_msg=label)


def _assert_df_carriers(out, df):
    if out.meta_df != "absent":
        assert out.meta_df == df
    for value in out.container_dfs:
        assert value == float(df)
    for arr in out.row_dfs:
        assert len(arr) > 0
        np.testing.assert_array_equal(arr, float(df))


# ---------------------------------------------------------------------------
# 1. Estimator matrix on real designs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ADAPTERS))
class TestEstimatorMatrix:
    @pytest.mark.parametrize("kind", KINDS_DEFINED)
    def test_inference_uses_design_df(self, name, kind):
        out = _run(name, kind)
        _assert_inference_at_df(out, DESIGN_DF)
        _assert_df_carriers(out, DESIGN_DF)

    def test_rank_deficient_df_is_not_raised(self, name):
        """Guards the matrix against a vacuous pass: t(4) and t(14) differ."""
        out = _run(name, "dropped_rank_deficient")
        for label, effect, se, _t, p, _lo, _hi in out.stats:
            _, p_replaced, _ = safe_inference(
                effect, se, alpha=0.05, df=REPLACED_DF["dropped_rank_deficient"]
            )
            assert abs(p - p_replaced) > 1e-12 * max(p, p_replaced), label
            assert p > p_replaced, label  # t(4) is the heavier tail

    def test_undefined_design_df_gives_nan_inference(self, name):
        out = _run(name, "undefined_dropped")
        assert out.stats
        for label, _effect, se, t, p, lo, hi in out.stats:
            # Non-vacuity: the SE is valid, so NaN inference comes from the df.
            assert np.isfinite(se) and se > 0, label
            assert np.isnan(t) and np.isnan(p) and np.isnan(lo) and np.isnan(hi), label
        if out.meta_df != "absent":
            assert out.meta_df is None  # never 0, never n_valid - 1
        for value in out.container_dfs:
            assert value == 0.0  # the replicate-undefined sentinel
        for arr in out.row_dfs:
            assert np.all(np.isnan(arr))


# ---------------------------------------------------------------------------
# 2. A genuine reduction below the design df is kept
# ---------------------------------------------------------------------------


def _report_failed(monkeypatch, fn_name, n_failed=3):
    """Make one replicate-variance routine report ``n_failed`` dropped replicates."""
    original = getattr(survey_mod, fn_name)

    def wrapped(*args, **kwargs):
        value, n_valid = original(*args, **kwargs)
        return value, n_valid - n_failed

    monkeypatch.setattr(survey_mod, fn_name, wrapped)


class TestCapBindsBelowDesignDf:
    """12 full-rank columns (design df 11), 3 reported failed -> df 8."""

    @staticmethod
    def _fit(name, monkeypatch, fn_name):
        build, key, adapter = ADAPTERS[name]
        d, design = _add_reps(build(), "full12", key=key)
        assert design.resolve(d).df_survey == 11
        _report_failed(monkeypatch, fn_name)
        return _quiet(lambda: adapter(d, design))

    @pytest.mark.parametrize("name", REFIT_PATH)
    def test_refit_path(self, name, monkeypatch):
        out = self._fit(name, monkeypatch, "compute_replicate_refit_variance")
        _assert_inference_at_df(out, 8)
        _assert_df_carriers(out, 8)

    @pytest.mark.parametrize("name", OLS_PATH)
    def test_ols_path(self, name, monkeypatch):
        # did_plain pins the metadata write: inference comes from
        # LinearRegression.survey_df_, and metadata must report the same df.
        out = self._fit(name, monkeypatch, "compute_replicate_vcov")
        _assert_inference_at_df(out, 8)
        _assert_df_carriers(out, 8)

    @pytest.mark.parametrize("name", IF_PATH)
    def test_if_path(self, name, monkeypatch):
        out = self._fit(name, monkeypatch, "compute_replicate_if_variance")
        _assert_inference_at_df(out, 8)
        _assert_df_carriers(out, 8)


# ---------------------------------------------------------------------------
# 3. LinearRegression direct
# ---------------------------------------------------------------------------


class TestLinearRegressionDirect:
    @staticmethod
    def _fit(kind):
        d, design = _add_reps(_panel(), kind)
        resolved = design.resolve(d)
        X = np.column_stack(
            [np.ones(len(d)), d["treated"], d["post"], d["treated"] * d["post"]]
        ).astype(float)
        reg = LinearRegression(
            include_intercept=False,
            weights=resolved.weights,
            weight_type="pweight",
            survey_design=resolved,
        )
        _quiet(lambda: reg.fit(X, d["outcome"].to_numpy()))
        return reg

    @pytest.mark.parametrize("kind", KINDS_DEFINED)
    def test_survey_df_is_design_df(self, kind):
        reg = self._fit(kind)
        assert reg.survey_df_ == DESIGN_DF
        inf = reg.get_inference(3)
        _, exp_p, exp_ci = safe_inference(inf.coefficient, inf.se, alpha=0.05, df=DESIGN_DF)
        np.testing.assert_allclose(inf.p_value, exp_p, rtol=1e-8)
        np.testing.assert_allclose(inf.conf_int, exp_ci, rtol=1e-10)

    def test_undefined_design_df_warns_and_nans(self):
        reg = self._fit("undefined_dropped")
        # None, not 0 and not n_valid - 1: same state as an undefined design
        # with no dropped replicate, so the same warning + NaN branch fires.
        assert reg.survey_df_ is None
        with pytest.warns(UserWarning, match="undefined survey d.f."):
            inf = reg.get_inference(3)
        assert np.isfinite(inf.se) and inf.se > 0
        assert np.isnan(inf.t_stat) and np.isnan(inf.p_value)
        assert np.isnan(inf.conf_int[0]) and np.isnan(inf.conf_int[1])

    def test_failed_solves_lower_the_df(self, monkeypatch):
        d, design = _add_reps(_panel(), "full12")
        resolved = design.resolve(d)
        _report_failed(monkeypatch, "compute_replicate_vcov")
        X = np.column_stack([np.ones(len(d)), d["treated"], d["post"]]).astype(float)
        reg = LinearRegression(
            include_intercept=False,
            weights=resolved.weights,
            weight_type="pweight",
            survey_design=resolved,
        )
        _quiet(lambda: reg.fit(X, d["outcome"].to_numpy()))
        assert reg.survey_df_ == 8


# ---------------------------------------------------------------------------
# 4. Per-statistic isolation (influence-function sites)
# ---------------------------------------------------------------------------


def _spy_one_call(monkeypatch, call_index):
    """Make exactly one ``compute_replicate_if_variance`` call fully degenerate."""
    original = survey_mod.compute_replicate_if_variance
    state = {"n": 0}

    def spy(psi, resolved):
        i = state["n"]
        state["n"] += 1
        if i == call_index:
            return float("nan"), 0
        return original(psi, resolved)

    monkeypatch.setattr(survey_mod, "compute_replicate_if_variance", spy)
    return state


def _assert_only_one_statistic_changed(clean, spied):
    """Every statistic but one is identical; the odd one out has NaN inference."""
    assert len(clean) == len(spied)
    changed = 0
    for c, s in zip(clean, spied):
        if np.isnan(s[2]):  # se
            changed += 1
            assert np.isnan(s[3]) and np.isnan(s[4]) and np.isnan(s[5]) and np.isnan(s[6])
            continue
        np.testing.assert_allclose(s[1:], c[1:], rtol=1e-12, err_msg=c[0])
    assert changed == 1


def _all_rows(container, label):
    """Every non-reference row, including ones whose SE went NaN."""
    att = np.asarray(container.att, dtype=float)
    ref = np.asarray(getattr(container, "is_reference", np.zeros(len(att), dtype=bool)))
    return [
        (
            f"{label}[{i}]",
            att[i],
            float(container.se[i]),
            float(container.t_stat[i]),
            float(container.p_value[i]),
            float(container.conf_int_lower[i]),
            float(container.conf_int_upper[i]),
        )
        for i in range(len(att))
        if not ref[i] and np.isfinite(att[i])
    ]


class TestPerStatisticIsolation:
    """A statistic with < 2 valid replicates must not change another's df.

    Each arm marked "discriminating" fails on the pre-fix tree.
    """

    def test_efficient_did_per_cell(self, monkeypatch):
        # Discriminating: the pre-fix sticky df became None, so every later
        # cell (and the overall ATT) silently switched to a z-reference.
        d, design = _add_reps(_panel(), "healthy")
        a, k = _staggered_args(d, design)

        def cells(res):
            out = []
            for gt, row in sorted(res.group_time_effects.items()):
                out.append(
                    (
                        str(gt),
                        row["effect"],
                        row["se"],
                        row["t_stat"],
                        row["p_value"],
                        row["conf_int"][0],
                        row["conf_int"][1],
                    )
                )
            return out

        clean = _quiet(lambda: EfficientDiD().fit(*a, **k))
        _spy_one_call(monkeypatch, 0)
        spied = _quiet(lambda: EfficientDiD().fit(*a, **k))
        _assert_only_one_statistic_changed(cells(clean), cells(spied))
        np.testing.assert_allclose(spied.p_value, clean.p_value, rtol=1e-12)
        np.testing.assert_allclose(spied.conf_int, clean.conf_int, rtol=1e-12)
        assert spied.survey_metadata.df_survey == DESIGN_DF

    def test_efficient_did_wif_aggregate(self, monkeypatch):
        # Regression guard only: the WIF sites discarded n_valid before the
        # fix as well, so this arm passes on both trees.
        d, design = _add_reps(_panel(), "healthy")
        a, k = _staggered_args(d, design)
        res = _quiet(lambda: EfficientDiD().fit(*a, **k))
        clean = _all_rows(_quiet(lambda: res.aggregate("event_study")), "es")
        _spy_one_call(monkeypatch, 0)
        spied = _all_rows(_quiet(lambda: res.aggregate("event_study")), "es")
        _assert_only_one_statistic_changed(clean, spied)

    def test_continuous_did(self, monkeypatch):
        # Discriminating: the pre-fix worst-case count fell to 0, which turned
        # EVERY statistic's inference NaN.
        d, design = _add_reps(_panel(), "healthy")

        def fit():
            return _quiet(
                lambda: ContinuousDiD(n_bootstrap=0, num_knots=2).fit(
                    d, "outcome", "unit", "time", "first_treat", "dose", survey_design=design
                )
            )

        clean = fit()
        _spy_one_call(monkeypatch, 1)  # the overall ACRT statistic
        spied = fit()
        assert np.isnan(spied.overall_acrt_se) and np.isnan(spied.overall_acrt_p_value)
        np.testing.assert_allclose(spied.se, clean.se, rtol=1e-12)
        np.testing.assert_allclose(spied.p_value, clean.p_value, rtol=1e-12)
        np.testing.assert_allclose(spied.conf_int, clean.conf_int, rtol=1e-12)
        assert spied.survey_metadata.df_survey == DESIGN_DF

    @pytest.mark.parametrize("level", ["event_study", "group"])
    def test_callaway_santanna_aggregates(self, level, monkeypatch):
        # Discriminating: the shared _se_from_psi returned the 0 sentinel for
        # the degenerate statistic, and the relay's min() over rows then gave
        # EVERY horizon / group NaN inference.
        d, design = _add_reps(_panel(), "healthy")
        a, k = _staggered_args(d, design)
        res = _quiet(lambda: CallawaySantAnna(estimation_method="reg").fit(*a, **k))
        clean = _all_rows(_quiet(lambda: res.aggregate(level)), level)
        _spy_one_call(monkeypatch, 0)
        spied = _all_rows(_quiet(lambda: res.aggregate(level)), level)
        _assert_only_one_statistic_changed(clean, spied)

    def test_callaway_santanna_overall_keeps_metadata(self, monkeypatch):
        # Discriminating: a degenerate overall statistic used to write the 0
        # sentinel into survey_metadata.df_survey.
        d, design = _add_reps(_panel(), "healthy")
        a, k = _staggered_args(d, design)
        original = survey_mod.compute_replicate_if_variance

        def degenerate(psi, resolved):
            _var, _n = original(psi, resolved)
            return float("nan"), 0

        monkeypatch.setattr(survey_mod, "compute_replicate_if_variance", degenerate)
        res = _quiet(lambda: CallawaySantAnna(estimation_method="reg").fit(*a, **k))
        assert np.isnan(res.se) and np.isnan(res.p_value)
        assert res.survey_metadata.df_survey == DESIGN_DF


class TestNoRepeatedDesignRank:
    """Post-fit aggregation must not recompute the design rank per statistic.

    ``ResolvedSurveyDesign.df_survey`` runs a pivoted QR on the n x R
    replicate matrix on every access, and an all-zero replicate column makes
    ``n_valid < R`` for every horizon and group.
    """

    @staticmethod
    def _count_df_survey(monkeypatch):
        calls = {"n": 0}
        original = survey_mod.ResolvedSurveyDesign.df_survey

        def counted(self):
            calls["n"] += 1
            return original.fget(self)

        monkeypatch.setattr(survey_mod.ResolvedSurveyDesign, "df_survey", property(counted))
        return calls

    @pytest.mark.parametrize("level", ["event_study", "group"])
    def test_callaway_santanna(self, level, monkeypatch):
        d, design = _add_reps(_panel(), "dropped_rank_deficient")
        a, k = _staggered_args(d, design)
        res = _quiet(lambda: CallawaySantAnna(estimation_method="reg").fit(*a, **k))
        calls = self._count_df_survey(monkeypatch)
        agg = _quiet(lambda: res.aggregate(level))
        assert calls["n"] == 0
        np.testing.assert_array_equal(_row_dfs(agg), float(DESIGN_DF))

    @pytest.mark.parametrize("level", ["event_study", "group"])
    def test_efficient_did(self, level, monkeypatch):
        d, design = _add_reps(_panel(), "dropped_rank_deficient")
        a, k = _staggered_args(d, design)
        res = _quiet(lambda: EfficientDiD().fit(*a, **k))
        calls = self._count_df_survey(monkeypatch)
        agg = _quiet(lambda: res.aggregate(level))
        assert calls["n"] == 0
        np.testing.assert_array_equal(_row_dfs(agg), float(DESIGN_DF))


# ---------------------------------------------------------------------------
# 5. EfficientDiD kit df on an undefined design
# ---------------------------------------------------------------------------


class TestEfficientDiDKit:
    def test_undefined_design_kit_df_is_sentinel(self):
        d, design = _add_reps(_panel(), "undefined_dropped")
        a, k = _staggered_args(d, design)
        res = _quiet(lambda: EfficientDiD().fit(*a, **k))
        # The working df stays the seeded 0 sentinel; it is never None (a
        # z-reference) and never n_valid - 1.
        assert res._aggregation_kit.bookkeeping["df_survey"] == 0
        assert res.survey_metadata.df_survey is None
        es = _quiet(lambda: res.aggregate("event_study"))
        assert es.df_survey == 0.0
        assert np.all(np.isnan(np.asarray(es.p_value, dtype=float)))
        grp = _quiet(lambda: res.aggregate("group"))
        assert np.all(np.isnan(np.asarray(grp.df, dtype=float)))
        assert np.all(np.isnan(np.asarray(grp.p_value, dtype=float)))


# ---------------------------------------------------------------------------
# 6. Influence-function path: the design df never exceeds n_valid - 1
# ---------------------------------------------------------------------------


class TestNeverBindsProperty:
    """QR rank <= number of nonzero replicate columns == n_valid (finite psi).

    This is why the influence-function sites that do not consume ``n_valid``
    cannot report a df above what a cap would give.
    """

    @pytest.mark.parametrize("combined", [True, False])
    @pytest.mark.parametrize("seed", range(12))
    def test_design_df_at_most_n_valid_minus_one(self, seed, combined):
        rng = np.random.default_rng(seed)
        n = 40
        n_base = int(rng.integers(1, 6))
        base = rng.uniform(0.2, 2.0, (n, n_base))
        picks = rng.integers(0, n_base, int(rng.integers(2, 10)))
        cols = [base[:, j] * rng.uniform(0.5, 2.0) for j in picks]
        cols += [np.zeros(n) for _ in range(int(rng.integers(0, 3)))]
        order = rng.permutation(len(cols))
        names = [f"r{i}" for i in range(len(cols))]
        d = pd.DataFrame({"w": rng.uniform(0.5, 2.0, n)})
        for name, j in zip(names, order):
            d[name] = cols[j]
        resolved = SurveyDesign(
            weights="w",
            replicate_weights=names,
            replicate_method="JK1",
            combined_weights=combined,
        ).resolve(d)
        _var, n_valid = compute_replicate_if_variance(rng.normal(size=n), resolved)
        assert n_valid == sum(1 for c in cols if np.any(c > 0))
        df = resolved.df_survey
        assert df is None or df <= n_valid - 1
        if df is not None and n_valid >= 2:
            assert reduced_replicate_df(df, n_valid) is None
            assert cap_replicate_df(df, n_valid) == df


# ---------------------------------------------------------------------------
# 7. Helper tables
# ---------------------------------------------------------------------------


class TestReducedReplicateDf:
    @pytest.mark.parametrize(
        "base_df, n_valid, expected",
        [
            (None, 7, None),  # undefined design df stays undefined
            (0, 7, None),  # already the sentinel
            (19, 0, None),  # no variance for this statistic
            (19, 1, None),
            (19, 20, None),  # nothing dropped
            (19, 21, None),  # never raised
            (4, 15, None),  # rank-deficient: design df is the ceiling
            (19, 8, 7),  # strict reduction
            (4, 5, None),  # equal: no override
            (4, 4, 3),
            (11.0, 9, 8),  # float design df
        ],
    )
    def test_table(self, base_df, n_valid, expected):
        got = reduced_replicate_df(base_df, n_valid)
        assert got == expected
        assert expected is None or isinstance(got, int)

    @pytest.mark.parametrize("base_df", [None, 0, 1, 4, 19])
    @pytest.mark.parametrize("n_valid", [0, 1, 2, 5, 20, 40])
    def test_consistent_with_cap(self, base_df, n_valid):
        """An override, when given, is exactly the cap; and it only lowers."""
        reduced = reduced_replicate_df(base_df, n_valid)
        if reduced is not None:
            assert reduced == cap_replicate_df(base_df, n_valid)
            assert 0 < reduced < base_df

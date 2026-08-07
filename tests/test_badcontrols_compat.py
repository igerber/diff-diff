import numpy as np
import pandas as pd
import pytest

from diff_diff import didbc, dr_ml_attgt, extract_att, two_by_two_subset


def _bad_control_panel():
    rows = []
    for unit in range(20):
        treated = unit >= 10
        x_pre = (unit % 5) / 5.0
        for period in (0, 1):
            noise = 0.01 * ((unit * 7) % 5) if period == 1 else 0.0
            x = x_pre + (1.5 if treated else 0.5) * period + noise
            y = 2.0 * x + (3.0 if treated and period == 1 else 0.0)
            rows.append({"id": unit, "period": period, "G": 1 if treated else 0, "Y": y, "X": x})
    return pd.DataFrame(rows)


def test_imputation_recovers_effect_through_a_bad_control():
    result = didbc(
        _bad_control_panel(),
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
    )
    assert np.isclose(result.att, 5.0)
    assert np.isfinite(result.se)
    assert extract_att(result) == {"att": result.att, "se": result.se}


def test_unknown_dr_nuisance_method_fails_closed():
    try:
        didbc(
            _bad_control_panel(),
            yname="Y",
            gname="G",
            tname="period",
            idname="id",
            bad_control="X",
            est_method="dr_ml",
            nuisance_method="unknown",
        )
    except NotImplementedError as exc:
        assert "dr_ml" in str(exc)
    else:
        raise AssertionError("unknown nuisance methods must fail closed")


def test_parametric_dr_returns_finite_att_and_influence_function():
    result = didbc(
        _bad_control_panel(),
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="parametric",
    )
    assert result.method == "dr_ml-parametric"
    assert np.isfinite(result.att)
    assert np.isfinite(result.se)
    assert np.isclose(result.influence_function.mean(), 0.0)


def test_dr_ml_attgt_accepts_r_style_gt_data():
    gt_data = _bad_control_panel().copy()
    gt_data["name"] = np.where(gt_data["period"].eq(0), "pre", "post")
    gt_data["D"] = (gt_data["G"] != 0).astype(int)
    result = dr_ml_attgt(
        gt_data,
        xformula="~1",
        bad_control_formula="~X",
        d_covs_formula="~-1",
        nuisance_method="parametric",
    )

    assert result.method == "dr_ml-parametric"
    assert np.isfinite(result.att)


def test_random_forest_dr_cross_fits_and_returns_finite_result():
    result = didbc(
        _bad_control_panel(),
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="ml",
        n_folds=2,
    )
    assert result.method == "dr_ml"
    assert np.isfinite(result.att)
    assert np.isfinite(result.se)


def test_dr_small_treated_group_falls_back_to_imputation():
    panel = _bad_control_panel().query("id < 14").copy()
    panel["G"] = (panel["id"] >= 10).astype(int)
    panel.loc[panel["G"].eq(1), "G"] = 1
    result = didbc(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="parametric",
        min_group_size=5,
    )
    assert result.method == "imputation"


def test_badcontrols_bootstrap_is_reproducible():
    kwargs = {
        "yname": "Y",
        "gname": "G",
        "tname": "period",
        "idname": "id",
        "bad_control": "X",
        "bstrap": True,
        "biters": 7,
        "seed": 12,
    }
    first = didbc(_bad_control_panel(), **kwargs)
    second = didbc(_bad_control_panel(), **kwargs)
    assert first.bootstrap_distribution is not None
    assert np.allclose(first.bootstrap_distribution, second.bootstrap_distribution)
    assert np.isfinite(first.se)
    assert first.conf_int[0] <= first.conf_int[1]


def test_imputation_accepts_covariate_changes():
    panel = _bad_control_panel()
    panel["Z"] = panel["id"] / 10 * panel["period"]
    result = didbc(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        d_covariates=["Z"],
    )
    assert np.isfinite(result.att)


def test_parametric_dr_accepts_covariate_changes():
    panel = _bad_control_panel()
    panel["Z"] = panel["id"] / 10 * panel["period"]
    result = didbc(
        panel,
        yname="Y",
        gname="G",
        tname="period",
        idname="id",
        bad_control="X",
        est_method="dr_ml",
        nuisance_method="parametric",
        d_covariates=["Z"],
    )
    assert np.isfinite(result.att)


def _well_overlapped_panel(n=120, seed=7):
    """A two-period panel whose propensity overlaps cleanly, so the DR score
    (rather than an overlap-induced imputation fallback) is actually computed."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        treated = unit >= n // 2
        x_pre = rng.normal(0.0, 1.0)
        d0 = rng.normal(0.0, 1.0)
        for period in (1, 2):
            noise = rng.normal(0.0, 0.3)
            x = x_pre * 0.8 + d0 + noise
            y = 1.2 * x + (0.7 if treated and period == 2 else 0.0) + rng.normal(0.0, 0.5)
            rows.append(
                {
                    "id": unit,
                    "period": period,
                    "G": 2 if treated else 0,
                    "Y": round(y, 6),
                    "X": round(x, 6),
                }
            )
    return pd.DataFrame(rows)


def test_parametric_dr_cross_fits_and_is_fold_dependent():
    """The parametric DR path must cross-fit like R ``dr_ml_attgt``: different
    fold assignments yield different finite-sample ATT (R returns a different
    value per ``set.seed``). A full-sample fit would be fold-invariant."""
    gt = _well_overlapped_panel()
    gt["name"] = np.where(gt["period"].eq(1), "pre", "post")
    gt["D"] = (gt["G"] != 0).astype(int)
    values = {
        dr_ml_attgt(
            gt,
            xformula="~1",
            bad_control_formula="~X",
            nuisance_method="parametric",
            n_folds=3,
            random_state=s,
        ).att
        for s in (0, 1, 2)
    }
    assert len(values) > 1, "parametric DR must be fold-dependent (cross-fitted)"


def test_parametric_dr_fold_ids_ingress_is_reproducible():
    gt = _well_overlapped_panel()
    gt["name"] = np.where(gt["period"].eq(1), "pre", "post")
    gt["D"] = (gt["G"] != 0).astype(int)
    n_units = gt["id"].nunique()
    fold_ids = (np.arange(n_units) % 5).astype(int)
    first = dr_ml_attgt(
        gt,
        xformula="~1",
        bad_control_formula="~X",
        nuisance_method="parametric",
        n_folds=5,
        fold_ids=fold_ids,
    )
    second = dr_ml_attgt(
        gt,
        xformula="~1",
        bad_control_formula="~X",
        nuisance_method="parametric",
        n_folds=5,
        fold_ids=fold_ids,
    )
    assert np.isclose(first.att, second.att)
    with pytest.raises(ValueError):
        dr_ml_attgt(
            gt,
            xformula="~1",
            bad_control_formula="~X",
            nuisance_method="parametric",
            n_folds=5,
            fold_ids=np.full(n_units, 5, dtype=int),
        )


def _separated_panel(n=40):
    """X perfectly separates treated from control, so a preliminary propensity
    fit exceeds 0.99 and R/Python fall back to imputation."""
    rows = []
    for unit in range(n):
        treated = unit >= n // 2
        x_pre = unit / 10.0
        for period in (1, 2):
            noise = 0.01 * ((unit * 7) % 5) if period == 2 else 0.0
            x = x_pre + (1.5 if treated else 0.5) * (period - 1) + noise
            y = 2.0 * x + (3.0 if treated and period == 2 else 0.0)
            rows.append({"id": unit, "period": period, "G": 2 if treated else 0, "Y": y, "X": x})
    return pd.DataFrame(rows)


def test_dr_ml_attgt_overlap_falls_back_to_imputation():
    """Mirror R's ``dr_ml_attgt`` overlap guard: a preliminary-fit propensity
    above 0.99 falls back to the imputation estimator for the whole cell."""
    gt = _separated_panel(20)
    gt["name"] = np.where(gt["period"].eq(1), "pre", "post")
    gt["D"] = (gt["G"] != 0).astype(int)
    result = dr_ml_attgt(
        gt,
        xformula="~1",
        bad_control_formula="~X",
        nuisance_method="parametric",
        n_folds=3,
    )
    assert result.method.startswith("imputation")


def test_dr_ml_attgt_accepts_two_by_two_subset_gt_data():
    panel = _bad_control_panel()
    panel["G"] = panel["G"].replace({1: 2})
    subset = two_by_two_subset(
        panel, 2, 1, gname="G", tname="period", idname="id", covariates=["X"]
    )
    result = dr_ml_attgt(
        subset.gt_data,
        xformula="~1",
        bad_control_formula="~X",
        nuisance_method="parametric",
        n_folds=3,
    )
    assert np.isfinite(result.att)

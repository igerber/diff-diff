import numpy as np
import pandas as pd
import pytest

from diff_diff import DIDOVBSensitivity


@pytest.fixture
def panel():
    rows = []
    rng = np.random.default_rng(123)
    for unit in range(80):
        treated = int(unit < 40)
        x = (unit - 39.5) / 40.0
        baseline = 0.4 * x + rng.normal(scale=0.1)
        trend = 0.25 * x + rng.normal(scale=0.08)
        effect = 0.8 if treated else 0.0
        rows.extend(
            [
                {"unit": unit, "period": 0, "treated": treated, "x": x, "y": baseline},
                {
                    "unit": unit,
                    "period": 1,
                    "treated": treated,
                    "x": x,
                    "y": baseline + trend + effect,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_fit_accepts_balanced_two_period_panel_and_reports_short_att(panel):
    result = DIDOVBSensitivity(n_folds=2, seed=7).fit(
        panel,
        outcome="y",
        treatment="treated",
        time="period",
        unit="unit",
        covariates=["x"],
    )

    assert np.isfinite(result.short_att)
    assert result.n_obs == len(panel) // 2
    assert result.short_se > 0


def test_fit_rejects_more_than_two_periods(panel):
    three_periods = pd.concat(
        [panel, panel.assign(period=2, y=panel["y"] + 0.1)], ignore_index=True
    )
    with pytest.raises(ValueError, match="exactly two periods"):
        DIDOVBSensitivity().fit(
            three_periods,
            outcome="y",
            treatment="treated",
            time="period",
            unit="unit",
            covariates=["x"],
        )


def test_components_obey_scale_identity(panel):
    result = DIDOVBSensitivity(n_folds=2, seed=11).fit(
        panel,
        outcome="y",
        treatment="treated",
        time="period",
        unit="unit",
        covariates=["x"],
    )

    assert result.scale == pytest.approx(np.sqrt(result.sigma2_control * result.nu2_selection))
    assert result.sigma2_control > 0
    assert result.nu2_selection >= 1


def test_bounds_are_symmetric_and_expand_with_strength(panel):
    result = DIDOVBSensitivity(n_folds=2, seed=7).fit(
        panel,
        outcome="y",
        treatment="treated",
        time="period",
        unit="unit",
        covariates=["x"],
    )
    narrow = result.bounds(trend_r2=0.2, selection_r2=0.2)
    wide = result.bounds(trend_r2=0.5, selection_r2=0.5)

    assert narrow.lower < result.short_att < narrow.upper
    assert narrow.lower == pytest.approx(result.short_att - narrow.radius)
    assert narrow.upper == pytest.approx(result.short_att + narrow.radius)
    assert wide.radius > narrow.radius


def test_null_inside_original_interval_has_zero_rv_and_xrv(panel):
    result = DIDOVBSensitivity(n_folds=2, seed=7).fit(
        panel,
        outcome="y",
        treatment="treated",
        time="period",
        unit="unit",
        covariates=["x"],
    )
    robustness = result.robustness_value(null_value=result.short_att, alpha=0.05)

    assert robustness.rv == 0.0
    assert robustness.xrv == 0.0


def test_result_exports_and_contour_are_structured(panel):
    result = DIDOVBSensitivity(n_folds=2, seed=7).fit(
        panel,
        outcome="y",
        treatment="treated",
        time="period",
        unit="unit",
        covariates=["x"],
    )

    contour = result.contour(null_value=0.0, n_grid=11)
    assert list(contour.columns) == [
        "strength",
        "rv_lower",
        "rv_upper",
        "xrv_lower",
        "xrv_upper",
    ]
    assert len(contour) == 11
    assert result.to_dict()["short_att"] == pytest.approx(result.short_att)
    assert "OVB Sensitivity" in result.summary()

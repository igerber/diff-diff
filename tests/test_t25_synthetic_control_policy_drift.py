"""Drift detection for Tutorial 25 (``docs/tutorials/25_synthetic_control_policy.ipynb``).

The tutorial narrative quotes seed-specific numbers (ATT, pre-RMSPE, donor weights,
placebo p-value, leave-one-out sensitivity, conformal p-values, the pointwise CI band, and
the average-effect interval). ``pytest --nbmake`` only checks the cells *execute*; it does
not check the prose. These asserts re-derive the same numbers from the library generator
``generate_synthetic_control_data`` with the locked kwargs and check them against the values
quoted in the markdown. If library numerics drift, this test fails and a maintainer must
update the prose or investigate.

Unlike the inline-DGP tutorials (e.g. T23), T25 builds its panel from the public generator,
so there is no DGP code to duplicate — the drift test calls the same generator. A
``test_notebook_kwargs_match`` sync-guard pins the generator/fit kwargs so a notebook-only
edit can't silently change the behavior the numbers describe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from diff_diff import SyntheticControl, generate_synthetic_control_data

# Locked config — must stay in sync with the notebook §1/§2 cells.
SEED = 0
N_DONORS = 20
N_PRE = 60
N_POST = 5
TREATMENT_EFFECT = 5.0
EFFECT_GROWTH = 1.0
NOISE_SD = 0.6
PREDICTORS = ["x1", "x2", "x3"]
IN_TIME_BACKDATES = [20, 35, 50]
TRUE_RAMP = np.array([5.0, 6.0, 7.0, 8.0, 9.0])


def _silence_matmul_warnings() -> None:
    """Mirror the notebook's narrow Apple-Silicon BLAS RuntimeWarning filter
    (numpy#28687); a no-op on other platforms. Never silences UserWarnings."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul")


@pytest.fixture(scope="module")
def fit():
    """Build the panel and run the full surface once (matches the notebook order)."""
    panel = generate_synthetic_control_data(
        n_donors=N_DONORS,
        n_pre=N_PRE,
        n_post=N_POST,
        n_factors=3,
        n_predictors=3,
        treatment_effect=TREATMENT_EFFECT,
        effect_type="ramp",
        effect_growth=EFFECT_GROWTH,
        noise_sd=NOISE_SD,
        seed=SEED,
    )
    with warnings.catch_warnings():
        _silence_matmul_warnings()
        res = SyntheticControl(
            v_method="nested",
            n_starts=1,
            inner_min_decrease=1e-3,
            optimizer_options={"maxiter": 50},
            seed=SEED,
        ).fit(
            panel,
            outcome="outcome",
            treatment="treatment",
            unit="unit",
            time="period",
            predictors=PREDICTORS,
        )
        res.in_space_placebo(n_starts=1)
        cs = res.confidence_set(family="constant", gamma=0.1)
        loo = res.leave_one_out(n_starts=1)
        itp = res.in_time_placebo(placebo_periods=IN_TIME_BACKDATES, n_starts=1)
        joint0 = res.conformal_test(0.0)
        joint_true = res.conformal_test(TRUE_RAMP)
        ci = res.conformal_confidence_intervals(alpha=0.1, scheme="moving_block")
        ae = res.conformal_average_effect(alpha=0.1, scheme="moving_block")
    return {
        "panel": panel,
        "res": res,
        "cs": cs,
        "loo": loo,
        "itp": itp,
        "joint0": joint0,
        "joint_true": joint_true,
        "ci": ci,
        "ae": ae,
    }


def test_panel_composition(fit):
    """§1 quoted: 1365 rows = 21 units (1 treated + 20 donors) × 65 quarters."""
    panel = fit["panel"]
    assert len(panel) == (N_DONORS + 1) * (N_PRE + N_POST) == 1365
    assert panel["unit"].nunique() == N_DONORS + 1 == 21
    assert panel["period"].nunique() == N_PRE + N_POST == 65
    assert panel.groupby("unit")["treat"].first().sum() == 1  # single treated unit


def test_true_effect_ramp(fit):
    """§1 quoted: the injected post-period effect ramps 5 -> 9."""
    te = (
        fit["panel"]
        .query("unit == 0 and treatment == 1")
        .sort_values("period")["true_effect"]
        .to_numpy()
    )
    np.testing.assert_array_equal(te, TRUE_RAMP)


def test_att_and_pre_rmspe(fit):
    """§2 quoted: ATT ≈ 6.8 (true mean 7.0), pre-period RMSPE ≈ 0.99.

    Bands (not exact pins) absorb cross-platform BLAS variation in the nested V-search."""
    res = fit["res"]
    assert 6.5 <= res.att <= 7.1, f"att={res.att:.3f} outside [6.5, 7.1]"
    assert 0.85 <= res.pre_rmspe <= 1.15, f"pre_rmspe={res.pre_rmspe:.3f} outside [0.85, 1.15]"


def test_top_donor_weights(fit):
    """§2 quoted: synthetic ≈ 0.51·unit 2 + 0.26·unit 3 + … (sparse; unit 2 dominant)."""
    w = fit["res"].donor_weights
    top_unit = max(w, key=w.get)
    assert top_unit == 2, f"top-weighted donor is {top_unit}, expected 2"
    assert 0.42 <= w[2] <= 0.58, f"weight[2]={w[2]:.3f} outside [0.42, 0.58]"
    assert 0.20 <= w[3] <= 0.32, f"weight[3]={w[3]:.3f} outside [0.20, 0.32]"


def test_predictor_balance(fit):
    """§2 quoted: synthetic matches treated on all 3 covariates, far better than donor mean."""
    pb = fit["res"].predictor_balance
    synth_gap = (pb["treated"] - pb["synthetic"]).abs().max()
    donor_gap = (pb["treated"] - pb["donor_mean"]).abs().max()
    assert synth_gap < 0.1, f"max |treated-synthetic|={synth_gap:.3f} not < 0.1"
    assert donor_gap > synth_gap, "synthetic should match treated better than the raw donor mean"


def test_nan_analytical_inference(fit):
    """Intro claim: classic SCM has no analytical SE — these stay NaN by design."""
    res = fit["res"]
    assert np.isnan(res.se)
    assert np.isnan(res.t_stat)
    assert np.isnan(res.p_value)
    assert all(np.isnan(b) for b in res.conf_int)
    assert res.is_significant is False


def test_in_space_placebo(fit):
    """§4 quoted: placebo p ≈ 0.048 (= 1/21, treated most extreme of 21)."""
    res = fit["res"]
    assert res.n_placebos == N_DONORS == 20
    assert res.placebo_p_value == pytest.approx(1.0 / (N_DONORS + 1), abs=1e-6)
    # Treated unit has the largest post/pre RMSPE ratio in the reference set.
    pl = res.get_placebo_df()
    assert bool(pl.loc[pl["is_treated"], "rmspe_ratio"].iloc[0] >= pl["rmspe_ratio"].max())


def test_constant_confidence_set_wide_and_positive(fit):
    """§4 quoted: the constant-effect set is wide and lopsided (rigid family vs a ramp)."""
    in_set = fit["cs"][fit["cs"]["in_set"]]
    lo, hi = in_set["param"].min(), in_set["param"].max()
    assert lo < 6.0 and hi > 12.0, f"constant set [{lo:.2f}, {hi:.2f}] not wide/positive"


def test_leave_one_out_robust(fit):
    """§4 quoted: dropping any one donor moves the ATT by at most ≈ 0.76."""
    moved = fit["loo"][fit["loo"]["status"] == "loo"]
    max_delta = moved["delta_att"].abs().max()
    assert max_delta < 1.2, f"max |delta_att|={max_delta:.3f} not < 1.2"


def test_in_time_placebo_small(fit):
    """§4 quoted: backdated 'effects' are all |·| < 1 vs the real 5–9."""
    ran = fit["itp"][fit["itp"]["status"] == "ran"]
    assert len(ran) == len(IN_TIME_BACKDATES)
    assert (ran["placebo_att"].abs() < 1.0).all()


def test_conformal_joint_test(fit):
    """§5 quoted: joint p(no effect) ≈ 0.015 (= 1/65); the true ramp is NOT rejected."""
    j0, jt = fit["joint0"], fit["joint_true"]
    assert j0["n_perms"] == N_PRE + N_POST == 65
    assert j0["p_value"] == pytest.approx(1.0 / 65, abs=1e-6)
    assert jt["p_value"] > 0.5, f"true-ramp p={jt['p_value']:.3f} should be large"


def test_pointwise_band_brackets_truth(fit):
    """§5 quoted: the pointwise conformal band contains the true effect in all 5 periods."""
    ci = fit["ci"].sort_values("period")
    covered = ((ci["lower"].to_numpy() <= TRUE_RAMP) & (TRUE_RAMP <= ci["upper"].to_numpy())).sum()
    assert covered == N_POST == 5, f"only {covered}/5 periods bracket the truth"


def test_average_effect_ci(fit):
    """§5 quoted: average-effect 90% CI ≈ [6.5, 7.5], bounded, bracketing 7.0; 13 blocks."""
    ae = fit["ae"]
    assert ae["status"] == "ran"
    assert ae["n_blocks"] == (N_PRE + N_POST) // N_POST == 13
    assert ae["lower"] <= 7.0 <= ae["upper"], "avg-effect CI must bracket the true mean 7.0"
    assert 6.0 <= ae["lower"] <= 6.8, f"avg lower={ae['lower']:.2f} outside [6.0, 6.8]"
    assert 7.2 <= ae["upper"] <= 7.9, f"avg upper={ae['upper']:.2f} outside [7.2, 7.9]"


def test_notebook_kwargs_match():
    """Sync guard: the notebook's generator/fit kwargs must match this module's locked
    config, so a notebook-only edit can't silently invalidate the quoted numbers.

    CI isolation note: the pure-Python / Rust CI jobs copy ``tests/`` without ``docs/``;
    this guard skips gracefully there (nbmake separately verifies execution)."""
    import json
    from pathlib import Path

    nb_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "tutorials"
        / "25_synthetic_control_policy.ipynb"
    )
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}; sync guard is local-dev only.")
    with nb_path.open() as f:
        nb = json.load(f)
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    # Generator + fit kwargs the quoted numbers depend on.
    for needle in (
        "n_donors=20",
        "n_pre=60",
        "n_post=5",
        "treatment_effect=5.0",
        'effect_type="ramp"',
        "effect_growth=1.0",
        "noise_sd=0.6",
        "seed=0",
        'predictors=["x1", "x2", "x3"]',
        "placebo_periods=[20, 35, 50]",
        "inner_min_decrease=1e-3",
    ):
        assert needle in src, f"notebook §1/§2 missing locked kwarg: {needle!r}"

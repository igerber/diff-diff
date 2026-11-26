"""Drift detection for Tutorial 34 (``docs/tutorials/34_design_diagnosis.ipynb``).

The tutorial walks the design-stage workflow on one simulated panel:
``diagnose_design`` on the untreated history, ``lock()``, ``DesignLock.fit`` after
the rollout, and ``BusinessReport(design_reconciliation=...)``. ``pytest --nbmake``
only checks that the cells execute, and ``nbsphinx_execute = "never"``
(``docs/conf.py``) means the committed outputs are what RTD renders, so the prose
can drift from the live library without any cell failing. These tests re-derive
what the prose states and check that the committed surface quotes it.

Two tiers, as in T24:

- **Deterministic** claims run unmarked. They do not depend on the number of
  placebo replicates: the panel shapes, the reconciliation of the locked analysis
  (``match`` on 28 checks), the estimate, and the two deviations. The lock they
  use is made with a handful of replicates, because the reconciled items do not
  depend on the Monte Carlo numbers.
- **Monte Carlo** claims (size verdicts, power, the MDE, the lock rerun) are
  ``@pytest.mark.slow`` and rerun the notebook's diagnosis at its seed and
  replicate count. They assert verdicts and orderings with margins, not exact
  rates; the exact rates the prose quotes are pinned to the committed outputs by
  the quote check.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    BusinessReport,
    CallawaySantAnna,
    HonestDiD,
    ImputationDiD,
    PreTrendsPower,
    TwoWayFixedEffects,
    diagnose_design,
)
from tests._tutorial_drift import assert_quotes_in_rendered, notebook_markdown

NB = "docs/tutorials/34_design_diagnosis.ipynb"

# Locked design -- must stay in sync with the notebook's code cells
# (cross-checked by ``test_notebook_design_matches``).
N_STORES, N_HISTORY, N_FOLLOW_UP = 50, 24, 6
N_TREATED = 25
SEED = 34
TRUE_EFFECT = 0.5
PLACEBO_ADOPTION = 19
EFFECT_SIZES = [0.25, 0.5]
N_SIMULATIONS = 200
COLUMNS = dict(outcome="sales", unit="store", time="month")


# --------------------------------------------------------------------------- #
# Helpers -- faithful copies of the notebook's.
# --------------------------------------------------------------------------- #


def simulate_outcomes(n_periods, rho=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    store_level = rng.normal(0.0, 1.0, N_STORES)
    noise = np.empty((N_STORES, n_periods))
    noise[:, 0] = rng.normal(0.0, 1.0, N_STORES)
    for t in range(1, n_periods):
        noise[:, t] = rho * noise[:, t - 1] + rng.normal(0.0, np.sqrt(1 - rho**2), N_STORES)
    months = np.arange(1, n_periods + 1)
    sales = 10.0 + store_level[:, None] + 0.05 * months[None, :] + noise
    return pd.DataFrame(
        {
            "store": np.repeat(np.arange(N_STORES), n_periods),
            "month": np.tile(months, N_STORES),
            "sales": sales.ravel(),
        }
    )


def analysis_panel():
    analysis = simulate_outcomes(N_HISTORY + N_FOLLOW_UP)
    treated = np.random.default_rng(SEED + 1).choice(N_STORES, size=N_TREATED, replace=False)
    is_treated = analysis["store"].isin(treated)
    analysis["first_treat"] = np.where(is_treated, N_HISTORY + 1, 0)
    analysis.loc[is_treated & (analysis["month"] > N_HISTORY), "sales"] += TRUE_EFFECT
    return analysis


def diagnose(history, estimators, n_simulations, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diagnose_design(
            history,
            estimators=estimators,
            cohort_periods=[PLACEBO_ADOPTION],
            n_treated=N_TREATED,
            n_simulations=n_simulations,
            seed=SEED,
            progress=False,
            **COLUMNS,
            **kwargs,
        )


@pytest.fixture(scope="module")
def history():
    return simulate_outcomes(N_HISTORY)


@pytest.fixture(scope="module")
def analysis():
    return analysis_panel()


@pytest.fixture(scope="module")
def lock(history):
    """A lock of the tutorial's chosen estimator; few replicates on purpose."""
    return diagnose(history, {"cs": CallawaySantAnna()}, n_simulations=5).lock(progress=False)


@pytest.fixture(scope="module")
def check(lock, analysis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lock.fit(analysis, first_treat="first_treat")


def deviating(reconciliation):
    return {c["item"] for c in reconciliation.checks if c["status"] == "deviates"}


# --------------------------------------------------------------------------- #
# Deterministic tier
# --------------------------------------------------------------------------- #


def test_panels(history, analysis):
    assert history.shape == (N_STORES * N_HISTORY, 3)
    assert analysis.shape == (N_STORES * (N_HISTORY + N_FOLLOW_UP), 4)
    # The analysis panel's first 24 months are the history the design saw.
    same = analysis.loc[analysis["month"] <= N_HISTORY, ["store", "month", "sales"]]
    assert same.reset_index(drop=True).equals(history)
    assert "first_treat" not in history.columns
    treated = analysis.groupby("store")["first_treat"].first()
    assert int((treated == N_HISTORY + 1).sum()) == N_TREATED
    assert set(treated.unique()) == {0, N_HISTORY + 1}


def test_locked_analysis_matches(lock, check):
    assert lock.estimator_name == "CallawaySantAnna"
    assert check.status == "match"
    assert check.declared is False
    assert len(check.checks) == 28  # "all 28 checks"
    assert {c["status"] for c in check.checks} == {"match"}


def test_estimate(check):
    """Section 6: ATT 0.393 with an interval that includes zero and covers 0.5."""
    estimate = check.results
    low, high = estimate.conf_int
    assert round(float(estimate.att), 3) == 0.393
    assert round(float(estimate.se), 3) == 0.228
    assert low < 0.0 < TRUE_EFFECT < high
    assert estimate.p_value > 0.05


def test_report_states_the_reconciliation(check):
    report = BusinessReport(
        check.results,
        design_reconciliation=check,
        outcome_label="the sales index",
        treatment_label="the new store layout",
        auto_diagnostics=False,
    )
    sentence = (
        "The analysis was fitted from design lock 'cs' and agrees with it on all 28 "
        "checks that were run. The lock does not record when it was made."
    )
    assert sentence in report.summary()
    assert "## Design Reconciliation" in report.full_report()
    block = report.to_dict()["design_reconciliation"]
    assert block["status"] == "match" and block["declared"] is False
    assert block["n_checks"] == block["n_match"] == 28


def test_shorter_follow_up_deviates(lock, analysis):
    """Section 8: a four-month readout deviates on the post window only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        early = lock.fit(analysis[analysis["month"] <= N_HISTORY + 4], first_treat="first_treat")
    assert early.status == "deviates" and early.declared is False
    assert deviating(early) == {"n_post_by_cohort"}
    row = next(c for c in early.checks if c["item"] == "n_post_by_cohort")
    assert (row["locked"], row["realized"]) == ([6], [4])


def test_by_hand_fit_with_another_comparison_group(lock, analysis, check):
    """Section 8: the deviation is named, and the estimate is the same here."""
    estimator = CallawaySantAnna(control_group="not_yet_treated")
    columns = dict(first_treat="first_treat", **COLUMNS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_hand = estimator.fit(analysis, **columns)
    declared = lock.reconcile(estimator, data=analysis, results=by_hand, **columns)
    assert declared.status == "deviates" and declared.declared is True
    assert deviating(declared) == {"params.control_group"}
    # "the two estimates are the same to three decimals"
    assert round(float(by_hand.att), 3) == round(float(check.results.att), 3)
    report = BusinessReport(by_hand, design_reconciliation=declared, auto_diagnostics=False)
    assert "As declared to reconcile(), the analysis deviates from design lock 'cs'" in (
        report.summary()
    )
    assert "design_deviation" in {caveat["topic"] for caveat in report.caveats()}


def test_quotes_in_rendered_surface():
    assert_quotes_in_rendered(
        NB,
        [
            "Size threshold with 200 replicates: 8.0%",
            "Status:                  match",
            "Checks:                  28 match, 0 deviate, 0 not checked",
            "Configuration:           fitted from the lock",
            "Configuration:           declared to reconcile()",
            "ATT 0.393",
            "all 28 checks that were run",
            "warning | design_deviation",
            "By-hand ATT 0.393 vs locked ATT 0.393",
            # Section 5: the planned pre-trend check (500 replicates, seed 36).
            "             2 0.357               0.084                   0.092",
            "             3   NaN               0.111                   0.152",
            "rejected 29.6% of the 500 no-effect replicates, against 25.0%",
            # Section 5: the planned sensitivity analysis (same 500 draws).
            "0.0                    0.04     0.890 0.652  0.616",
            "0.5                    0.00     2.178 1.311  0.010",
            "1.0                    0.00     3.467 2.046  0.000",
            "0.0                   0.202      0.73  NaN    NaN",
            "M=0 (excludes zero with no effect in 20.2%; 0",
        ],
        surface="output",
    )
    assert_quotes_in_rendered(
        NB,
        [
            "**32.5%**",  # twfe_classical, no-effect rejection rate
            "**56.0%**",  # cs power at 0.5, comparison
            "**0.65**",  # cs MDE, comparison
            "**8.0%**",  # size threshold
            "**6.5%**",  # cs no-effect rejection rate, rerun
            "**55.5%**",  # cs power at 0.5, rerun
            "**28 checks**",
            "**0.393**",
            "**8.4%**",  # formula's false-alarm rate, two pre-period months
            "**9.2%**",  # the same rate counted in the placebo draws
            "**0.357**",  # planned MDV, two pre-period months
            "**15.2%**",  # counted, three months
            "**11.1%**",  # assumed, three months
            "**0.652**",  # relative magnitudes, planned MDE at M = 0
            "**1.311**",  # at M = 0.5
            "**2.046**",  # at M = 1
            "**61.6%**",  # power at an effect of 0.5, M = 0
            "**1.0%**",  # the same at M = 0.5
            "**4.0%**",  # no-effect draws whose interval excluded zero, M = 0
            "**20.2%**",  # the same for the smoothness interval at M = 0
        ],
        surface="markdown",
    )


def test_prose_never_claims_pre_specification():
    prose = notebook_markdown(NB).lower()
    for claim in ("pre-specified", "prespecified", "pre-registered analysis"):
        assert claim not in prose
    assert "the lock does not record when it was made" in prose


def test_notebook_design_matches():
    """Sync guard: the notebook's design constants and calls match this module."""
    import json
    from pathlib import Path

    nb_path = Path(__file__).resolve().parents[1] / NB
    if not nb_path.exists():
        pytest.skip(f"Notebook not found at {nb_path}; sync guard is local-dev only.")
    nb = json.loads(nb_path.read_text())
    src = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
    for fragment in (
        "N_STORES, N_HISTORY, N_FOLLOW_UP = 50, 24, 6",
        "N_TREATED = 25",
        "SEED = 34",
        "def simulate_outcomes(n_periods, rho=0.8, seed=SEED):",
        "cohort_periods=[19]",
        "effect_sizes=[0.25, 0.5]",
        "n_simulations=200",
        "power=0.80",
        '"twfe_classical": TwoWayFixedEffects(vcov_type="classical")',
        'result.lock("cs", data=history, progress=False)',
        "TRUE_EFFECT = 0.5",
        "np.random.default_rng(SEED + 1).choice(N_STORES, size=N_TREATED, replace=False)",
        'lock.fit(analysis, first_treat="first_treat")',
        'CallawaySantAnna(control_group="not_yet_treated")',
        'estimators={"cs": CallawaySantAnna(base_period="universal")}',
        "n_simulations=500",
        "seed=SEED + 2,",
        "event_study=True,",
        "PreTrendsPower().power_by_pre_periods(plan)",
        "relative = HonestDiD().power_by_M(",
        "M_grid=[0, 0.5, 1.0],",
        "effect_size=0.5,",
        'smooth = HonestDiD(method="smoothness").power_by_M(plan, M_grid=[0], progress=False)',
    ):
        assert fragment in src, f"notebook no longer contains: {fragment}"


# --------------------------------------------------------------------------- #
# Monte Carlo tier
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def diagnosis(history):
    return diagnose(
        history,
        {
            "twfe": TwoWayFixedEffects(),
            "twfe_classical": TwoWayFixedEffects(vcov_type="classical"),
            "cs": CallawaySantAnna(),
            "imp": ImputationDiD(),
        },
        n_simulations=N_SIMULATIONS,
        effect_sizes=EFFECT_SIZES,
        power=0.80,
    )


@pytest.mark.slow
def test_size_verdicts(diagnosis):
    """Section 3: classical SEs over-reject by a wide margin; the others pass."""
    size = {label: runs[0].rejection_rate for label, runs in diagnosis.results.items()}
    threshold = 0.05 + 1.96 * np.sqrt(0.05 * 0.95 / N_SIMULATIONS)
    assert round(threshold, 3) == 0.080
    assert size["twfe_classical"] > 0.25  # prose: 32.5%
    assert diagnosis.size_controlled["twfe_classical"] is False
    assert np.isnan(diagnosis.mde["twfe_classical"])
    for label in ("twfe", "cs", "imp"):
        assert size[label] <= threshold
        assert diagnosis.size_controlled[label] is True
    # "cs passes at 7.0%, which is closer to the threshold than twfe or imp"
    assert size["cs"] > max(size["twfe"], size["imp"])


@pytest.mark.slow
def test_power_and_mde(diagnosis):
    """Section 3: cs has the most power and the smallest MDE among those passing."""
    passing = ("twfe", "cs", "imp")
    for cell in (1, 2):
        power = {label: diagnosis.results[label][cell].power for label in passing}
        assert max(power, key=power.get) == "cs"
    assert abs(diagnosis.results["cs"][2].power - 0.56) < 0.05  # "roughly a coin flip"
    assert min(passing, key=lambda label: diagnosis.mde[label]) == "cs"
    assert abs(diagnosis.mde["cs"] - 0.65) < 0.05
    assert diagnosis.mde["cs"] > TRUE_EFFECT  # the design cannot reliably detect 0.5


@pytest.mark.slow
def test_lock_rerun(diagnosis, history):
    """Section 4: the rerun is close to the comparison and keeps its verdict."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lock = diagnosis.lock("cs", data=history, progress=False)
    assert not [w for w in caught if "different verdicts" in str(w.message)]
    assert lock.size_controlled is True
    null, _, at_half = lock.rerun_results
    assert null["true_effect"] == 0.0 and at_half["true_effect"] == 0.5
    assert abs(null["rejection_rate"] - 0.065) < 0.02
    assert abs(at_half["rejection_rate"] - 0.555) < 0.05
    assert abs(lock.mde - 0.66) < 0.05


def pretrend_plan(history, n_simulations):
    """Section 5's run: the chosen estimator alone, new seed, universal base."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = diagnose_design(
            history,
            estimators={"cs": CallawaySantAnna(base_period="universal")},
            cohort_periods=[PLACEBO_ADOPTION],
            n_treated=N_TREATED,
            n_simulations=n_simulations,
            seed=SEED + 2,
            event_study=True,
            progress=False,
            **COLUMNS,
        )
        return plan, PreTrendsPower().power_by_pre_periods(plan)


@pytest.mark.slow
def test_pretrend_plan(history):
    """Section 5: two rows are planned, the checks over three or more months are not."""
    plan, table = pretrend_plan(history, 500)
    assert len(table) == 17
    assert table["mdv"].notna().tolist() == [True, True] + [False] * 15
    two, three = table.iloc[1], table.iloc[2]
    assert two["mdv"] == pytest.approx(0.357, abs=2e-3)
    assert two["power_no_violation"] == pytest.approx(0.084, abs=1e-3)
    assert two["placebo_rejection_rate"] == pytest.approx(0.092, abs=1e-12)
    assert three["power_no_violation"] == pytest.approx(0.111, abs=1e-3)
    assert three["placebo_rejection_rate"] == pytest.approx(0.152, abs=1e-12)
    checks = plan._event_study_checks["cs"]
    assert checks["pretest_rejection_rate"] == pytest.approx(0.296, abs=1e-12)
    assert checks["pretest_expected_rate"] == pytest.approx(0.250, abs=1e-3)
    assert checks["pretest_agrees"] is False


@pytest.mark.slow
def test_pretrend_plan_with_more_replicates(history):
    """Section 5: with 2,000 replicates every row is blank; one coefficient rejects ~6%."""
    _, table = pretrend_plan(history, 2000)
    assert table["mdv"].isna().all()
    assert 0.055 <= table.iloc[0]["placebo_rejection_rate"] <= 0.07


@pytest.mark.slow
def test_sensitivity_plan(history):
    """Section 5: relative magnitudes is planned, the smoothness interval is withheld."""
    plan, _ = pretrend_plan(history, 500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        relative = HonestDiD().power_by_M(
            plan, M_grid=[0, 0.5, 1.0], effect_size=TRUE_EFFECT, progress=False
        )
        smooth = HonestDiD(method="smoothness").power_by_M(plan, M_grid=[0], progress=False)
    assert relative["n_simulations"].tolist() == [500, 500, 500]
    assert relative["n_simulation_failures"].tolist() == [0, 0, 0]
    assert relative["placebo_rejection_rate"].tolist() == pytest.approx([0.04, 0.0, 0.0], abs=1e-12)
    assert relative["mde"].tolist() == pytest.approx([0.652, 1.311, 2.046], abs=2e-3)
    assert relative["power"].tolist() == pytest.approx([0.616, 0.010, 0.0], abs=1e-12)
    assert relative["ci_width"].tolist() == pytest.approx([0.890, 2.178, 3.467], abs=2e-3)
    row = smooth.iloc[0]
    assert row["placebo_rejection_rate"] == pytest.approx(0.202, abs=1e-12)
    assert row["ci_width"] == pytest.approx(0.73, abs=5e-3)
    assert np.isnan(row["mde"])

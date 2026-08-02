"""Monte Carlo coverage simulation for the cell-period IF allocator
applied to the heterogeneity WLS path (PR 3).

Validates empirical null coverage under a DGP with PSU that varies
across cells of the same group — the regime unlocked by PR 3's lift
of the heterogeneity gate. Under within-group-constant PSU the cell
allocator's PSU-level aggregate matches the previous group allocator
byte-for-byte (see `TestHeterogeneityCellPeriod::
test_psu_level_byte_identity_under_psu_equals_group` in
`test_survey_dcdh.py`), so the coverage check only exercises the new
within-group-varying-PSU code path.

Null-coverage rationale:

    beta_het_true = 0 is deliberate. The heterogeneity test's
    identifying estimand under Lemma 7 is a variance-weighted average
    of effect differences, so a non-zero DGP-level beta does not have
    a clean "true value" for coverage. A constant-treatment-effect DGP
    ensures beta_het_true = 0 exactly, so the CI must cover 0 at
    nominal 95% regardless of Lemma 7's variance weighting.

Marked ``slow`` and excluded from the default pytest run. To execute:

    pytest tests/test_dcdh_heterogeneity_cell_period_coverage.py -m slow -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille
from diff_diff.survey import SurveyDesign


def _simulate_panel(
    n_groups: int,
    n_periods: int,
    first_treated_period: int,
    tau: float,
    rng: np.random.Generator,
    psu_sigma: float = 0.5,
    obs_sigma: float = 1.0,
    group_sigma: float = 1.0,
) -> pd.DataFrame:
    """Generate a one-obs-per-cell panel with within-group-varying PSU
    and a group-level time-invariant covariate X_het.

    Treatment effect ``tau`` is constant (independent of ``X_het``) so
    the true heterogeneity coefficient on X_het is exactly 0 under
    Lemma 7 — the null-coverage target for this test.
    """
    groups = np.arange(n_groups)
    treated = groups < n_groups // 2
    group_fe = rng.normal(0.0, group_sigma, size=n_groups)
    # (group, psu) effect — constant across time within the (g, psu)
    # pair. Drives within-group residual correlation that varies by
    # cell, so design-based variance must capture it per-PSU.
    psu_fe = rng.normal(0.0, psu_sigma, size=(n_groups, 2))
    # Group-level binary covariate (time-invariant).
    x_het = rng.binomial(1, 0.5, size=n_groups).astype(float)

    rows = []
    for g in groups:
        for t in range(n_periods):
            parity = int(t % 2)
            # PSU labels are globally unique per (group, parity) so
            # Binder variance sees 2 * n_groups sampling clusters
            # rather than 2 global PSU codes reused across groups.
            # Within-cell constancy (one obs per cell) is trivially
            # satisfied.
            psu_id = int(g) * 2 + parity
            d = 1 if (treated[g] and t >= first_treated_period) else 0
            y = group_fe[g] + 0.1 * t + tau * d + psu_fe[g, parity] + rng.normal(0.0, obs_sigma)
            rows.append(
                {
                    "group": int(g),
                    "period": int(t),
                    "treatment": int(d),
                    "outcome": float(y),
                    "psu": psu_id,
                    "pw": 1.0,
                    "x_het": float(x_het[g]),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_heterogeneity_cell_period_null_coverage_varying_psu():
    """Empirical 95% null coverage for beta_het under the cell-period
    allocator on a DGP with within-group-varying PSU. Tolerance ±2.5pp
    mirrors the ATT coverage test in test_dcdh_cell_period_coverage.py
    (about 2 MC-sigma at n_reps=500 for a true 95% target).
    """
    n_reps = 500
    n_groups = 40
    n_periods = 6
    first_treated_period = 3
    tau_true = 2.0

    rng = np.random.default_rng(20260419)
    covered = 0
    failed = 0

    for r in range(n_reps):
        df = _simulate_panel(
            n_groups=n_groups,
            n_periods=n_periods,
            first_treated_period=first_treated_period,
            tau=tau_true,
            rng=rng,
        )
        sd = SurveyDesign(weights="pw", psu="psu")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ChaisemartinDHaultfoeuille(seed=r + 1).fit(
                    df,
                    outcome="outcome",
                    unit="group",
                    time="period",
                    treatment="treatment",
                    heterogeneity="x_het",
                    survey_design=sd,
                    L_max=1,
                )
        except Exception:
            failed += 1
            continue

        if res.heterogeneity_effects is None:
            failed += 1
            continue
        entry = res.heterogeneity_effects[1]
        ci = entry["conf_int"]
        if ci is None or not all(np.isfinite(ci)):
            failed += 1
            continue
        lo, hi = float(ci[0]), float(ci[1])
        # Null coverage: beta_het_true = 0 since tau is constant.
        if lo <= 0.0 <= hi:
            covered += 1

    completed = n_reps - failed
    assert completed >= int(0.95 * n_reps), (
        f"MC simulation had {failed}/{n_reps} fit failures, above " f"the 5% tolerance."
    )
    coverage = covered / completed
    assert 0.925 <= coverage <= 0.975, (
        f"Empirical null coverage {coverage:.3f} (completed "
        f"{completed}) outside [0.925, 0.975]; tau_true={tau_true}, "
        f"beta_het_true=0, n_groups={n_groups}, n_periods={n_periods}, "
        f"n_reps={n_reps}."
    )

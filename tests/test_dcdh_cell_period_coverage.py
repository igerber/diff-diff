"""Monte Carlo coverage simulation for the cell-period IF allocator.

Validates empirical coverage under a DGP with PSU that varies across
cells of the same group — the regime unlocked by PR 2's relaxation of
the within-group constancy rule. Under within-group-constant PSU the
cell allocator reduces to the previous group allocator byte-for-byte,
so the coverage check only exercises the new code path.

Marked ``slow`` and excluded from the default pytest run. To execute:

    pytest tests/test_dcdh_cell_period_coverage.py -m slow -v
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
    """Generate a one-obs-per-cell panel with within-group-varying PSU.

    Each group has two PSUs assigned by ``period % 2``. The PSU
    contributes a ``(group, psu)``-specific intercept, creating the
    within-group residual correlation that design-based variance is
    meant to capture. Treatment starts at ``first_treated_period`` for
    the first half of groups and never for the rest.
    """
    groups = np.arange(n_groups)
    treated = groups < n_groups // 2
    group_fe = rng.normal(0.0, group_sigma, size=n_groups)
    # (group, psu) effect — constant across time within the (g, psu)
    # pair. Drives within-group correlation that varies by cell.
    psu_fe = rng.normal(0.0, psu_sigma, size=(n_groups, 2))

    rows = []
    for g in groups:
        for t in range(n_periods):
            parity = int(t % 2)
            # PSU labels are globally unique per (group, parity) so the
            # Binder variance sees 2 * n_groups sampling clusters rather
            # than just 2 global PSU codes reused across groups. Each
            # PSU collects the cells of a single group's odd or even
            # periods; within-cell constancy (one obs per cell) is
            # trivially satisfied.
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
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_cell_period_allocator_coverage_within_group_varying_psu():
    """Empirical 95% coverage for DID_M under the cell-period allocator
    on a DGP with within-group-varying PSU. Tolerance ±2.5pp is ~2
    MC-sigma at n_reps=500 for a true 95% target (sqrt(0.95*0.05/500)
    = 0.0098, so 2-sigma = 1.96pp — ±2.5pp is deliberately a touch
    looser to absorb small-sample bias without masking a real
    under-coverage bug).
    """
    n_reps = 500
    n_groups = 40
    n_periods = 6
    first_treated_period = 3
    tau_true = 2.0

    rng = np.random.default_rng(20260418)
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
                    survey_design=sd,
                    L_max=1,
                )
        except Exception:
            failed += 1
            continue

        ci = res.event_study_effects[1]["conf_int"]
        if ci is None or not all(np.isfinite(ci)):
            failed += 1
            continue
        lo, hi = float(ci[0]), float(ci[1])
        if lo <= tau_true <= hi:
            covered += 1

    completed = n_reps - failed
    assert completed >= int(0.95 * n_reps), (
        f"MC simulation had {failed}/{n_reps} fit failures, above " f"the 5% tolerance."
    )
    coverage = covered / completed
    assert 0.925 <= coverage <= 0.975, (
        f"Empirical coverage {coverage:.3f} (completed {completed}) "
        f"outside [0.925, 0.975]; true tau={tau_true}, "
        f"n_groups={n_groups}, n_periods={n_periods}, n_reps={n_reps}."
    )

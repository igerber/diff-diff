#!/usr/bin/env python3
"""Does planned HonestDiD power, counted on placebo draws, match what happens?

``diagnose_design(..., event_study=True)`` keeps the event-study container of
every no-effect placebo draw. ``HonestDiD.power_by_M`` runs the sensitivity
analysis on each of them and counts: how often the robust interval excludes
zero with no effect, and the smallest effect it would detect with the target
power (``mde``). The effect is never injected there: an effect that is the
same across treated units moves a draw's interval by the value of the target
parameter and changes nothing else. Rambachan and Roth (2023) give no
planning procedure; this is the library's own. This script checks it:

1. Level. For each design point, candidate and restriction, the rate at which
   the robust interval excludes zero in the no-effect draws of the diagnosis,
   and whether the size rule of ``power_by_M`` reports or withholds the row.
2. Out-of-sample check, with no use of the shift identity. On fresh placebo
   draws: (a) with no effect, the rate at which the interval excludes zero;
   (b) with the planned ``mde`` really added to the treated units' outcomes
   after adoption, the share of draws whose interval lies entirely on the
   effect's side of zero. The injected sign is the one that sets ``mde`` (it
   is the larger of the two signs' thresholds), so if the plan is right (b)
   is the target power, 0.80. Rows the size rule withholds are run too, at
   the value that would have been reported, and tabulated separately.
3. Monte Carlo stability. The planned ``mde`` over 10 diagnosis seeds at 200
   and 500 replicates.
4. Seconds taken by the per-draw sensitivity analyses of step 1, per row.

Everything below was fixed before the first run and is not searched.

Panels: as in ``pretrend_planning_check.py``. ``y_it = a_i + d_t + e_it``,
``a_i`` and ``d_t`` ``N(0, 1)``, ``e_it`` an AR(1) with coefficient ``rho``
and innovation sd 1 started from its stationary distribution. 12 periods;
``rho`` in {0, 0.5, 0.8}; 40 and 200 units; treated share 0.25; units
resampled in every draw (the default). Common adoption at period 9, and a
staggered design with adoption at 8 and 10. 12 design points, one fixed panel
each.

Candidates: ``CallawaySantAnna(base_period="universal")`` on all 12 points;
``StackedDiD(kappa_pre=5, kappa_post=3)`` on the 6 common-adoption points.
18 runs.

Restrictions, per run: ``method="smoothness"`` at ``M = 0`` and at one
positive value (a quarter of the placebo standard deviation of the average
post-period coefficient, so that it scales with the panel), and
``method="relative_magnitude"`` at 0.5 and 1.0. Default ``l_vec`` (the
average post-period effect) and ``alpha = 0.05``.

Flags: a row is flagged for power when the share in 2(b) differs from 0.80 by
more than 5 points plus two Monte Carlo standard errors, and for level when
the rate in 2(a) exceeds ``alpha`` by more than 3 points plus two Monte Carlo
standard errors. The claim under test is that no row the size rule reports is
flagged.

Usage::

    python benchmarks/python/sensitivity_planning_check.py \\
        --output benchmarks/data/sensitivity_planning_check.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diff_diff import CallawaySantAnna, HonestDiD, StackedDiD, diagnose_design  # noqa: E402
from diff_diff import design as _design  # noqa: E402
from diff_diff.honest_did import _score_planned_intervals  # noqa: E402

SEED = 20261127
N_PERIODS = 12
TREATED_SHARE = 0.25
ALPHA = 0.05
TARGET = 0.80
N_DIAGNOSIS = 500
N_DIRECT = 1000
RHOS = (0.0, 0.5, 0.8)
UNITS = (40, 200)
DESIGNS = {"common": [9], "staggered": [8, 10]}
STABILITY_SEEDS = 10
STABILITY_SIZES = (200, 500)
BAND_POWER = 0.05
BAND_LEVEL = 0.03
_ERRORS = (ValueError, np.linalg.LinAlgError, KeyError, RuntimeError, ZeroDivisionError)


def make_panel(n_units, rho, rng):
    """A fixed untreated panel with unit and period effects and AR(1) errors."""
    unit_effect = rng.normal(size=n_units)
    period_effect = rng.normal(size=N_PERIODS)
    errors = np.empty((n_units, N_PERIODS))
    errors[:, 0] = rng.normal(scale=1.0 / np.sqrt(1.0 - rho**2), size=n_units)
    for t in range(1, N_PERIODS):
        errors[:, t] = rho * errors[:, t - 1] + rng.normal(size=n_units)
    y = unit_effect[:, None] + period_effect[None, :] + errors
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), N_PERIODS),
            "t": np.tile(np.arange(1, N_PERIODS + 1), n_units),
            "y": y.ravel(),
        }
    )


def candidates(design):
    out = {"CallawaySantAnna": CallawaySantAnna(base_period="universal")}
    if design == "common":
        out["StackedDiD"] = StackedDiD(kappa_pre=5, kappa_post=3)
    return out


def diagnose(panel, estimators, cohorts, n_simulations, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diagnose_design(
            panel,
            outcome="y",
            unit="id",
            time="t",
            estimators=estimators,
            cohort_periods=cohorts,
            treatment_fraction=TREATED_SHARE,
            n_simulations=n_simulations,
            alpha=ALPHA,
            seed=seed,
            progress=False,
            event_study=True,
        )


def positive_m(surface):
    """A quarter of the placebo sd of the average post-period coefficient."""
    index = np.asarray(surface.vcov_index)
    post = np.flatnonzero(index >= 0)
    weights = np.full(post.size, 1.0 / post.size)
    variance = float(weights @ np.asarray(surface.vcov)[np.ix_(post, post)] @ weights)
    return 0.25 * float(np.sqrt(variance))


def specs(m_positive):
    return [
        ("smoothness", 0.0),
        ("smoothness", float(m_positive)),
        ("relative_magnitude", 0.5),
        ("relative_magnitude", 1.0),
    ]


def intervals(method, M, containers):
    """The robust interval ``HonestDiD.fit`` gives on each container (NaN if it fails)."""
    honest = HonestDiD(method=method, alpha=ALPHA)
    out = np.full((len(containers), 2), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, container in enumerate(containers):
            if container is None:
                continue
            try:
                fitted = honest.fit(container, M=M)
            except _ERRORS:
                continue
            out[i] = (fitted.ci_lb, fitted.ci_ub)
    return out


def plan(method, M, draws, n_lost):
    """One row as ``power_by_M`` scores it, and the mde without the size rule."""
    started = time.perf_counter()
    cis = intervals(method, M, draws)
    seconds = time.perf_counter() - started
    row = _score_planned_intervals(cis[:, 0], cis[:, 1], n_lost, ALPHA, TARGET, None)
    scored = np.isfinite(cis).all(axis=1)
    a, b = cis[scored, 0], cis[scored, 1]
    unchecked, sign = float("nan"), 1.0
    if a.size:
        k = max(int(np.ceil(TARGET * a.size - 1e-9)), 1)
        positive = float(np.sort(-a)[k - 1])
        negative = float(np.sort(b)[k - 1])
        unchecked = max(positive, negative, 0.0)
        sign = 1.0 if positive >= negative else -1.0
    return row, unchecked, sign, seconds


def fresh_containers(panel, estimator, cohorts, seeds, effect):
    """Event-study containers of fresh placebo draws with ``effect`` really injected."""
    n_units = panel["id"].nunique()
    n_treated = int(n_units * TREATED_SHARE)
    generator = _design._make_placebo_generator(panel, "y", "id", "t", n_treated, cohorts, True)
    fit_kwargs = _design._fit_kwargs(estimator, "y", "id", "t", {})
    out = []
    for seed in seeds:
        frame = generator(treatment_effect=float(effect), seed=int(seed))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = type(estimator)(**estimator.get_params()).fit(frame, **fit_kwargs)
                out.append(fitted.aggregate("event_study"))
        except _ERRORS:
            out.append(None)
    return out


def _se(p, n):
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n)) if n else float("nan")


def run_candidate(task):
    """Steps 1, 2 and 4 for one design point and candidate."""
    rho, n_units, design, label, panel_seed, diag_seed, direct_seeds = task
    cohorts = DESIGNS[design]
    panel = make_panel(n_units, rho, np.random.default_rng(panel_seed))
    estimator = candidates(design)[label]
    diagnosis = diagnose(panel, {label: estimator}, cohorts, N_DIAGNOSIS, diag_seed)
    out = {"rho": rho, "n_units": n_units, "design": design, "candidate": label}
    surface = diagnosis.event_study_results.get(label)
    if surface is None:
        return {**out, "recorded": False}
    draws, n_lost = _design._placebo_draw_record(surface)
    m_positive = positive_m(surface)
    fresh_null = fresh_containers(panel, estimator, cohorts, direct_seeds, 0.0)
    rows = []
    for method, M in specs(m_positive):
        row, unchecked, sign, seconds = plan(method, M, draws, n_lost)
        null = intervals(method, M, fresh_null)
        scored = np.isfinite(null).all(axis=1)
        n_null = int(scored.sum())
        null_rate = (
            float(np.mean((null[scored, 0] > 0) | (null[scored, 1] < 0)))
            if n_null
            else float("nan")
        )
        observed, n_alt = float("nan"), 0
        if np.isfinite(unchecked) and unchecked > 0:
            effect = sign * unchecked
            alt = intervals(
                method, M, fresh_containers(panel, estimator, cohorts, direct_seeds, effect)
            )
            scored_alt = np.isfinite(alt).all(axis=1)
            n_alt = int(scored_alt.sum())
            if n_alt:
                detected = alt[scored_alt, 0] > 0 if sign > 0 else alt[scored_alt, 1] < 0
                observed = float(np.mean(detected))
        reported = bool(row["size_controlled"])
        rows.append(
            {
                "method": method,
                "M": round(float(M), 6),
                "placebo_rejection_rate": round(float(row["placebo_rejection_rate"]), 4),
                "ci_width": round(float(row["ci_width"]), 6),
                "n_simulations": int(row["n_simulations"]),
                "n_simulation_failures": int(row["n_simulation_failures"]),
                "reported": reported,
                "mde": round(float(row["mde"]), 6) if reported else None,
                "mde_unchecked": round(unchecked, 6),
                "injected_sign": int(sign),
                "fresh_no_effect_rate": round(null_rate, 4),
                "fresh_no_effect_n": n_null,
                "observed_power": round(observed, 4),
                "observed_power_n": n_alt,
                "flag_power": bool(
                    n_alt > 0 and abs(observed - TARGET) > BAND_POWER + 2.0 * _se(TARGET, n_alt)
                ),
                "flag_level": bool(
                    n_null > 0 and null_rate > ALPHA + BAND_LEVEL + 2.0 * _se(ALPHA, n_null)
                ),
                "seconds": round(seconds, 2),
            }
        )
    return {
        **out,
        "recorded": True,
        "n_draws": len(draws),
        "n_lost": int(n_lost),
        "M_positive": round(m_positive, 6),
        "rows": rows,
    }


def run_stability(task):
    """Step 3: the planned mde over diagnosis seeds, one (panel, size) per task."""
    n_units, n_simulations, panel_seed, seeds = task
    panel = make_panel(n_units, 0.5, np.random.default_rng(panel_seed))
    label = "CallawaySantAnna"
    estimator = CallawaySantAnna(base_period="universal")
    m_positive = None
    values = {}
    for seed in seeds:
        diagnosis = diagnose(panel, {label: estimator}, DESIGNS["common"], n_simulations, int(seed))
        surface = diagnosis.event_study_results.get(label)
        if surface is None:
            continue
        draws, n_lost = _design._placebo_draw_record(surface)
        if m_positive is None:
            # Fixed from the first seed so that every seed plans the same M.
            m_positive = positive_m(surface)
        for method, M in (
            ("smoothness", 0.0),
            ("smoothness", m_positive),
            ("relative_magnitude", 1.0),
        ):
            _row, unchecked, _sign, _seconds = plan(method, M, draws, n_lost)
            values.setdefault((method, round(float(M), 6)), []).append(unchecked)
    rows = []
    for (method, M), mdes in values.items():
        mdes = np.asarray(mdes, dtype=float)
        rows.append(
            {
                "n_units": n_units,
                "n_simulations": n_simulations,
                "method": method,
                "M": M,
                "n_seeds": int(mdes.size),
                "mde_mean": round(float(np.mean(mdes)), 6),
                "mde_sd": round(float(np.std(mdes, ddof=1)), 6),
                "relative_spread": round(float(np.std(mdes, ddof=1) / np.mean(mdes)), 4),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="time 40 replicates of one run and exit",
    )
    args = parser.parse_args()

    root = np.random.SeedSequence(SEED)
    panel_children, diag_children, direct_children, stability_child = root.spawn(4)
    grid = [(rho, n, design) for rho in RHOS for n in UNITS for design in DESIGNS]
    panel_seeds = panel_children.spawn(len(grid))
    diag_seeds = diag_children.generate_state(len(grid))
    direct_seed_seqs = direct_children.spawn(len(grid))
    point_index = {point: i for i, point in enumerate(grid)}

    if args.timing_only:
        panel = make_panel(200, 0.5, np.random.default_rng(panel_seeds[0]))
        label, estimator = "CallawaySantAnna", CallawaySantAnna(base_period="universal")
        started = time.perf_counter()
        diagnosis = diagnose(panel, {label: estimator}, DESIGNS["common"], 40, 1)
        per_fit = (time.perf_counter() - started) / 40
        surface = diagnosis.event_study_results[label]
        draws, n_lost = _design._placebo_draw_record(surface)
        _row, _mde, _sign, seconds = plan("smoothness", positive_m(surface), draws, n_lost)
        per_solve = seconds / len(draws)
        total = (113000 * per_fit + 59000 * per_solve) / 3600.0
        print(
            f"about {per_fit * 1e3:.0f} ms per fit and {per_solve * 1e3:.0f} ms per "
            f"smoothness solve at 200 units; 113,000 fits and 59,000 solves = "
            f"{total:.1f} core-hours, {total * 60 / args.workers:.0f} minutes on "
            f"{args.workers} workers"
        )
        return

    started = time.perf_counter()
    # Largest runs first, so the pool stays balanced.
    tasks = []
    for n in sorted(UNITS, reverse=True):
        for rho in RHOS:
            for design in DESIGNS:
                i = point_index[(rho, n, design)]
                direct_seeds = direct_seed_seqs[i].generate_state(N_DIRECT).tolist()
                for label in candidates(design):
                    tasks.append(
                        (rho, n, design, label, panel_seeds[i], int(diag_seeds[i]), direct_seeds)
                    )
    stability_tasks = []
    stability_seeds = stability_child.spawn(len(UNITS) * len(STABILITY_SIZES))
    position = 0
    for n in UNITS:
        i = point_index[(0.5, n, "common")]
        for size in STABILITY_SIZES:
            seeds = stability_seeds[position].generate_state(STABILITY_SEEDS).tolist()
            stability_tasks.append((n, size, panel_seeds[i], seeds))
            position += 1

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        stability_future = [pool.submit(run_stability, task) for task in stability_tasks]
        runs = list(pool.map(run_candidate, tasks))
        stability = [row for future in stability_future for row in future.result()]

    order = {point: i for i, point in enumerate(grid)}
    runs.sort(
        key=lambda run: (order[(run["rho"], run["n_units"], run["design"])], run["candidate"])
    )
    rows = [row for run in runs if run["recorded"] for row in run["rows"]]
    reported = [row for row in rows if row["reported"]]
    withheld = [row for row in rows if not row["reported"]]

    def _count(cells, key):
        return int(sum(cell[key] for cell in cells))

    payload = {
        "meta": {
            "seed": SEED,
            "n_periods": N_PERIODS,
            "treated_share": TREATED_SHARE,
            "alpha": ALPHA,
            "target_power": TARGET,
            "n_diagnosis": N_DIAGNOSIS,
            "n_direct": N_DIRECT,
            "band_power": BAND_POWER,
            "band_level": BAND_LEVEL,
            "n_runs": len(runs),
            "n_rows": len(rows),
            "n_rows_reported": len(reported),
            "n_rows_withheld": len(withheld),
            "n_flag_power_reported": _count(reported, "flag_power"),
            "n_flag_level_reported": _count(reported, "flag_level"),
            "n_flag_power_withheld": _count(withheld, "flag_power"),
            "n_flag_level_withheld": _count(withheld, "flag_level"),
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
            "workers": args.workers,
        },
        "runs": runs,
        "stability": stability,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    meta = payload["meta"]
    print(
        f"{meta['n_rows']} rows: {meta['n_rows_reported']} reported "
        f"({meta['n_flag_power_reported']} flagged for power, "
        f"{meta['n_flag_level_reported']} for level), {meta['n_rows_withheld']} withheld "
        f"({meta['n_flag_power_withheld']} flagged for power, "
        f"{meta['n_flag_level_withheld']} for level); {meta['minutes']} minutes"
    )


if __name__ == "__main__":
    main()

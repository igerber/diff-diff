#!/usr/bin/env python3
"""Does pre-trend power planned from placebo event studies match what happens?

``diagnose_design(..., event_study=True)`` records, for each candidate, the
covariance of its event-study coefficients across placebo draws.
``PreTrendsPower`` turns that covariance into planned pre-trend power with
Roth's (2022) formula. Roth calibrates the formula to the covariance of an
estimated event study; using the covariance across placebo draws of a design
that has not been run is this library's extension. This script checks it:

1. Planned numbers. For each design point and candidate, the planned minimum
   detectable violation (``mdv``, target power 0.80, the default NIS pre-test)
   and the planned rejection probability with no violation, for several
   numbers ``K`` of tested pre-periods.
2. Direct check. On fresh placebo draws, add a linear trend
   ``gamma * (t - g + 1)`` to the placebo-treated units over all periods
   (``g`` the unit's adoption period, so the trend is zero in the reference
   period ``g - 1``), fit, apply the NIS test to the ``K`` pre-period
   coefficients nearest to treatment with that fit's own reported standard
   errors, and count rejections: once with ``gamma = 0`` and once per ``K``
   with ``gamma`` set to that ``K``'s planned ``mdv``. If the plan is right
   the second rate is 0.80 and the first equals the planned no-violation
   rate.
3. Monte Carlo stability. The planned ``mdv`` over 20 diagnosis seeds at 200,
   500 and 1,000 replicates.
4. Selection effect. A diagnosis compares two candidates and the one with the
   higher power at an injected effect is picked; its planned ``mdv`` on the
   comparison draws is compared with its planned ``mdv`` from a rerun alone
   on a new seed.

The formula assumes that each coefficient's own test rejects at its nominal
level when nothing is wrong. ``PreTrendsPower`` therefore reports planned
power for a set of pre-periods only when the pre-trend test over that set
rejected, in the placebo draws of the diagnosis, at the rate the formula
assumes (the "pre-test check"). So that the planned numbers of the cells the
check withholds can still be shown, this script computes them from the same
internal pieces without the check, and records the check's verdict per cell.
The question it answers is then: does the formula hold on the cells the check
lets through, and does the check catch the cells on which it does not?

Everything below was fixed before the first run and is not searched. (The
pre-test check was added after a first run of steps 1-4, which showed the
formula failing on the 40-unit points; the check adds the verdict columns
and changes nothing else.) The NIS formula is integrated by a randomized
routine, so a rerun reproduces the planned numbers to about five decimals
and the direct rates to within a draw or two; the committed JSON is the
record.

Panels: ``y_it = a_i + d_t + e_it``, ``a_i`` and ``d_t`` ``N(0, 1)``, ``e_it``
an AR(1) with coefficient ``rho`` and innovation sd 1 started from its
stationary distribution. 12 periods; ``rho`` in {0, 0.5, 0.8}; 40 and 200
units; treated share 0.25; units resampled in every draw (the default).
Common adoption at period 9, and a staggered design with adoption at 8 and
10. 12 design points, one fixed panel each.

Candidates: ``CallawaySantAnna(base_period="universal")`` on all 12 points;
``StackedDiD(kappa_pre=5, kappa_post=3)`` on the 6 common-adoption points
only (in a staggered design its default comparison group uses later cohorts
as controls, and those would receive the injected trend too, so the injected
alternative would not be the one the formula evaluates). ``K`` = 2, 4 and all
testable pre-periods for ``CallawaySantAnna``; 2 and 4 for ``StackedDiD``.

Steps 1-2 use 500 diagnosis replicates and 1,000 direct draws. Step 3 runs on
the two common-adoption ``rho = 0.5`` points, ``CallawaySantAnna`` only.
Step 4 runs on the two staggered ``rho = 0.5`` points with
``CallawaySantAnna(base_period="universal")`` under
``control_group="never_treated"`` and ``"not_yet_treated"`` (two candidates
whose estimates differ, so either can be picked), 50 repetitions of 500
replicates, the injected effect set to twice the no-effect spread of the
first candidate in a 200-replicate pilot.

Flag rules (set in advance; they flag, they hide nothing). A ``(run, K)``
cell is flagged when the direct rejection rate at the planned ``mdv`` differs
from 0.80 by more than 5 percentage points plus two Monte Carlo standard
errors of that rate, or when the direct no-violation rate differs from the
planned one by more than 3 points plus two such standard errors. The
selection step is flagged when the picked candidate's planned ``mdv`` on the
comparison draws is on average more than 3% below its rerun value.

The output JSON underwrites the "Placebo event study" notes in
``docs/methodology/REGISTRY.md`` section ``DesignDiagnosis``.

Usage:
    python benchmarks/python/pretrend_planning_check.py \\
        --output benchmarks/data/pretrend_planning_check.json
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
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diff_diff import CallawaySantAnna, PreTrendsPower, StackedDiD, diagnose_design  # noqa: E402
from diff_diff import design as _design  # noqa: E402

SEED = 20261126
N_PERIODS = 12
TREATED_SHARE = 0.25
ALPHA = 0.05
TARGET = 0.80
N_DIAGNOSIS = 500
N_DIRECT = 1000
RHOS = (0.0, 0.5, 0.8)
UNITS = (40, 200)
DESIGNS = {"common": [9], "staggered": [8, 10]}
STABILITY_SEEDS = 20
STABILITY_SIZES = (200, 500, 1000)
SELECTION_REPS = 50
BAND_POWER = 0.05
BAND_NULL = 0.03
BAND_SELECTION = 0.03


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


def pre_labels(surface):
    keep = (~surface.is_reference) & (surface.event_time < 0)
    return sorted(surface.event_time[keep].tolist())


def k_values(label, n_pre):
    if label == "StackedDiD":
        return [2, 4]
    return [2, 4, n_pre]


def diagnose(panel, estimators, cohorts, n_simulations, seed, event_study=True, effect_sizes=None):
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
            event_study=event_study,
            effect_sizes=effect_sizes,
        )


def planned(surface, tested):
    """Planned MDV and no-violation rate for ``tested``, with the check's verdict.

    The numbers are computed whether or not the check passes (``fit`` would
    refuse the cells on which it fails).
    """
    planner = PreTrendsPower(alpha=ALPHA, power=TARGET)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, vcov, n_pre, relative_times, _ = planner._extract_pre_period_params(surface, tested)
    weights = planner._get_violation_weights(n_pre, relative_times=relative_times)
    counted, expected, _, agrees = planner._planned_pretest_check(surface, tested, weights, vcov)
    return float(planner._compute_mdv(weights, vcov)), float(expected), float(counted), bool(agrees)


def score_event_study(surface, required, tested_sets, critical):
    """Rejection indicators of the NIS test on one fitted event study, or ``None``.

    The draw is scored only if every label in ``required`` (all the planned
    pre-periods) has a finite coefficient-to-standard-error ratio, so that
    every tested set of a run is scored on the same draws and a draw of
    another layout (a cohort that received no treated unit) is never counted
    for some sets and dropped for others. Returns, per tested set, whether
    any ratio exceeds ``critical`` and whether any reported p-value is below
    ``ALPHA``.
    """
    position = {label: i for i, label in enumerate(surface.event_time.tolist())}
    if any(label not in position for label in required):
        return None
    rows = [position[label] for label in required]
    if not np.all(np.isfinite(surface.att[rows] / surface.se[rows])):
        return None
    indicators = []
    for tested in tested_sets:
        rows = [position[label] for label in tested]
        ratio = np.abs(surface.att[rows] / surface.se[rows])
        indicators.append(
            (bool(np.any(ratio > critical)), bool(np.any(surface.p_value[rows] < ALPHA)))
        )
    return indicators


def direct_rejections(panel, estimator, cohorts, gammas, tested_sets, seeds, required):
    """Rejection counts of the NIS test on fresh draws, per injected trend.

    ``gammas[j]`` is injected and scored on ``tested_sets[j]`` (a list of
    lists of event times: the no-trend entry is scored on every set).
    ``required`` is the full list of planned pre-periods (see
    ``score_event_study``). Returns the counts by the normal critical value,
    the counts by the fits' reported p-values, and the number of draws
    scored, per injected trend.
    """
    n_units = panel["id"].nunique()
    n_treated = int(n_units * TREATED_SHARE)
    generator = _design._make_placebo_generator(panel, "y", "id", "t", n_treated, cohorts, True)
    fit_kwargs = _design._fit_kwargs(estimator, "y", "id", "t", {})
    z = stats.norm.ppf(1.0 - ALPHA / 2.0)
    counts = [np.zeros(len(sets), dtype=int) for sets in tested_sets]
    counts_p = [np.zeros(len(sets), dtype=int) for sets in tested_sets]
    scored = np.zeros(len(gammas), dtype=int)
    for seed in seeds:
        frame = generator(treatment_effect=0.0, seed=int(seed))
        base = frame["y"].to_numpy(dtype=float)
        first = frame[_design._FIRST_TREAT_COLUMN].to_numpy(dtype=float)
        trend = np.where(first > 0, frame["t"].to_numpy(dtype=float) - first + 1.0, 0.0)
        for j, gamma in enumerate(gammas):
            frame["y"] = base + gamma * trend
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = type(estimator)(**estimator.get_params()).fit(frame, **fit_kwargs)
                    surface = fitted.aggregate("event_study")
            except (ValueError, np.linalg.LinAlgError, KeyError, RuntimeError, ZeroDivisionError):
                continue
            indicators = score_event_study(surface, required, tested_sets[j], z)
            if indicators is None:
                continue
            scored[j] += 1
            for s, (by_critical, by_p_value) in enumerate(indicators):
                counts[j][s] += int(by_critical)
                counts_p[j][s] += int(by_p_value)
    return counts, counts_p, scored


def run_point(task):
    """Steps 1 and 2 for one design point."""
    index, rho, n_units, design, panel_seed, diag_seed, direct_seeds = task
    cohorts = DESIGNS[design]
    panel = make_panel(n_units, rho, np.random.default_rng(panel_seed))
    estimators = candidates(design)
    started = time.perf_counter()
    diagnose(panel, estimators, cohorts, N_DIAGNOSIS, diag_seed, event_study=False)
    seconds_off = time.perf_counter() - started
    started = time.perf_counter()
    diagnosis = diagnose(panel, estimators, cohorts, N_DIAGNOSIS, diag_seed)
    seconds_on = time.perf_counter() - started
    runs = []
    for label, estimator in estimators.items():
        surface = diagnosis.event_study_results.get(label)
        if surface is None:
            runs.append({"candidate": label, "recorded": False})
            continue
        labels = pre_labels(surface)
        checks = diagnosis._event_study_checks[label]
        ks = k_values(label, len(labels))
        tested = [labels[-k:] for k in ks]
        plans = [planned(surface, t) for t in tested]
        gammas = [0.0] + [plan[0] for plan in plans]
        tested_sets = [tested] + [[t] for t in tested]
        counts, counts_p, scored = direct_rejections(
            panel, estimator, cohorts, gammas, tested_sets, direct_seeds, labels
        )
        cells = []
        for j, k in enumerate(ks):
            mdv, planned_null, counted_null, check_passed = plans[j]
            n_null = int(scored[0])
            n_alt = int(scored[j + 1])
            direct_null = counts[0][j] / n_null if n_null else float("nan")
            direct_alt = counts[j + 1][0] / n_alt if n_alt else float("nan")
            se_null = np.sqrt(direct_null * (1 - direct_null) / n_null) if n_null else float("nan")
            se_alt = np.sqrt(direct_alt * (1 - direct_alt) / n_alt) if n_alt else float("nan")
            flag_power = bool(abs(direct_alt - TARGET) > BAND_POWER + 2 * se_alt)
            flag_null = bool(abs(direct_null - planned_null) > BAND_NULL + 2 * se_null)
            cells.append(
                {
                    "k": int(k),
                    "check_passed": check_passed,
                    "placebo_rejection_rate": round(counted_null, 4),
                    "planned_mdv": round(mdv, 6),
                    "planned_no_violation": round(planned_null, 4),
                    "direct_no_violation": round(float(direct_null), 4),
                    "direct_no_violation_pvalue": round(counts_p[0][j] / n_null, 4),
                    "direct_power_at_mdv": round(float(direct_alt), 4),
                    "direct_power_at_mdv_pvalue": round(counts_p[j + 1][0] / n_alt, 4),
                    "n_direct_null": n_null,
                    "n_direct_mdv": n_alt,
                    "flag_power": flag_power,
                    "flag_no_violation": flag_null,
                }
            )
        ratios = list(checks["se_ratio"].values())
        runs.append(
            {
                "candidate": label,
                "recorded": True,
                "n_draws": checks["n_draws"],
                "n_excluded": checks["n_excluded"],
                "n_pre": len(labels),
                "se_ratio_min": round(min(ratios), 4),
                "se_ratio_max": round(max(ratios), 4),
                "size": round(float(diagnosis.power_results[label][0].rejection_rate), 4),
                "pretest_counted": round(checks["pretest_rejection_rate"], 4),
                "pretest_expected": round(checks["pretest_expected_rate"], 4),
                "pretest_agrees": bool(checks["pretest_agrees"]),
                "cells": cells,
            }
        )
    return {
        "index": index,
        "rho": rho,
        "n_units": n_units,
        "design": design,
        "seconds_event_study_off": round(seconds_off, 2),
        "seconds_event_study_on": round(seconds_on, 2),
        "runs": runs,
    }


def run_stability(task):
    """Step 3: the planned mdv over diagnosis seeds, one (point, size) per task."""
    n_units, n_simulations, panel_seed, seeds = task
    panel = make_panel(n_units, 0.5, np.random.default_rng(panel_seed))
    estimators = {"CallawaySantAnna": CallawaySantAnna(base_period="universal")}
    values = []
    for seed in seeds:
        diagnosis = diagnose(panel, estimators, DESIGNS["common"], n_simulations, int(seed))
        surface = diagnosis.event_study_results.get("CallawaySantAnna")
        if surface is None:
            values.append(float("nan"))
            continue
        values.append(planned(surface, pre_labels(surface))[0])
    values = np.asarray(values)
    return {
        "n_units": n_units,
        "n_simulations": n_simulations,
        "n_seeds": len(seeds),
        "mdv_mean": round(float(np.nanmean(values)), 6),
        "mdv_sd": round(float(np.nanstd(values, ddof=1)), 6),
        "relative_spread": round(float(np.nanstd(values, ddof=1) / np.nanmean(values)), 4),
        "n_unreported": int(np.sum(~np.isfinite(values))),
    }


def selection_candidates():
    return {
        "never_treated": CallawaySantAnna(base_period="universal", control_group="never_treated"),
        "not_yet_treated": CallawaySantAnna(
            base_period="universal", control_group="not_yet_treated"
        ),
    }


def run_selection(task):
    """Step 4: one repetition of pick-then-rerun on one staggered point."""
    n_units, panel_seed, effect, compare_seed, rerun_seed = task
    cohorts = DESIGNS["staggered"]
    panel = make_panel(n_units, 0.5, np.random.default_rng(panel_seed))
    estimators = selection_candidates()
    comparison = diagnose(
        panel, estimators, cohorts, N_DIAGNOSIS, int(compare_seed), effect_sizes=[effect]
    )
    powers = {label: float(comparison.power_results[label][1].power) for label in estimators}
    picked = max(estimators, key=lambda label: powers[label])
    surface = comparison.event_study_results.get(picked)
    rerun = diagnose(panel, {picked: estimators[picked]}, cohorts, N_DIAGNOSIS, int(rerun_seed))
    rerun_surface = rerun.event_study_results.get(picked)
    if surface is None or rerun_surface is None:
        return {"n_units": n_units, "picked": picked, "relative_difference": float("nan")}
    mdv_comparison = planned(surface, pre_labels(surface))[0]
    mdv_rerun = planned(rerun_surface, pre_labels(rerun_surface))[0]
    return {
        "n_units": n_units,
        "picked": picked,
        "relative_difference": (mdv_comparison - mdv_rerun) / mdv_rerun,
    }


def selection_effect_size(n_units, panel_seed, pilot_seed):
    panel = make_panel(n_units, 0.5, np.random.default_rng(panel_seed))
    pilot = diagnose(
        panel,
        {"never_treated": selection_candidates()["never_treated"]},
        DESIGNS["staggered"],
        200,
        int(pilot_seed),
        event_study=False,
    )
    return 2.0 * float(pilot.power_results["never_treated"][0].std_estimate)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="time 20 replicates on the largest design point and exit",
    )
    args = parser.parse_args()

    root = np.random.SeedSequence(SEED)
    panel_children, diag_children, direct_children, stability_child, selection_child = root.spawn(5)
    grid = [(rho, n, design) for rho in RHOS for n in UNITS for design in DESIGNS]
    panel_seeds = panel_children.spawn(len(grid))
    diag_seeds = diag_children.generate_state(len(grid))
    direct_seed_seqs = direct_children.spawn(len(grid))

    if args.timing_only:
        panel = make_panel(200, 0.5, np.random.default_rng(panel_seeds[0]))
        started = time.perf_counter()
        diagnose(panel, candidates("common"), DESIGNS["common"], 20, 1)
        per_fit = (time.perf_counter() - started) / 40
        print(
            f"about {per_fit * 1e3:.1f} ms per fit at 200 units; 393,000 fits = "
            f"{393000 * per_fit / 60:.0f} core-minutes"
        )
        return

    started = time.perf_counter()
    point_tasks = [
        (
            i,
            rho,
            n,
            design,
            panel_seeds[i],
            int(diag_seeds[i]),
            direct_seed_seqs[i].generate_state(N_DIRECT).tolist(),
        )
        for i, (rho, n, design) in enumerate(grid)
    ]
    point_index = {(rho, n, design): i for i, (rho, n, design) in enumerate(grid)}

    stability_tasks = []
    stability_seeds = stability_child.spawn(len(UNITS) * len(STABILITY_SIZES))
    position = 0
    for n in UNITS:
        for size in STABILITY_SIZES:
            stability_tasks.append(
                (
                    n,
                    size,
                    panel_seeds[point_index[(0.5, n, "common")]],
                    stability_seeds[position].generate_state(STABILITY_SEEDS).tolist(),
                )
            )
            position += 1

    selection_tasks = []
    selection_effects = {}
    selection_seeds = selection_child.spawn(len(UNITS))
    for u, n in enumerate(UNITS):
        panel_seed = panel_seeds[point_index[(0.5, n, "staggered")]]
        state = selection_seeds[u].generate_state(1 + 2 * SELECTION_REPS).tolist()
        effect = selection_effect_size(n, panel_seed, state[0])
        selection_effects[n] = effect
        for r in range(SELECTION_REPS):
            selection_tasks.append((n, panel_seed, effect, state[1 + 2 * r], state[2 + 2 * r]))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        points = list(pool.map(run_point, point_tasks))
        stability = list(pool.map(run_stability, stability_tasks))
        selection_raw = list(pool.map(run_selection, selection_tasks, chunksize=2))

    selection = []
    for n in UNITS:
        rows = [row for row in selection_raw if row["n_units"] == n]
        diffs = np.asarray([row["relative_difference"] for row in rows], dtype=float)
        picks = {}
        for row in rows:
            picks[row["picked"]] = picks.get(row["picked"], 0) + 1
        mean = float(np.nanmean(diffs))
        selection.append(
            {
                "n_units": n,
                "injected_effect": round(selection_effects[n], 6),
                "repetitions": len(rows),
                "picked": picks,
                "measurable": bool(max(picks.values()) / len(rows) <= 0.9),
                "relative_difference_mean": round(mean, 4),
                "relative_difference_sd": round(float(np.nanstd(diffs, ddof=1)), 4),
                "relative_difference_se": round(
                    float(np.nanstd(diffs, ddof=1) / np.sqrt(np.sum(np.isfinite(diffs)))), 4
                ),
                "flag": bool(mean < -BAND_SELECTION),
            }
        )

    n_cells = sum(len(run.get("cells", [])) for point in points for run in point["runs"])
    all_cells = [cell for point in points for run in point["runs"] for cell in run.get("cells", [])]
    passed_cells = [cell for cell in all_cells if cell["check_passed"]]
    failed_cells = [cell for cell in all_cells if not cell["check_passed"]]

    def _count(cells, key):
        return int(sum(cell[key] for cell in cells))

    n_flag_power = _count(all_cells, "flag_power")
    n_flag_null = _count(all_cells, "flag_no_violation")
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
            "band_no_violation": BAND_NULL,
            "band_selection": BAND_SELECTION,
            "n_cells": n_cells,
            "n_cells_check_passed": len(passed_cells),
            "n_cells_check_failed": len(failed_cells),
            "n_flag_power_check_passed": _count(passed_cells, "flag_power"),
            "n_flag_no_violation_check_passed": _count(passed_cells, "flag_no_violation"),
            "n_flag_power_check_failed": _count(failed_cells, "flag_power"),
            "n_flag_no_violation_check_failed": _count(failed_cells, "flag_no_violation"),
            "n_flag_power": int(n_flag_power),
            "n_flag_no_violation": int(n_flag_null),
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
            "workers": args.workers,
        },
        "design_points": points,
        "stability": stability,
        "selection": selection,
    }
    lines = ["{", f' "meta": {json.dumps(payload["meta"], sort_keys=True)},', ' "design_points": [']
    rows = [json.dumps(record, sort_keys=True) for record in payload["design_points"]]
    lines.append(",\n".join("  " + row for row in rows))
    lines.append(" ],")
    lines.append(f' "stability": {json.dumps(payload["stability"], sort_keys=True)},')
    lines.append(f' "selection": {json.dumps(payload["selection"], sort_keys=True)}')
    lines.append("}")
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(
        f"wrote {args.output}: {n_cells} cells, {n_flag_power} power flags, "
        f"{n_flag_null} no-violation flags, {payload['meta']['minutes']} minutes"
    )


if __name__ == "__main__":
    main()

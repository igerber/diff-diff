#!/usr/bin/env python3
"""Simulation study for the design lock's rerun (``DesignDiagnosisResults.lock``).

Question: after several estimators are compared with ``diagnose_design`` and
one is chosen, how far is the chosen estimator's quoted power from its
reference power, and does holding the last periods out of the comparison
change that?

Per simulated panel, three runs:

  A  comparison on the frame without its last ``k`` periods (the "held-out"
     design that was evaluated and not adopted), seed ``s``
  B  comparison on the whole frame, seed ``s``
  F  every candidate on the whole frame at the lock's derived seed; ``F[e]``
     is what ``lock()`` reports for candidate ``e`` after a rerun

Choice in A and in B: the highest power among candidates that pass the size
check; ties go to the earlier label; no choice if none passes. A candidate's
reference power (and size) is the mean of its F value over all panels, which
no choice touched. Per arm the script reports the mean of
``F[chosen] - reference[chosen]`` (what a rerun quotes) and of
``comparison[chosen] - reference[chosen]`` (what the comparison itself
quotes), over the panels with a choice in that arm, and the paired
held-out-minus-same-panel difference over panels with a choice in both.

Panel ``m`` (seed ``10_000 + m``): ``y_it = a_i + trend_t + e_it`` with
``a_i ~ N(0, 1)``, ``trend = linspace(0, 1, T)`` and ``e`` a stationary AR(1)
with unit variance. Comparison seed ``s = 500_000 + m``.

In the specification-search scenario each candidate is its own
``diagnose_design`` call with one estimator, so ``lock()`` makes no rerun
there; that scenario's rerun numbers show what a rerun would give.

The output JSON underwrites ``Note (lock rerun)`` and ``Note (size verdict
near the threshold)`` in ``docs/methodology/REGISTRY.md`` section
``DesignDiagnosis``, and ``tests/test_methodology_design.py`` reads it.

Usage:
    # Full run (about 2 hours on 12 cores, Rust backend)
    python benchmarks/python/design_lock_rerun.py \\
        --output benchmarks/data/design_lock_rerun.json

    # One scenario, fewer panels
    python benchmarks/python/design_lock_rerun.py \\
        --scenarios S1_typical --n-panels 40 --output /tmp/s1.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import diff_diff  # noqa: E402
from diff_diff import (  # noqa: E402
    CallawaySantAnna,
    DifferenceInDifferences,
    EfficientDiD,
    ImputationDiD,
    StackedDiD,
    SunAbraham,
    TwoStageDiD,
    TwoWayFixedEffects,
    diagnose_design,
)

N_COVARIATES = 10
N_LOCK_CHECKS = 12  # panels per scenario on which lock() is compared with F

_WINDOW_CANDIDATES = {
    "twfe": lambda: TwoWayFixedEffects(),
    "twfe_classical": lambda: TwoWayFixedEffects(vcov_type="classical"),
    "did_cl": lambda: DifferenceInDifferences(cluster="unit"),
    "cs": lambda: CallawaySantAnna(),
    "imp": lambda: ImputationDiD(),
}

SCENARIOS = {
    # A handful of fixed estimators, common adoption.
    "S1_typical": dict(
        n_units=40,
        n_periods=12,
        rho=0.5,
        cohort_periods=[9],
        n_treated=20,
        delta=0.4,
        n_holdout=2,
        n_simulations=200,
        candidates={
            "did_cl": lambda: DifferenceInDifferences(cluster="unit"),
            "twfe": lambda: TwoWayFixedEffects(),
            "cs": lambda: CallawaySantAnna(),
            "imp": lambda: ImputationDiD(),
        },
    ),
    # Six staggered estimators, several nearly tied.
    "S2_staggered": dict(
        n_units=40,
        n_periods=12,
        rho=0.5,
        cohort_periods=[7, 9],
        n_treated=20,
        delta=0.4,
        n_holdout=2,
        n_simulations=200,
        candidates={
            "cs": lambda: CallawaySantAnna(),
            "sa": lambda: SunAbraham(),
            "imp": lambda: ImputationDiD(),
            "ts": lambda: TwoStageDiD(),
            "stk": lambda: StackedDiD(),
            "eff": lambda: EfficientDiD(),
        },
    ),
    # A long post window and a candidate whose over-rejection grows with it.
    "S3_window": dict(
        n_units=40,
        n_periods=20,
        rho=0.25,
        cohort_periods=[13],
        n_treated=20,
        delta=0.3,
        n_holdout=4,
        n_simulations=200,
        candidates=_WINDOW_CANDIDATES,
    ),
    # Specification search: ten covariate choices, all pure noise, one
    # diagnose_design call per choice.
    "S4_specification_search": dict(
        n_units=24,
        n_periods=12,
        rho=0.5,
        cohort_periods=[9],
        n_treated=8,
        delta=0.6,
        n_holdout=2,
        n_simulations=200,
        candidates={f"x{k}": k for k in range(N_COVARIATES)},
        specification_search=True,
    ),
    # The default replicate count, with a candidate just above the size threshold.
    "S5_default_replicates": dict(
        n_units=40,
        n_periods=20,
        rho=0.11,
        cohort_periods=[13],
        n_treated=20,
        delta=0.3,
        n_holdout=4,
        n_simulations=500,
        candidates=_WINDOW_CANDIDATES,
    ),
}


def _ar1(rng, n_units, n_periods, rho):
    noise = np.empty((n_units, n_periods))
    noise[:, 0] = rng.normal(size=n_units)
    for period in range(1, n_periods):
        noise[:, period] = rho * noise[:, period - 1] + np.sqrt(1 - rho**2) * rng.normal(
            size=n_units
        )
    return noise


def make_panel(cfg, seed):
    rng = np.random.default_rng(seed)
    n_units, n_periods = cfg["n_units"], cfg["n_periods"]
    y = (
        rng.normal(size=n_units)[:, None]
        + np.linspace(0, 1, n_periods)[None, :]
        + _ar1(rng, n_units, n_periods, cfg["rho"])
    )
    frame = pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), n_periods),
            "period": np.tile(np.arange(1, n_periods + 1), n_units),
            "y": y.ravel(),
        }
    )
    if cfg.get("specification_search"):
        for k in range(N_COVARIATES):
            frame[f"x{k}"] = _ar1(rng, n_units, n_periods, 0.5).ravel()
    return frame


def _rows(result):
    out = {}
    for record in result.to_dataframe().to_dict("records"):
        slot = out.setdefault(record["label"], {"size_controlled": bool(record["size_controlled"])})
        slot["size" if record["true_effect"] == 0 else "power"] = float(record["power"])
    return out


def compare(cfg, frame, seed):
    """Every candidate on ``frame``: per-label size, power and size verdict."""
    common = dict(
        cohort_periods=cfg["cohort_periods"],
        n_treated=cfg["n_treated"],
        effect_sizes=[cfg["delta"]],
        n_simulations=cfg["n_simulations"],
        seed=seed,
        progress=False,
    )
    if cfg.get("specification_search"):
        table = {}
        for label, k in cfg["candidates"].items():
            result = diagnose_design(
                frame, "y", "unit", "period", {label: TwoWayFixedEffects()},
                estimator_kwargs={"covariates": [f"x{k}"]}, **common,
            )  # fmt: skip
            table.update(_rows(result))
        return table, None
    estimators = {label: build() for label, build in cfg["candidates"].items()}
    result = diagnose_design(frame, "y", "unit", "period", estimators, **common)
    return _rows(result), result


def choose(table):
    passing = [
        (row["power"], -position, label)
        for position, (label, row) in enumerate(table.items())
        if row["size_controlled"]
    ]
    return max(passing)[2] if passing else None


def one_panel(args):
    name, m = args
    warnings.simplefilter("ignore")
    cfg = SCENARIOS[name]
    panel = make_panel(cfg, 10_000 + m)
    seed = 500_000 + m
    lock_seed = int(np.random.SeedSequence([seed, 1]).generate_state(1)[0])
    shortened = panel[panel["period"] <= cfg["n_periods"] - cfg["n_holdout"]]
    held_out, _ = compare(cfg, shortened, seed)
    same_panel, diagnosis = compare(cfg, panel, seed)
    rerun, _ = compare(cfg, panel, lock_seed)
    record = dict(
        held_out=held_out,
        same_panel=same_panel,
        rerun=rerun,
        pick_held_out=choose(held_out),
        pick_same_panel=choose(same_panel),
    )
    chosen = record["pick_same_panel"]
    if diagnosis is not None and chosen is not None and m < N_LOCK_CHECKS:
        lock = diagnosis.lock(chosen, data=panel, progress=False)
        got = {
            ("size" if row["true_effect"] == 0 else "power"): row["power"]
            for row in lock.rerun_results
        }
        record["lock_matches_rerun"] = bool(
            abs(got["power"] - rerun[chosen]["power"]) < 1e-12
            and abs(got["size"] - rerun[chosen]["size"]) < 1e-12
            and lock.size_controlled == rerun[chosen]["size_controlled"]
        )
    return record


def _arm(records, pick_key, comparison_key, reference_power, reference_size, labels):
    used = [r for r in records if r[pick_key] is not None]
    rerun_gap = np.array(
        [r["rerun"][r[pick_key]]["power"] - reference_power[r[pick_key]] for r in used]
    )
    comparison_gap = np.array(
        [r[comparison_key][r[pick_key]]["power"] - reference_power[r[pick_key]] for r in used]
    )
    picks = {label: int(sum(r[pick_key] == label for r in used)) for label in labels}
    fails = {
        label: float(
            np.mean(
                [not r["rerun"][label]["size_controlled"] for r in used if r[pick_key] == label]
            )
        )
        for label in labels
        if picks[label]
    }
    return dict(
        n_chosen=len(used),
        rerun_minus_reference=float(rerun_gap.mean()),
        rerun_minus_reference_se=float(rerun_gap.std(ddof=1) / np.sqrt(len(used))),
        comparison_minus_reference=float(comparison_gap.mean()),
        comparison_minus_reference_se=float(comparison_gap.std(ddof=1) / np.sqrt(len(used))),
        reference_power_of_chosen=float(np.mean([reference_power[r[pick_key]] for r in used])),
        reference_size_of_chosen=float(np.mean([reference_size[r[pick_key]] for r in used])),
        rerun_size_fail=float(
            np.mean([not r["rerun"][r[pick_key]]["size_controlled"] for r in used])
        ),
        picks=picks,
        rerun_size_fail_by_candidate=fails,
    )


def summarize(name, records):
    cfg = SCENARIOS[name]
    labels = list(cfg["candidates"])
    reference_power = {k: float(np.mean([r["rerun"][k]["power"] for r in records])) for k in labels}
    reference_size = {k: float(np.mean([r["rerun"][k]["size"] for r in records])) for k in labels}
    both = [
        r for r in records if r["pick_held_out"] is not None and r["pick_same_panel"] is not None
    ]
    paired = np.array(
        [
            (r["rerun"][r["pick_held_out"]]["power"] - reference_power[r["pick_held_out"]])
            - (r["rerun"][r["pick_same_panel"]]["power"] - reference_power[r["pick_same_panel"]])
            for r in both
        ]
    )
    return dict(
        n_panels=len(records),
        n_simulations=cfg["n_simulations"],
        n_units=cfg["n_units"],
        n_periods=cfg["n_periods"],
        rho=cfg["rho"],
        cohort_periods=cfg["cohort_periods"],
        n_treated=cfg["n_treated"],
        effect_size=cfg["delta"],
        n_holdout=cfg["n_holdout"],
        reference_power=reference_power,
        reference_size=reference_size,
        held_out=_arm(
            records, "pick_held_out", "held_out", reference_power, reference_size, labels
        ),
        same_panel=_arm(
            records, "pick_same_panel", "same_panel", reference_power, reference_size, labels
        ),
        paired_difference=float(paired.mean()),
        paired_difference_se=float(paired.std(ddof=1) / np.sqrt(len(paired))),
        same_choice_share=float(
            np.mean([r["pick_held_out"] == r["pick_same_panel"] for r in both])
        ),
        lock_matches_rerun=[r["lock_matches_rerun"] for r in records if "lock_matches_rerun" in r],
    )


def main():
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="Path of the summary JSON to write.")
    parser.add_argument("--n-panels", type=int, default=400)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--processes", type=int, default=12)
    args = parser.parse_args()

    scenarios = {}
    with Pool(args.processes) as pool:
        for name in args.scenarios:
            started = time.time()
            records = pool.map(one_panel, [(name, m) for m in range(args.n_panels)], chunksize=1)
            scenarios[name] = summarize(name, records)
            print(f"{name}: {time.time() - started:.0f} s", flush=True)
            payload = dict(
                meta=dict(
                    script="benchmarks/python/design_lock_rerun.py",
                    diff_diff_version=diff_diff.__version__,
                    n_panels=args.n_panels,
                    panel_seed="10_000 + m",
                    comparison_seed="500_000 + m",
                    rerun_seed="SeedSequence([comparison_seed, 1])",
                    choice_rule="highest power among candidates passing the size check",
                ),
                scenarios=scenarios,
            )
            # Written after every scenario so a stopped run keeps what finished.
            Path(args.output).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

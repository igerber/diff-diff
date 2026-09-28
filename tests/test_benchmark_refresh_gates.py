"""
Regression tests for the 2026-07 benchmark-refresh publication gates.

These lock the fail-closed behavior of the refresh harness in
benchmarks/refresh_2026_07/: out-of-tolerance headline SE parity must gate
even when confidence intervals overlap, BOTH rendered Python arms must
independently satisfy the tolerances vs R, and the table generator must
refuse any results payload carrying hard correctness flags.

No R, venvs, or subprocesses involved - pure unit tests on the gate logic.
"""

import sys
from pathlib import Path

import pytest

# The refresh harness is repo tooling, not wheel content: CI's Python-test
# jobs copy tests/ to a temp directory and run against the installed wheel,
# where benchmarks/ does not exist. Skip the whole module there (same
# convention as the benchmarks/data golden-file skips).
REFRESH_DIR = Path(__file__).parent.parent / "benchmarks" / "refresh_2026_07"
if not (REFRESH_DIR / "refresh_common.py").exists():
    pytest.skip(
        "benchmark refresh harness not present (installed-wheel test layout)",
        allow_module_level=True,
    )
sys.path.insert(0, str(REFRESH_DIR))

import gen_benchmark_tables as gen  # noqa: E402
import refresh_common as rc  # noqa: E402


class TestHardFlagClassification:
    def test_all_hard_tokens_detected(self):
        flags = [
            "parity_fail",
            "mpdta_known_answer_fail:r:-0.05",
            "headline_att_gate_fail:python_pure:1e-2",
            "headline_se_gate_fail:python_rust:0.500",
            "ci_gate_fail",
            "sdid_weights_gate_fail:unit_weights:python_rust:0.01",
            "detail_keys_mismatch:event_study:python_pure",
            "sdid_att_gate_fail:1e-6",
            "pure_rust_att_gate_fail:2e-7",
        ]
        assert rc.hard_flags(flags) == flags

    def test_soft_flags_ignored(self):
        soft = ["python_rust:cv_flag:0.150"]
        assert rc.hard_flags(soft) == []

    def test_rep_att_spread_is_hard(self):
        # Non-deterministic point estimates across seeded replications must
        # block publication (rep_att_gate_fail carries the hard token).
        assert rc.hard_flags(["python_rust:rep_att_gate_fail:1.0e-09"])


class TestHeadlineGates:
    """Strict ATT/SE gates - CI overlap must NOT be an escape hatch."""

    def test_se_mismatch_gates_despite_ci_overlap(self):
        # ATT identical, SE 30% off with wide CIs that overlap heavily:
        # compare_estimates() would report passed=True via ci_overlap; the
        # headline gate must still fail the SE.
        flags = rc.headline_gate_flags(
            "python_rust",
            py_att=2.5,
            py_se=0.13,
            r_att=2.5,
            r_se=0.10,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert any(fl.startswith("headline_se_gate_fail:python_rust") for fl in flags)
        assert rc.hard_flags(flags), "SE gate flag must be a hard flag"

    def test_att_mismatch_gates(self):
        flags = rc.headline_gate_flags(
            "python_pure",
            py_att=2.60,
            py_se=0.10,
            r_att=2.50,
            r_se=0.10,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert any(fl.startswith("headline_att_gate_fail:python_pure") for fl in flags)
        assert rc.hard_flags(flags)

    def test_nonfinite_values_gate(self):
        flags = rc.headline_gate_flags(
            "python_rust",
            py_att=float("nan"),
            py_se=0.1,
            r_att=2.5,
            r_se=0.1,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert any("nonfinite" in fl for fl in flags)
        assert rc.hard_flags(flags)
        flags_se = rc.headline_gate_flags(
            "python_rust",
            py_att=2.5,
            py_se=float("inf"),
            r_att=2.5,
            r_se=0.1,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert any(fl.startswith("headline_se_gate_fail") for fl in flags_se)

    def test_missing_values_gate(self):
        flags = rc.headline_gate_flags(
            "python_pure",
            py_att=None,
            py_se=None,
            r_att=2.5,
            r_se=0.1,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert flags and rc.hard_flags(flags)

    def test_ci_overlap_gated_per_arm(self):
        # Pure-arm CI disjoint from R must gate even though ATT/SE checks
        # are relative: ATT far apart with tiny SEs -> no overlap.
        flags = rc.headline_gate_flags(
            "python_pure",
            py_att=10.0,
            py_se=0.01,
            r_att=2.5,
            r_se=0.01,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert any(fl == "ci_gate_fail:python_pure" for fl in flags)
        assert rc.hard_flags(["ci_gate_fail:python_pure"])

    def test_overlapping_cis_produce_no_ci_flag(self):
        flags = rc.headline_gate_flags(
            "python_rust",
            py_att=2.5,
            py_se=0.13,
            r_att=2.5,
            r_se=0.10,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert not any(fl.startswith("ci_gate_fail") for fl in flags)

    def test_within_tolerance_produces_no_flags(self):
        flags = rc.headline_gate_flags(
            "python_rust",
            py_att=2.5000000001,
            py_se=0.1000001,
            r_att=2.5,
            r_se=0.10,
            att_atol=1e-4,
            se_rtol=0.10,
        )
        assert flags == []


class TestCompareEffectArraysFailClosed:
    PY = [{"group": 3, "time": 4, "att": 1.0, "se": 0.1}]
    R = [{"group": 3, "time": 4, "att": 1.0, "se": 0.1}]

    def test_clean_arrays_pass(self):
        m = rc.compare_effect_arrays(self.PY, self.R, ["group", "time"], se_rtol=0.1)
        assert m["keys_match"] and m["att_ok"] and m["se_ok"]

    def test_duplicate_join_keys_break_keys_match(self):
        py = self.PY + [{"group": 3, "time": 4, "att": 1.0, "se": 0.1}]
        m = rc.compare_effect_arrays(py, self.R, ["group", "time"], se_rtol=0.1)
        assert not m["keys_match"]
        assert m["n_dup_python"] == 1

    def test_nonfinite_att_breaks_keys_match(self):
        py = self.PY + [{"group": 3, "time": 5, "att": float("nan"), "se": 0.1}]
        r = self.R + [{"group": 3, "time": 5, "att": float("nan"), "se": 0.1}]
        m = rc.compare_effect_arrays(py, r, ["group", "time"], se_rtol=0.1)
        assert not m["keys_match"]
        assert m["n_dropped_python"] == 1 and m["n_dropped_r"] == 1

    def test_missing_se_breaks_se_ok(self):
        py = [{"group": 3, "time": 4, "att": 1.0, "se": None}]
        m = rc.compare_effect_arrays(py, self.R, ["group", "time"], se_rtol=0.1)
        assert m["att_ok"] and not m["se_ok"]
        assert m["n_se_compared"] == 0


class TestEstimatorFieldGuard:
    """A silently ignored --type flag can never publish mislabeled numbers."""

    def test_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="estimator mismatch"):
            rc.validate_estimator_field(
                {"estimator": "diff_diff.DifferenceInDifferences"},
                "diff_diff.TwoWayFixedEffects",
            )

    def test_missing_field_raises(self):
        with pytest.raises(RuntimeError, match="estimator mismatch"):
            rc.validate_estimator_field({}, "diff_diff.CallawaySantAnna")

    def test_match_passes(self):
        rc.validate_estimator_field(
            {"estimator": "fixest::feols (absorbed FE)"},
            "fixest::feols (absorbed FE)",
        )

    def test_refresh_specs_declare_distinct_wiring(self):
        # Every headline spec must pin the expected estimator fields, and the
        # basic/twfe split must be wired to the correct scripts.
        import run_refresh

        for name, spec in run_refresh.BENCH_SPECS.items():
            assert spec.get("py_estimator"), f"{name}: missing py_estimator"
            assert spec.get("r_estimator"), f"{name}: missing r_estimator"
            assert spec.get("se_gate_rtol"), f"{name}: missing se_gate_rtol"
        basic = run_refresh.BENCH_SPECS["basic"]
        twfe = run_refresh.BENCH_SPECS["twfe"]
        assert basic["py_script"] == "benchmark_basic.py"
        assert basic["py_estimator"] == "diff_diff.DifferenceInDifferences"
        assert twfe["py_script"] == "benchmark_twfe.py"
        assert twfe["r_script"] == "benchmark_twfe.R"
        assert twfe["py_estimator"] == "diff_diff.TwoWayFixedEffects"
        assert twfe["r_estimator"] == "fixest::feols (absorbed FE)"


class TestSyntheticDiDSEGateDocumented:
    """The wider SDID SE gate is a documented Monte Carlo bound.

    Both implementations estimate the placebo variance by Monte Carlo and
    R's placebo permutation is unseeded, so SEs agree in distribution, not
    draw-by-draw - see the REGISTRY.md SyntheticDiD note "benchmark SE gate
    is Monte Carlo-bounded (2026-07)".
    """

    def test_gate_value_matches_registry_note(self):
        import run_refresh

        assert run_refresh.BENCH_SPECS["synthdid"]["se_gate_rtol"] == 0.35

    def test_within_mc_bound_passes_beyond_it_gates(self):
        ok = rc.headline_gate_flags(
            "python_rust",
            py_att=3.84,
            py_se=0.12,
            r_att=3.84,
            r_se=0.10,
            att_atol=1e-4,
            se_rtol=0.35,
        )
        assert ok == []  # 20% is within the documented MC bound
        bad = rc.headline_gate_flags(
            "python_rust",
            py_att=3.84,
            py_se=0.145,
            r_att=3.84,
            r_se=0.10,
            att_atol=1e-4,
            se_rtol=0.35,
        )
        assert any(fl.startswith("headline_se_gate_fail") for fl in bad)


class TestEnvironmentFingerprint:
    META = {
        "r_version": "R version 4.5.2",
        "r_packages": {"did": "2.5.1"},
        "hardware": {"cpu": "Apple M4 Max", "os": "macOS 15"},
        "orchestrator_python": "3.14.4",
        "protocol": "p",
        "thread_policy": "t",
    }

    def test_fingerprint_changes_with_environment(self):
        fp1 = rc.env_fingerprint(self.META, "3.7.0")
        meta2 = dict(self.META, r_packages={"did": "2.6.0"})
        assert fp1 != rc.env_fingerprint(meta2, "3.7.0")
        assert fp1 != rc.env_fingerprint(self.META, "3.7.1")
        assert fp1 == rc.env_fingerprint(dict(self.META), "3.7.0")

    def test_fingerprint_changes_with_python_arm_provenance(self):
        py1 = {"numpy": "2.5.1", "pandas": "3.0.3", "python": "3.14.4"}
        py2 = dict(py1, numpy="2.6.0")
        fp1 = rc.env_fingerprint(self.META, "3.7.0", python_env=py1)
        assert fp1 != rc.env_fingerprint(self.META, "3.7.0", python_env=py2)
        assert fp1 == rc.env_fingerprint(self.META, "3.7.0", python_env=dict(py1))

    def test_thread_env_vars_stripped_from_child_env(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.setenv("RAYON_NUM_THREADS", "2")
        env = rc._child_env()
        assert "OMP_NUM_THREADS" not in env
        assert "RAYON_NUM_THREADS" not in env
        assert env["DIFF_DIFF_BENCH_USE_INSTALLED"] == "1"

    def test_slim_result_redacts_local_paths(self):
        slim = rc.slim_result(
            {
                "provenance": {
                    "diff_diff_path": "/Users/someone/repo/venvs/dd370/lib/"
                    "python3.14/site-packages/diff_diff/__init__.py",
                    "python_executable": "/Users/someone/repo/venvs/dd370/bin/python",
                }
            }
        )
        prov = slim["provenance"]
        assert "someone" not in prov["diff_diff_path"]
        assert prov["diff_diff_path"].startswith(".../")
        assert "someone" not in prov["python_executable"]

    def test_generator_refuses_mixed_fingerprints(self):
        payload = {
            "env_fingerprint": "abc",
            "cells": {
                "basic/small": {"flags": [], "env_fingerprint": "abc"},
                "callaway/small": {"flags": [], "env_fingerprint": "OLD"},
            },
        }
        with pytest.raises(SystemExit, match=r"different\s+environment"):
            gen.assert_uniform_environment(payload)

    def test_generator_accepts_uniform_fingerprints(self):
        payload = {
            "env_fingerprint": "abc",
            "cells": {
                "basic/small": {"flags": [], "env_fingerprint": "abc"},
            },
        }
        gen.assert_uniform_environment(payload)


class TestRunArmNonFiniteReplicationGate:
    def test_nan_on_intermediate_rep_hard_flags(self):
        # A NaN replication must gate even when the LAST rep (which feeds
        # the headline gates) is finite.
        results = iter(
            [
                {"timing": {"total_seconds": 0.01}, "att": float("nan"), "se": 0.1},
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": 0.1},
            ]
        )
        out = rc.run_arm("test/nan", lambda: next(results), 2, allow_cv_rerun=False)
        assert out["n_nonfinite_att"] == 1
        assert any(fl.startswith("rep_att_gate_fail:nonfinite") for fl in out["flags"])
        assert rc.hard_flags(out["flags"])

    def test_nan_pass_with_high_cv_is_not_masked_by_rerun(self):
        # First pass: NaN SE on rep 1 AND noisy timings (CV > 10%). The CV
        # rerun must NOT discard the correctness failure by replacing the
        # pass with a clean second run.
        results = iter(
            [
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": float("nan")},
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": 0.1},
                {"timing": {"total_seconds": 0.50}, "att": 2.5, "se": 0.1},
                # If a rerun were (wrongly) attempted, these would be consumed:
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": 0.1},
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": 0.1},
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": 0.1},
            ]
        )
        consumed = {"n": 0}

        def rep_fn():
            consumed["n"] += 1
            return next(results)

        out = rc.run_arm("test/nan+cv", rep_fn, 3, allow_cv_rerun=True)
        assert out["cv"] > rc.CV_FLAG  # the rerun trigger was genuinely armed
        assert consumed["n"] == 3, "rerun must be skipped when correctness flags exist"
        assert any(fl.startswith("rep_se_gate_fail:nonfinite") for fl in out["flags"])
        assert rc.hard_flags(out["flags"])

    def test_all_finite_reps_produce_no_nonfinite_flags(self):
        results = iter(
            [
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": 0.1},
                {"timing": {"total_seconds": 0.01}, "att": 2.5, "se": 0.1},
            ]
        )
        out = rc.run_arm("test/ok", lambda: next(results), 2, allow_cv_rerun=False)
        assert out["n_nonfinite_att"] == 0 and out["n_nonfinite_se"] == 0
        assert not rc.hard_flags(out["flags"])


class TestRunnerExitScansWholeArtifact:
    def test_stale_flagged_cells_fail_the_run(self):
        # merge-on-write can preserve older hard-flagged cells during --only
        # reruns; the exit status must reflect the whole artifact.
        import run_refresh

        payload = {
            "cells": {
                "basic/small": {"flags": []},  # fresh, clean
                "synthdid/5k": {"flags": ["headline_se_gate_fail:python_pure:0.9"]},
            }
        }
        failures = run_refresh.collect_hard_failures(payload)
        assert failures == ["synthdid/5k: ['headline_se_gate_fail:python_pure:0.9']"]

    def test_clean_artifact_no_failures(self):
        import run_refresh

        assert run_refresh.collect_hard_failures({"cells": {"a/b": {"flags": []}}}) == []


class TestWeightVectorComparison:
    def test_identical_weights_pass(self):
        m = rc.compare_weight_vectors([0.5, 0.5, 0.0], [0.5, 0.5, 0.0])
        assert m["ok"] and m["max_abs_diff"] == 0.0

    def test_out_of_order_weights_align_by_ids(self):
        # Python's get_unit_weights_df() sorts by DESCENDING WEIGHT while R
        # emits panel order - identical weights must compare equal once
        # aligned by unit id (a positional comparison would fabricate a
        # 0.4 max diff here).
        py_w, py_ids = [0.5, 0.3, 0.2], [7, 2, 5]  # weight-sorted
        r_w, r_ids = [0.3, 0.2, 0.5], [2, 5, 7]  # panel order
        m = rc.compare_weight_vectors(py_w, r_w, py_ids=py_ids, r_ids=r_ids)
        assert m["aligned_by_ids"] and m["ok"] and m["max_abs_diff"] == 0.0

    def test_id_alignment_catches_true_divergence(self):
        m = rc.compare_weight_vectors(
            [0.5, 0.3, 0.2], [0.3, 0.3, 0.4], py_ids=[7, 2, 5], r_ids=[2, 5, 7]
        )
        assert m["aligned_by_ids"] and not m["ok"]
        assert abs(m["max_abs_diff"] - 0.1) < 1e-15

    def test_string_ids_from_r_align_with_int_ids(self):
        # R rownames are character; Python ids are ints - keys normalize.
        m = rc.compare_weight_vectors([0.6, 0.4], [0.4, 0.6], py_ids=[10, 3], r_ids=["3", "10"])
        assert m["aligned_by_ids"] and m["ok"]

    def test_missing_ids_fail_closed_when_required(self):
        # The SDID publication gate documents id alignment; positionally
        # identical weights must still FAIL if either side stops emitting
        # ids (a script regression cannot silently weaken the contract).
        m = rc.compare_weight_vectors([0.5, 0.5], [0.5, 0.5], require_ids=True)
        assert not m["ok"] and not m["aligned_by_ids"]
        m2 = rc.compare_weight_vectors(
            [0.5, 0.5], [0.5, 0.5], py_ids=[1, 2], r_ids=None, require_ids=True
        )
        assert not m2["ok"]

    def test_require_ids_passes_with_valid_ids(self):
        m = rc.compare_weight_vectors(
            [0.5, 0.5], [0.5, 0.5], py_ids=[1, 2], r_ids=[1, 2], require_ids=True
        )
        assert m["ok"] and m["aligned_by_ids"]

    def test_positional_fallback_still_allowed_when_not_required(self):
        m = rc.compare_weight_vectors([0.5, 0.5], [0.5, 0.5])
        assert m["ok"] and not m["aligned_by_ids"]

    def test_mismatched_id_sets_fail_closed(self):
        m = rc.compare_weight_vectors([0.5, 0.5], [0.5, 0.5], py_ids=[1, 2], r_ids=[1, 3])
        assert not m["ok"] and m["max_abs_diff"] is None

    def test_duplicate_ids_fail_closed(self):
        m = rc.compare_weight_vectors([0.5, 0.5], [0.5, 0.5], py_ids=[1, 1], r_ids=[1, 2])
        assert not m["ok"]

    def test_divergent_weights_fail(self):
        m = rc.compare_weight_vectors([0.5, 0.5], [0.4, 0.6])
        assert not m["ok"] and abs(m["max_abs_diff"] - 0.1) < 1e-15

    def test_length_mismatch_fails_closed(self):
        m = rc.compare_weight_vectors([0.5, 0.5], [1.0])
        assert not m["ok"] and m["max_abs_diff"] is None

    def test_nonfinite_fails_closed(self):
        m = rc.compare_weight_vectors([float("nan"), 0.5], [0.5, 0.5])
        assert not m["ok"]

    def test_empty_fails_closed(self):
        assert not rc.compare_weight_vectors([], [])["ok"]


class TestGeneratorRefusesHardGatedPayloads:
    @staticmethod
    def _payload(flags):
        return {"cells": {"callaway/small": {"flags": flags}}}

    def test_hard_flag_aborts_generation(self):
        for flag in (
            "headline_se_gate_fail:python_pure:0.300",
            "parity_fail",
            "detail_keys_mismatch:event_study:python_rust",
            "pure_rust_att_gate_fail:1e-6",
        ):
            with pytest.raises(SystemExit, match="REFUSING"):
                gen.assert_no_hard_flags(self._payload([flag]))

    def test_rep_att_spread_aborts_generation(self):
        with pytest.raises(SystemExit, match="REFUSING"):
            gen.assert_no_hard_flags(
                {"cells": {"basic/small": {"flags": ["python_rust:rep_att_gate_fail:1e-9"]}}}
            )

    def test_soft_flags_do_not_abort(self):
        gen.assert_no_hard_flags(self._payload(["python_rust:cv_flag:0.200"]))

    def test_flag_free_payload_passes(self):
        gen.assert_no_hard_flags(self._payload([]))

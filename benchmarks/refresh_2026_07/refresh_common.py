#!/usr/bin/env python3
"""
Shared machinery for the 2026-07 public benchmark refresh.

Used by run_refresh.py (headline docs/benchmarks.rst refresh: released
diff-diff wheel vs R) and run_version_story.py (internal 3.5.3-vs-3.7.0
optimization story). Both runners execute every replication in a fresh
subprocess, strictly sequentially, against pinned diff-diff wheels installed
in isolated uv venvs - never against the dev tree.

Key invariants enforced here:
- Provenance hard-fail: a Python arm whose imported diff_diff does not come
  from the expected venv/version, or whose backend does not match the arm,
  aborts the run (guards against dev-tree sys.path shadowing).
- Sequential execution: one benchmark subprocess at a time, cool-downs
  between arms, load-average guard before timed runs.
- Warm-up inside every subprocess (--warmup) so R byte-compiler JIT and
  first-call setup stay out of the timing window on BOTH sides.
"""

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REFRESH_DIR = Path(__file__).parent
BENCHMARK_DIR = REFRESH_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent
VENVS_DIR = REFRESH_DIR / "venvs"
RESULTS_DIR = REFRESH_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
DATA_DIR = BENCHMARK_DIR / "data" / "synthetic"  # gitignored (*.csv)
MPDTA_SOURCE = BENCHMARK_DIR / "data" / "real" / "mpdta.csv"  # committed

sys.path.insert(0, str(REPO_ROOT))

from benchmarks.python.utils import (  # noqa: E402
    compute_timing_stats,
    generate_basic_did_data,
    generate_multiperiod_data,
    generate_sdid_data,
    generate_staggered_data,
    save_benchmark_data,
)

UV_BIN = os.environ.get("UV_BIN", os.path.expanduser("~/.local/bin/uv"))
PYTHON_PIN = "3.14"

# Coefficient-of-variation gate (std/mean of counted reps), matching the
# CV_FLAG convention in benchmarks/speed_review/bench_fe_absorption.py.
CV_FLAG = 0.10
COOLDOWN_SECONDS = 5
LOAD_GUARD_1MIN = 2.0

# Published MPDTA overall ATT (docs/benchmarks.rst) - known-answer gate.
MPDTA_KNOWN_ATT = -0.039951

# Flag tokens that mark a cell as CORRECTNESS-failed (R-parity, known
# answers, detailed effect surfaces) - as opposed to timing-noise flags.
# Single source of truth: run_refresh.py fails the run on these and
# gen_benchmark_tables.py refuses to render payloads containing them.
# Thread-count knobs that would silently change per-arm performance while
# the artifacts claim package defaults - stripped from every benchmark
# subprocess (Python AND R) so "defaults" is enforced, not assumed.
THREAD_ENV_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "R_DATATABLE_NUM_THREADS",
)

HARD_FLAG_TOKENS = (
    "parity_fail",
    "known_answer_fail",
    "att_gate_fail",
    "se_gate_fail",
    "ci_gate_fail",
    "weights_gate_fail",
    "keys_mismatch",
)


def hard_flags(flags: List[str]) -> List[str]:
    # Subset of flags that are hard correctness gates.
    return [fl for fl in flags if any(tok in fl for tok in HARD_FLAG_TOKENS)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def env_fingerprint(
    meta: Dict[str, Any], pin: str, python_env: Optional[Dict[str, Any]] = None
) -> str:
    """
    Short hash of everything that must be CONSTANT across all cells rendered
    onto the docs page together: package versions, hardware, pin, protocol.
    Excludes timestamps/load. Publication refuses to mix fingerprints, so a
    partial --only rerun under a changed environment can never silently blend
    with older cells.
    """
    basis = {
        "pin": pin,
        "r_version": meta.get("r_version"),
        "r_packages": meta.get("r_packages"),
        "cpu": (meta.get("hardware") or {}).get("cpu"),
        "os": (meta.get("hardware") or {}).get("os"),
        "orchestrator_python": meta.get("orchestrator_python"),
        "protocol": meta.get("protocol"),
        "thread_policy": meta.get("thread_policy"),
        # Python-arm provenance (venv python/numpy/pandas + BLAS linkage):
        # a dependency upgrade between partial reruns must change the
        # fingerprint so mixed-environment cells can never blend.
        "python_env": python_env,
    }
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()[:16]


def compare_weight_vectors(
    py_weights: Any,
    r_weights: Any,
    atol: float = 1e-8,
    py_ids: Any = None,
    r_ids: Any = None,
    require_ids: bool = False,
) -> Dict[str, Any]:
    """
    Compare SDID unit/time weight vectors, aligning by unit/period id when
    both sides provide ids (ordering-robust: Python's
    get_unit_weights_df() sorts by descending weight, R emits panel order).
    With require_ids=True (the SDID publication gate), missing ids on
    either side FAILS the comparison - the documented id-alignment contract
    can never silently degrade to positional order if a benchmark script
    regresses. Without it, positional comparison is a permitted fallback
    for legacy artifacts. Fail-closed on length/key mismatch, duplicates,
    or non-finite entries; metrics are committed for audit.
    """
    py = list(py_weights or [])
    r = list(r_weights or [])
    metrics: Dict[str, Any] = {
        "n_python": len(py),
        "n_r": len(r),
        "max_abs_diff": None,
        "aligned_by_ids": False,
        "ok": False,
    }
    if require_ids and (py_ids is None or r_ids is None):
        return metrics
    if not py or len(py) != len(r):
        return metrics

    def _key(x):
        return round(float(x), 9)

    if py_ids is not None and r_ids is not None:
        py_ids = list(py_ids)
        r_ids = list(r_ids)
        if len(py_ids) != len(py) or len(r_ids) != len(r):
            return metrics
        try:
            py_map = {_key(k): w for k, w in zip(py_ids, py)}
            r_map = {_key(k): w for k, w in zip(r_ids, r)}
        except (TypeError, ValueError):
            return metrics
        if len(py_map) != len(py) or len(r_map) != len(r):
            return metrics  # duplicate ids
        if set(py_map) != set(r_map):
            return metrics  # different unit/period sets
        pairs = [(py_map[k], r_map[k]) for k in sorted(py_map)]
        metrics["aligned_by_ids"] = True
    else:
        pairs = list(zip(py, r))

    diffs = []
    for a, b in pairs:
        try:
            a_f, b_f = float(a), float(b)
        except (TypeError, ValueError):
            return metrics
        if not (math.isfinite(a_f) and math.isfinite(b_f)):
            return metrics
        diffs.append(abs(a_f - b_f))
    metrics["max_abs_diff"] = max(diffs)
    metrics["ok"] = metrics["max_abs_diff"] < atol
    return metrics


def validate_estimator_field(result: Dict[str, Any], expected: str) -> None:
    """
    Hard-fail when a benchmark subprocess ran a different estimator than the
    cell spec claims (e.g. a silently ignored --type flag). The emitted
    "estimator" field is distinct per script/model, so miswiring can never
    publish numbers under the wrong label.
    """
    actual = result.get("estimator")
    if actual != expected:
        raise RuntimeError(
            f"estimator mismatch: spec expects {expected!r}, "
            f"benchmark subprocess reported {actual!r}"
        )


def headline_gate_flags(
    arm_name: str,
    py_att: Any,
    py_se: Any,
    r_att: Any,
    r_se: Any,
    att_atol: float,
    se_rtol: float,
) -> List[str]:
    """
    Strict headline gates for one Python arm vs R: ATT and SE against the
    documented tolerances, plus the published CI-overlap criterion - each
    enforced independently (overlapping CIs never waive an out-of-tolerance
    SE, and every RENDERED arm is gated, not just the rust-vs-R comparison).
    Non-finite values always gate. Returned flags carry HARD_FLAG_TOKENS
    substrings.
    """

    def _finite(x) -> Optional[float]:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    flags: List[str] = []
    py_att_f, r_att_f = _finite(py_att), _finite(r_att)
    py_se_f, r_se_f = _finite(py_se), _finite(r_se)
    if py_att_f is None or r_att_f is None:
        flags.append(f"headline_att_gate_fail:{arm_name}:nonfinite")
    else:
        att_diff = abs(py_att_f - r_att_f)
        att_rel = att_diff / (abs(r_att_f) + 1e-10)
        if not (att_diff < att_atol or att_rel < 0.01):
            flags.append(f"headline_att_gate_fail:{arm_name}:{att_diff:.2e}")
    if py_se_f is None or r_se_f is None:
        flags.append(f"headline_se_gate_fail:{arm_name}:nonfinite")
    else:
        se_rel = abs(py_se_f - r_se_f) / (abs(r_se_f) + 1e-10)
        if se_rel >= se_rtol:
            flags.append(f"headline_se_gate_fail:{arm_name}:{se_rel:.3f}")
    if None not in (py_att_f, py_se_f, r_att_f, r_se_f):
        # Published criterion: 95% CIs must overlap (same formula as
        # compare_estimates); non-finite cases are already gated above.
        py_lo, py_hi = py_att_f - 1.96 * py_se_f, py_att_f + 1.96 * py_se_f
        r_lo, r_hi = r_att_f - 1.96 * r_se_f, r_att_f + 1.96 * r_se_f
        overlap = (py_lo <= r_att_f <= py_hi) or (r_lo <= py_att_f <= r_hi)
        if not overlap:
            flags.append(f"ci_gate_fail:{arm_name}")
    return flags


# ---------------------------------------------------------------------------
# Venv management
# ---------------------------------------------------------------------------


def venv_python(venv_name: str) -> Path:
    return VENVS_DIR / venv_name / "bin" / "python"


def setup_venv(venv_name: str, pin: str) -> None:
    """Create an isolated venv with a pinned diff-diff release wheel."""
    vdir = VENVS_DIR / venv_name
    py = venv_python(venv_name)
    if not py.exists():
        print(f"[setup] creating venv {vdir} (python {PYTHON_PIN})")
        subprocess.run(
            [UV_BIN, "venv", str(vdir), "--python", PYTHON_PIN],
            check=True,
        )
    print(f"[setup] installing diff-diff=={pin} into {venv_name}")
    subprocess.run(
        [UV_BIN, "pip", "install", "--python", str(py), f"diff-diff=={pin}"],
        check=True,
    )


def preflight_venv(venv_name: str, pin: str) -> Dict[str, Any]:
    """Import diff_diff inside the venv and assert wheel provenance."""
    code = (
        "import json, diff_diff\n"
        "from diff_diff import HAS_RUST_BACKEND\n"
        "try:\n"
        "    from diff_diff._backend import rust_backend_info\n"
        "    rbi = rust_backend_info()\n"
        "except Exception:\n"
        "    rbi = None\n"
        "import numpy, pandas, sys\n"
        "print(json.dumps({'version': diff_diff.__version__,"
        " 'path': diff_diff.__file__, 'has_rust': bool(HAS_RUST_BACKEND),"
        " 'rust_backend_info': rbi, 'python': sys.version.split()[0],"
        " 'numpy': numpy.__version__, 'pandas': pandas.__version__}))\n"
    )
    env = _child_env()
    proc = subprocess.run(
        [str(venv_python(venv_name)), "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(RAW_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"preflight failed for {venv_name}: {proc.stderr.strip()}")
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    if info["version"] != pin:
        raise RuntimeError(f"{venv_name}: expected diff-diff=={pin}, got {info['version']}")
    if str(VENVS_DIR / venv_name) not in info["path"]:
        raise RuntimeError(f"{venv_name}: diff_diff resolved OUTSIDE the venv: {info['path']}")
    if not info["has_rust"]:
        raise RuntimeError(f"{venv_name}: wheel has no Rust backend")
    rbi = info.get("rust_backend_info")
    if rbi is not None and not rbi.get("accelerate"):
        raise RuntimeError(f"{venv_name}: wheel is not linked against Accelerate: {rbi}")
    print(
        f"[preflight] {venv_name}: diff-diff {info['version']} "
        f"(python {info['python']}, numpy {info['numpy']}, "
        f"pandas {info['pandas']}, rust+accelerate OK)"
    )
    return info


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _child_env() -> Dict[str, str]:
    """
    Child env: no PYTHONPATH (installed wheel wins over the dev tree) and no
    inherited DIFF_DIFF_* knobs - a stray DIFF_DIFF_DEMEAN_CHUNK_COLS or
    similar in the parent shell would silently change benchmark code paths
    while the artifacts claim wheel defaults. The benchmark script itself
    sets DIFF_DIFF_BACKEND inside the subprocess from --backend.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("DIFF_DIFF_") and k not in THREAD_ENV_VARS
    }
    env.pop("PYTHONPATH", None)
    env["DIFF_DIFF_BENCH_USE_INSTALLED"] = "1"
    return env


def run_python_rep(
    script_name: str,
    data_path: Path,
    out_path: Path,
    backend: str,
    venv: str,
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    """One Python replication in a fresh subprocess using a pinned venv."""
    cmd = [
        str(venv_python(venv)),
        str(BENCHMARK_DIR / "python" / script_name),
        "--data",
        str(data_path),
        "--output",
        str(out_path),
        "--backend",
        backend,
    ]
    if warmup:
        cmd.append("--warmup")
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env(),
        cwd=str(RAW_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"python rep failed ({script_name}, backend={backend}, venv={venv}):\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    with open(out_path) as f:
        return json.load(f)


def run_r_rep(
    script_name: str,
    data_path: Path,
    out_path: Path,
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    """One R replication in a fresh Rscript subprocess."""
    cmd = [
        "Rscript",
        # --vanilla: no user/site .Rprofile or .Renviron - those could set
        # fixest/data.table thread counts behind the enforced-defaults
        # claim. Package discovery is unaffected (built-in R_LIBS_USER
        # default applies without .Renviron).
        "--vanilla",
        str(BENCHMARK_DIR / "R" / script_name),
        "--data",
        str(data_path),
        "--output",
        str(out_path),
        "--warmup",
        "true" if warmup else "false",
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env(),
        cwd=str(RAW_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"R rep failed ({script_name}):\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    with open(out_path) as f:
        return json.load(f)


def validate_python_provenance(
    result: Dict[str, Any],
    expected_version: str,
    expected_venv: str,
    backend_arm: str,
) -> None:
    """Hard-fail if the subprocess imported the wrong diff_diff."""
    prov = result.get("provenance")
    if not prov:
        raise RuntimeError("no provenance block in Python result JSON")
    if prov.get("diff_diff_version") != expected_version:
        raise RuntimeError(
            f"provenance: version {prov.get('diff_diff_version')} != pin " f"{expected_version}"
        )
    path = prov.get("diff_diff_path") or ""
    if str(VENVS_DIR / expected_venv) not in path:
        raise RuntimeError(
            f"provenance: diff_diff imported from OUTSIDE venv "
            f"{expected_venv}: {path} (dev-tree shadowing?)"
        )
    has_rust = prov.get("has_rust_backend")
    if backend_arm == "python" and has_rust:
        raise RuntimeError("provenance: pure-python arm ran WITH rust backend")
    if backend_arm == "rust" and not has_rust:
        raise RuntimeError("provenance: rust arm ran WITHOUT rust backend")
    if backend_arm == "rust":
        rbi = prov.get("rust_backend_info")
        if rbi is not None and not rbi.get("accelerate"):
            raise RuntimeError(f"provenance: rust arm not linked against Accelerate: {rbi}")
    # No unexpected DIFF_DIFF_* knobs may reach the benchmark subprocess:
    # the artifacts claim wheel defaults (plus the explicit backend arm).
    env_rec = prov.get("diff_diff_env") or {}
    allowed = {"DIFF_DIFF_BENCH_USE_INSTALLED", "DIFF_DIFF_BACKEND"}
    unexpected = sorted(set(env_rec) - allowed)
    if unexpected:
        raise RuntimeError(
            f"provenance: unexpected DIFF_DIFF_* knobs in benchmark env: {unexpected}"
        )
    if env_rec.get("DIFF_DIFF_BENCH_USE_INSTALLED") != "1":
        raise RuntimeError(
            "provenance: DIFF_DIFF_BENCH_USE_INSTALLED not active - "
            "the dev-tree path guard was off"
        )
    backend_env = env_rec.get("DIFF_DIFF_BACKEND")
    if backend_arm == "python" and backend_env != "python":
        raise RuntimeError(
            f"provenance: pure arm expected DIFF_DIFF_BACKEND=python, got {backend_env!r}"
        )
    if backend_arm == "rust" and backend_env not in (None, "rust"):
        raise RuntimeError(
            f"provenance: rust arm expected DIFF_DIFF_BACKEND unset/rust, got {backend_env!r}"
        )


# ---------------------------------------------------------------------------
# Arm-level replication loop
# ---------------------------------------------------------------------------


def run_arm(
    label: str,
    rep_fn,
    n_reps: int,
    allow_cv_rerun: bool = True,
) -> Dict[str, Any]:
    """
    Run n_reps sequential subprocess replications of one arm.

    rep_fn() -> parsed result JSON of a single replication. Returns a dict
    with the last result, timing stats over all reps (first excluded), the
    per-rep ATT determinism check, and CV flag status.
    """

    def _one_pass() -> Dict[str, Any]:
        timings: List[float] = []
        atts: List[float] = []
        ses: List[float] = []
        n_nonfinite_att = 0
        n_nonfinite_se = 0
        result = None
        for rep in range(n_reps):
            result = rep_fn()
            t = result["timing"]["total_seconds"]
            timings.append(t)
            att = result.get("overall_att", result.get("att"))
            if att is not None:
                att_f = float(att)
                if math.isfinite(att_f):
                    atts.append(att_f)
                else:
                    # Fail-closed: a NaN/Inf replication is a correctness
                    # failure even if the LAST rep (which feeds the headline
                    # gates) happens to be finite.
                    n_nonfinite_att += 1
            se = result.get("overall_se", result.get("se"))
            if se is not None:
                se_f = float(se)
                if math.isfinite(se_f):
                    ses.append(se_f)
                else:
                    n_nonfinite_se += 1
            print(f"    [{label}] rep {rep + 1}/{n_reps}: {t:.3f}s")
        stats = compute_timing_stats(timings)
        att_spread = (max(atts) - min(atts)) if atts else 0.0
        cv = stats["stats"]["std"] / stats["stats"]["mean"] if stats["stats"].get("mean") else 0.0
        return {
            "result": result,
            "timing": stats,
            "cv": cv,
            "att_spread": att_spread,
            "n_nonfinite_att": n_nonfinite_att,
            "n_nonfinite_se": n_nonfinite_se,
            # Rep-to-rep SE values: seeded arms are constant; the unseeded R
            # synthdid placebo varies - recorded so the documented SE gate
            # tolerance is auditable against actual Monte Carlo dispersion.
            "se_values": ses,
        }

    def _correctness_flags(o: Dict[str, Any]) -> List[str]:
        f: List[str] = []
        if o["att_spread"] > 1e-12:
            # Same data + seeded estimators: ATT must be identical across
            # reps. Non-deterministic point estimates are a correctness
            # failure - the token makes this a HARD publication gate.
            f.append(f"rep_att_gate_fail:{o['att_spread']:.2e}")
        if o["n_nonfinite_att"]:
            f.append(f"rep_att_gate_fail:nonfinite:{o['n_nonfinite_att']}")
        if o["n_nonfinite_se"]:
            f.append(f"rep_se_gate_fail:nonfinite:{o['n_nonfinite_se']}")
        return f

    out = _one_pass()
    flags: List[str] = _correctness_flags(out)
    if out["cv"] > CV_FLAG and allow_cv_rerun and n_reps > 2:
        if flags:
            # A hard correctness failure was already observed; a timing
            # rerun would DISCARD that evidence if the second pass came
            # back clean. Keep the failed pass and let the gates fire.
            print(
                f"    [{label}] CV {out['cv']:.1%} > {CV_FLAG:.0%} but "
                f"correctness flags present - skipping rerun: {flags}"
            )
            flags.append(f"cv_flag:{out['cv']:.3f}")
        else:
            print(f"    [{label}] CV {out['cv']:.1%} > {CV_FLAG:.0%} - rerunning arm once")
            time.sleep(COOLDOWN_SECONDS)
            out = _one_pass()
            flags.extend(_correctness_flags(out))
            if out["cv"] > CV_FLAG:
                flags.append(f"cv_flag:{out['cv']:.3f}")
                print(f"    [{label}] still flagged (CV {out['cv']:.1%})")
    elif out["cv"] > CV_FLAG:
        flags.append(f"cv_flag:{out['cv']:.3f}")
    out["flags"] = flags
    return out


def compare_effect_arrays(
    py_effects: List[Dict[str, Any]],
    r_effects: List[Dict[str, Any]],
    join_keys: List[str],
    se_rtol: float,
    att_atol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Align two per-effect arrays (period/group-time/event-study/group) on
    join_keys and compare att/se. Returns compact metrics for the committed
    results JSON; the caller turns failures into hard gate flags.

    This is what enforces the docs' detailed parity claims (per-period and
    per-(g,t)/event-time effects matching R) - the headline ATT/SE comparison
    alone cannot certify them.
    """

    def key_of(row: Dict[str, Any]):
        return tuple(round(float(row[k]), 9) for k in join_keys)

    def build(rows):
        out = {}
        dups = 0
        dropped = 0
        for row in rows:
            att = row.get("att")
            if att is None or not math.isfinite(float(att)):
                dropped += 1  # fail-closed: a non-finite effect is a defect
                continue
            k = key_of(row)
            if k in out:
                dups += 1  # duplicate join key would silently overwrite
            out[k] = row
        return out, dups, dropped

    py_map, py_dups, py_dropped = build(py_effects or [])
    r_map, r_dups, r_dropped = build(r_effects or [])
    common = sorted(set(py_map) & set(r_map))
    keys_clean = (
        len(common) > 0
        and len(py_map) == len(common)
        and len(r_map) == len(common)
        and py_dups == r_dups == 0
        and py_dropped == r_dropped == 0
    )
    metrics: Dict[str, Any] = {
        "n_python": len(py_map),
        "n_r": len(r_map),
        "n_common": len(common),
        "n_only_python": len(py_map) - len(common),
        "n_only_r": len(r_map) - len(common),
        "n_dup_python": py_dups,
        "n_dup_r": r_dups,
        "n_dropped_python": py_dropped,
        "n_dropped_r": r_dropped,
        "n_se_compared": 0,
        "max_att_diff": None,
        "max_se_rel_diff": None,
        "keys_match": keys_clean,
        "att_ok": False,
        "se_ok": False,
    }
    if not common:
        return metrics
    max_att = max(abs(float(py_map[k]["att"]) - float(r_map[k]["att"])) for k in common)
    se_rels = []
    for k in common:
        py_se = py_map[k].get("se")
        r_se = r_map[k].get("se")
        if py_se is None or r_se is None:
            continue
        py_se, r_se = float(py_se), float(r_se)
        if math.isfinite(py_se) and math.isfinite(r_se) and r_se != 0.0:
            se_rels.append(abs(py_se - r_se) / abs(r_se))
    max_se_rel = max(se_rels) if se_rels else 0.0
    metrics["n_se_compared"] = len(se_rels)
    metrics["max_att_diff"] = max_att
    metrics["max_se_rel_diff"] = max_se_rel
    metrics["att_ok"] = max_att < att_atol
    # Fail-closed: every common row must contribute a comparable finite SE.
    metrics["se_ok"] = len(se_rels) == len(common) and max_se_rel < se_rtol
    return metrics


def slim_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky per-observation fields before committing to results JSON."""
    slim = dict(result)
    prov = slim.get("provenance")
    if isinstance(prov, dict):
        # Full paths are needed for runtime provenance validation (which
        # runs BEFORE slimming); the committed artifact keeps only the
        # trailing components so local usernames never land in the repo.
        prov = dict(prov)
        for key in ("diff_diff_path", "python_executable"):
            val = prov.get(key)
            if isinstance(val, str) and val:
                prov[key] = ".../" + "/".join(Path(val).parts[-4:])
        slim["provenance"] = prov
    for key in (
        "unit_weights",
        "unit_weight_ids",
        "time_weights",
        "time_weight_ids",
        "group_time_effects",
        "event_study",
        "group_effects",
        "period_effects",
        "coefficients",
    ):
        if key in slim:
            val = slim.pop(key)
            try:
                slim[f"n_{key}"] = len(val)
            except TypeError:
                pass
    return slim


# ---------------------------------------------------------------------------
# Machine state / metadata
# ---------------------------------------------------------------------------


def check_load(force: bool, enforce: bool = True) -> float:
    load1 = os.getloadavg()[0]
    if load1 > LOAD_GUARD_1MIN:
        msg = f"1-min load average {load1:.2f} > {LOAD_GUARD_1MIN} - " f"the machine is not idle"
        if enforce and not force:
            raise SystemExit(f"ABORT: {msg}. Re-run with --force to override.")
        print(f"WARNING: {msg}")
    return load1


def _cmd_out(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, env=_child_env()
        ).stdout.strip()
    except Exception:
        return None


def collect_run_metadata() -> Dict[str, Any]:
    r_pkgs = _cmd_out(
        [
            "Rscript",
            "--vanilla",
            "-e",
            'for (p in c("did","fixest","synthdid","jsonlite","data.table"))'
            ' cat(p, as.character(packageVersion(p)), "\\n")',
        ]
    )
    packages = {}
    if r_pkgs:
        for line in r_pkgs.splitlines():
            parts = line.split()
            if len(parts) == 2:
                packages[parts[0]] = parts[1]
    mem_bytes = _cmd_out(["sysctl", "-n", "hw.memsize"])
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": {
            "cpu": _cmd_out(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_gb": round(int(mem_bytes) / 2**30) if mem_bytes else None,
            "os": f"macOS {platform.mac_ver()[0]}",
            "arch": platform.machine(),
        },
        "orchestrator_python": sys.version.split()[0],
        "repo_git_sha": _cmd_out(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "r_version": _cmd_out(["Rscript", "--vanilla", "-e", "cat(R.version.string)"]),
        "r_packages": packages,
        "thread_policy": (
            "No arm is thread-restricted: R runs at fixest/data.table "
            "defaults; diff-diff wheels run at Accelerate/rayon defaults. "
            "Thread-count env vars (RAYON/OMP/OPENBLAS/VECLIB/MKL/"
            "data.table) are stripped from every benchmark subprocess and "
            "R runs under --vanilla (no user/site .Rprofile/.Renviron), so "
            "package defaults are enforced, not assumed. Per-arm thread "
            "counts are recorded in each result's metadata."
        ),
        "loadavg_at_start": list(os.getloadavg()),
        "protocol": (
            "Each replication is a fresh subprocess run strictly "
            "sequentially (one benchmark process on the machine at a time) "
            "with an untimed in-process warm-up fit before the timed fit. "
            "The first replication is additionally excluded from statistics. "
            "Published statistic: median of the counted replications. "
            f"Arms with CV > {CV_FLAG:.0%} are rerun once and flagged if "
            "still noisy."
        ),
    }


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_GENERATORS = {
    "basic": lambda cfg: generate_basic_did_data(treatment_effect=5.0, seed=42, **cfg),
    "staggered": lambda cfg: generate_staggered_data(treatment_effect=2.0, seed=42, **cfg),
    "sdid": lambda cfg: generate_sdid_data(treatment_effect=4.0, seed=42, **cfg),
    "multiperiod": lambda cfg: generate_multiperiod_data(treatment_effect=3.0, seed=42, **cfg),
}


def ensure_dataset(kind: str, scale: str, cfg: Dict[str, int]) -> Path:
    """
    Deterministically (re)generate a synthetic dataset CSV shared by all arms.

    The dataset is ALWAYS regenerated in memory and compared against any
    existing file; a stale on-disk CSV (older generator, different config)
    can never be silently benchmarked under seed-42 claims. Byte-identical
    regeneration keeps the file untouched.
    """
    path = DATA_DIR / f"{kind}_{scale}.csv"
    df = _GENERATORS[kind](cfg)
    csv_bytes = df.to_csv(index=False).encode()
    digest = hashlib.sha256(csv_bytes).hexdigest()
    if path.exists() and sha256_file(path) == digest:
        return path
    if path.exists():
        print(f"[data] STALE {path.name} - overwriting with deterministic regeneration")
    else:
        print(f"[data] generating {path.name} ({cfg})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_bytes)
    return path


def ensure_mpdta() -> Path:
    """
    Normalize the committed MPDTA CSV (R `did` schema) to the staggered
    benchmark schema so the existing callaway/did benchmark pair runs it
    unmodified. Asserts the known shape so a wrong/partial source file can
    never silently produce a fake benchmark.
    """
    import pandas as pd

    path = DATA_DIR / "mpdta_refresh.csv"
    df = pd.read_csv(MPDTA_SOURCE)
    df = df.rename(
        columns={
            "countyreal": "unit",
            "year": "time",
            "lemp": "outcome",
            "first.treat": "first_treat",
        }
    )[["unit", "time", "outcome", "first_treat"]]
    if len(df) != 2500:
        raise RuntimeError(f"MPDTA rows: {len(df)} != 2500")
    if df["unit"].nunique() != 500:
        raise RuntimeError("MPDTA counties != 500")
    if set(df["first_treat"].unique()) != {0, 2004, 2006, 2007}:
        raise RuntimeError("MPDTA cohorts unexpected")
    save_benchmark_data(df, path)
    return path


def data_determinism_check() -> None:
    """Same seed -> byte-identical CSV content (checked at small scale)."""
    a = _GENERATORS["basic"]({"n_units": 100, "n_periods": 4}).to_csv(index=False)
    b = _GENERATORS["basic"]({"n_units": 100, "n_periods": 4}).to_csv(index=False)
    if hashlib.sha256(a.encode()).hexdigest() != hashlib.sha256(b.encode()).hexdigest():
        raise RuntimeError("data generation is NOT deterministic at seed 42")
    print("[preflight] data generation determinism OK")


def cooldown(seconds: int = COOLDOWN_SECONDS) -> None:
    time.sleep(seconds)

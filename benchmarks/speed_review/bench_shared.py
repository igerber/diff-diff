"""
Shared harness for the practitioner-workflow performance scenarios.

Each ``bench_<scenario>.py`` script imports ``run_scenario`` and hands it a
list of phases (label, callable). The harness times each phase, wraps the
full chain in a pyinstrument profile, and writes:

- ``benchmarks/speed_review/baselines/<scenario>_<backend>.json`` - per-phase wall-clock
- ``benchmarks/speed_review/baselines/profiles/<scenario>_<backend>.html`` - flame profile

If any phase raises, the exception is caught and recorded as
``{"ok": false}`` in the per-phase JSON, AND the process exits 1 after
artifacts are written so that ``run_all.py`` and CI can detect the failure.

Backend is auto-detected via ``diff_diff._backend.HAS_RUST_BACKEND`` and the
``DIFF_DIFF_BACKEND`` env var. Run each script twice - once with
``DIFF_DIFF_BACKEND=python`` and once with ``DIFF_DIFF_BACKEND=rust`` - to
populate both files.

See ``docs/performance-scenarios.md`` for scenario definitions and
``docs/performance-plan.md`` for the per-scenario findings and action
recommendations derived from these results.
"""

import atexit
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

try:
    from pyinstrument import Profiler
    HAS_PYINSTRUMENT = True
except ImportError:
    HAS_PYINSTRUMENT = False
    Profiler = None  # type: ignore[assignment,misc]

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None  # type: ignore[assignment]


class _RSSSampler:
    """Background thread that samples process RSS every ~10ms.

    Gives per-scenario peak memory without depending on
    `resource.getrusage(RUSAGE_SELF).ru_maxrss` (which is monotonic across
    the whole process and so would leak scale-1 peaks into scale-2 reports
    in multi-scale scripts). If psutil is missing, the sampler reports
    peak=0 and the caller falls back to not recording memory.
    """

    def __init__(self, interval_s=0.01):
        self.interval = interval_s
        self.peak_bytes = 0
        self.start_bytes = 0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process() if HAS_PSUTIL else None

    def start(self):
        if self._proc is None:
            return
        self.start_bytes = self._proc.memory_info().rss
        self.peak_bytes = self.start_bytes
        self._stop.clear()

        def sample():
            while not self._stop.is_set():
                try:
                    rss = self._proc.memory_info().rss
                    if rss > self.peak_bytes:
                        self.peak_bytes = rss
                except Exception:
                    pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc is None:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)

    @property
    def peak_mb(self):
        return self.peak_bytes / (1024 * 1024)

    @property
    def start_mb(self):
        return self.start_bytes / (1024 * 1024)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diff_diff._backend import HAS_RUST_BACKEND

RESULTS_DIR = Path(__file__).resolve().parent / "baselines"
PROFILE_DIR = RESULTS_DIR / "profiles"

# Module-level failure flag. Set True whenever run_scenario sees any phase
# record ok=False. atexit handler below translates this into a nonzero
# process exit code so run_all.py and CI can detect partial-failure runs.
# Multi-scale scripts still complete all scales before the process exits.
_any_phase_failed = False


def _exit_with_failure_status():
    if _any_phase_failed:
        print(
            "\n  [bench_shared] at least one phase failed; "
            "exiting nonzero", file=sys.stderr,
        )
        os._exit(1)


atexit.register(_exit_with_failure_status)


def _backend_label():
    """Return 'rust' or 'python' for file naming."""
    env = os.environ.get("DIFF_DIFF_BACKEND", "auto").lower()
    if env == "python":
        return "python"
    if env == "rust":
        return "rust"
    return "rust" if HAS_RUST_BACKEND else "python"


def run_scenario(scenario_name, phases, metadata=None):
    """Time a list of phases and write JSON + pyinstrument profile.

    Parameters
    ----------
    scenario_name : str
        Filename stem, e.g. ``"campaign_staggered"``. Output files use
        ``<scenario_name>_<backend>.(json|html)``.
    phases : list of (label, callable) tuples
        Each callable takes no arguments and may return a value that is
        passed forward via a shared ``context`` dict — but for simplicity
        phases are independent here; each callable captures what it needs
        from its enclosing scope.
    metadata : dict, optional
        Extra fields folded into the JSON under ``metadata`` (data shape,
        params, etc.). Pure data, no callables.
    """
    backend = _backend_label()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings(
        "ignore", message=".*invalid value encountered in matmul.*",
        category=RuntimeWarning,
    )

    profile = None
    if HAS_PYINSTRUMENT:
        profile = Profiler(async_mode="disabled")
        profile.start()

    sampler = _RSSSampler()
    sampler.start()

    phase_times = {}
    total_start = time.perf_counter()
    try:
        for label, fn in phases:
            t0 = time.perf_counter()
            try:
                fn()
                phase_times[label] = {
                    "seconds": time.perf_counter() - t0,
                    "ok": True,
                    "error": None,
                }
            except Exception as e:
                phase_times[label] = {
                    "seconds": time.perf_counter() - t0,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }
                print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
    finally:
        total_elapsed = time.perf_counter() - total_start
        sampler.stop()
        if profile is not None:
            profile.stop()
            html_path = PROFILE_DIR / f"{scenario_name}_{backend}.html"
            with open(html_path, "w") as f:
                f.write(profile.output_html())
            repo_root = Path(__file__).resolve().parents[2]
            print(f"  profile -> {html_path.relative_to(repo_root)}")

    memory = {
        "available": HAS_PSUTIL,
        "start_mb": round(sampler.start_mb, 2) if HAS_PSUTIL else None,
        "peak_mb": round(sampler.peak_mb, 2) if HAS_PSUTIL else None,
        "growth_mb": (
            round(sampler.peak_mb - sampler.start_mb, 2)
            if HAS_PSUTIL else None
        ),
        "sampler_interval_s": sampler.interval,
    }

    record = {
        "scenario": scenario_name,
        "backend": backend,
        "has_rust_backend": HAS_RUST_BACKEND,
        "total_seconds": total_elapsed,
        "memory": memory,
        "phases": phase_times,
        "metadata": metadata or {},
        "diff_diff_version": _get_version(),
        "numpy_version": np.__version__,
    }

    json_path = RESULTS_DIR / f"{scenario_name}_{backend}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    mem_str = (
        f"  peak_rss={memory['peak_mb']:.0f}MB  "
        f"(+{memory['growth_mb']:.0f}MB during run)"
        if HAS_PSUTIL else "  [no psutil; skipping memory]"
    )
    print(
        f"\n  [{scenario_name}] backend={backend}  "
        f"total={total_elapsed:.2f}s{mem_str}"
    )
    for label, info in phase_times.items():
        status = "OK " if info["ok"] else "ERR"
        print(f"    {status} {label:<40} {info['seconds']:>8.3f}s")
    repo_root = Path(__file__).resolve().parents[2]
    print(f"  json    -> {json_path.relative_to(repo_root)}")

    if any(not info["ok"] for info in phase_times.values()):
        global _any_phase_failed
        _any_phase_failed = True

    return record


def _get_version():
    try:
        import diff_diff
        return diff_diff.__version__
    except Exception:
        return "unknown"

"""
Smoke tests for Python code blocks in RST documentation.

Extracts ``.. code-block:: python`` snippets from RST files and executes them
in isolated namespaces with synthetic data and mock dataset loaders. Fails on
all exceptions except NameError (context-dependent snippets) and
ImportError for known third-party/optional packages (comparison-page
snippets and optional-dependency guards like matplotlib).
"""

import re
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# RST files to validate (the ones that had review findings + key user-facing)
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

RST_FILES = [
    "choosing_estimator.rst",
    "troubleshooting.rst",
    "quickstart.rst",
    "index.rst",
    "api/datasets.rst",
    "api/diagnostics.rst",
    "api/utils.rst",
    "api/prep.rst",
    "api/two_stage.rst",
    "api/bacon.rst",
    "api/visualization.rst",
    "api/honest_did.rst",
    "api/pretrends.rst",
    "api/power.rst",
    "python_comparison.rst",
    "r_comparison.rst",
]

# ---------------------------------------------------------------------------
# Snippet extraction
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(
    r"^\.\.\s+code-block::\s+python\s*$\n"  # directive line
    r"(?:\s*:\w[^:]*:.*\n)*"  # optional directive options
    r"\n"  # blank separator
    r"((?:[ \t]+\S.*\n|[ \t]*\n)+)",  # indented body
    re.MULTILINE,
)

# RST ``::`` shorthand code blocks (paragraph ending with ``::``, blank line,
# indented body).  Only matches paragraph-ending ``::`` — excludes RST
# directives (lines starting with ``..``).
_SHORTHAND_BLOCK_RE = re.compile(
    r"^(?!\s*\.\.).*\S::\s*$\n"  # non-directive line ending with ::
    r"\n"  # blank separator
    r"((?:[ \t]+\S.*\n|[ \t]*\n)+)",  # indented body
    re.MULTILINE,
)

# Heuristic: skip ``::`` blocks that look like shell or prose, not Python.
_SHELL_HINTS_RE = re.compile(r"^\s*(\$\s|#!|pip\s+install|maturin\s)", re.MULTILINE)
_PROSE_HINT_RE = re.compile(r"^[A-Z][a-z]+ [a-z]+ [a-z]+", re.MULTILINE)  # English prose sentence


def _extract_snippets(rst_path: Path) -> List[Tuple[int, str]]:
    """Return list of (block_index, dedented_code) from an RST file."""
    text = rst_path.read_text()
    snippets = []
    idx = 0
    for m in _CODE_BLOCK_RE.finditer(text):
        code = textwrap.dedent(m.group(1))
        snippets.append((idx, code))
        idx += 1
    for m in _SHORTHAND_BLOCK_RE.finditer(text):
        code = textwrap.dedent(m.group(1))
        # Skip blocks that look like shell commands or prose, not Python
        if _SHELL_HINTS_RE.search(code) or _PROSE_HINT_RE.search(code):
            continue
        snippets.append((idx, code))
        idx += 1
    return snippets


# ---------------------------------------------------------------------------
# Skip heuristics
# ---------------------------------------------------------------------------
_SKIP_PATTERNS = [
    r"%matplotlib",  # Jupyter magics
    r"plt\.show\(\)",  # interactive display
    r"^\s*fig\s*$",  # bare variable display in Jupyter
    r"maturin\s+develop",  # shell commands in python block
    r"pip\s+install",
    r"wild_bootstrap_se\(X,",  # low-level array API pseudo-code
    r"wide_to_long\(",  # references undefined wide_data variable
]

# Third-party packages imported by comparison-page snippets that may not
# be installed in the test environment.  Only these are exempt from
# ImportError failures — diff_diff and stdlib imports must succeed.
_THIRD_PARTY_MODULES = {"pyfixest", "linearmodels", "differences", "matplotlib"}


def _should_skip(code: str) -> Optional[str]:
    """Return a reason string if the snippet should be skipped, else None."""
    for pat in _SKIP_PATTERNS:
        if re.search(pat, code, re.MULTILINE):
            return f"matches skip pattern: {pat}"
    # Skip if no actual Python statements (just comments / blank)
    lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return "no executable statements"
    return None


# ---------------------------------------------------------------------------
# Build parameterized test cases
# ---------------------------------------------------------------------------
def _collect_cases() -> List[Tuple[str, str, Optional[str]]]:
    """Collect (test_id, code, skip_reason) triples."""
    cases = []
    for rel in RST_FILES:
        rst_path = DOCS_DIR / rel
        if not rst_path.exists():
            continue
        label = rel.replace("/", "_").removesuffix(".rst")
        for idx, code in _extract_snippets(rst_path):
            test_id = f"{label}:block{idx}"
            skip = _should_skip(code)
            cases.append((test_id, code, skip))
    return cases


_CASES = _collect_cases()


# ---------------------------------------------------------------------------
# Shared namespace builder
# ---------------------------------------------------------------------------
def _build_namespace() -> dict:
    """
    Build an exec namespace with diff_diff imports and synthetic data.

    Provides ``data`` (staggered panel) and ``balanced`` (same ref) so that
    most snippets that reference ``data`` can execute.
    """
    import diff_diff

    ns: dict = {"__builtins__": __builtins__}

    # Make all public diff_diff names available
    for name in dir(diff_diff):
        if not name.startswith("_"):
            ns[name] = getattr(diff_diff, name)

    ns["diff_diff"] = diff_diff

    # Remove 'results' module — it shadows the common variable name that
    # context-dependent snippets use for fit() return values.
    ns.pop("results", None)

    # Synthetic datasets that doc snippets commonly reference
    rng = np.random.default_rng(42)
    staggered = diff_diff.generate_staggered_data(n_units=60, n_periods=10, seed=42)
    # Add alias columns that doc snippets expect
    # Use a simple time split (not unit-specific) so basic 2x2 DID works
    mid = staggered["period"].median()
    staggered["post"] = (staggered["period"] >= mid).astype(int)
    staggered["treatment"] = staggered["treated"]
    staggered["y"] = staggered["outcome"]
    staggered["unit_id"] = staggered["unit"]
    staggered["x1"] = rng.normal(size=len(staggered))
    staggered["x2"] = rng.normal(size=len(staggered))
    staggered["x3"] = rng.normal(size=len(staggered))
    staggered["state"] = staggered["unit_id"]
    staggered["time"] = staggered["period"]
    # Uppercase aliases for comparison page snippets (R naming conventions)
    staggered["Y"] = staggered["outcome"]
    staggered["id"] = staggered["unit"]
    staggered["G"] = staggered["first_treat"]
    staggered["X1"] = staggered["x1"]
    staggered["X2"] = staggered["x2"]
    staggered["ever_treated"] = staggered["treated"]
    staggered["group"] = np.where(staggered["treated"] == 1, "treatment", "control")
    staggered["exposure"] = rng.uniform(0, 1, size=len(staggered))
    staggered["dose"] = rng.choice([0.0, 0.5, 1.0, 2.0], size=len(staggered))

    ns["data"] = staggered
    ns["balanced"] = staggered.copy()
    ns["df"] = staggered

    # numpy / pandas always handy
    ns["np"] = np
    ns["pd"] = pd

    # matplotlib stub so plot calls don't actually render
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns["plt"] = plt
        ns["matplotlib"] = matplotlib
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Mock dataset loaders — return synthetic DataFrames matching schemas
    # so that dataset doc snippets execute without network access.
    # ------------------------------------------------------------------
    def _mock_load_card_krueger(**kwargs):
        n = 40
        return pd.DataFrame(
            {
                "store_id": range(n),
                "state": ["NJ"] * (n // 2) + ["PA"] * (n // 2),
                "chain": (["bk", "kfc", "roys", "wendys"] * 10)[:n],
                "emp_pre": rng.normal(20, 5, n),
                "emp_post": rng.normal(21, 5, n),
                "wage_pre": rng.normal(4.5, 0.3, n),
                "wage_post": rng.normal(5.0, 0.3, n),
                "treated": [1] * (n // 2) + [0] * (n // 2),
            }
        )

    def _mock_load_castle_doctrine(**kwargs):
        states = [f"S{i:02d}" for i in range(10)]
        years = list(range(2000, 2011))
        rows = [(s, y) for s in states for y in years]
        n = len(rows)
        ft = [0] * 55 + [2005] * 22 + [2007] * 22 + [2009] * 11
        return pd.DataFrame(
            {
                "state": [r[0] for r in rows],
                "year": [r[1] for r in rows],
                "first_treat": ft[:n],
                "homicide_rate": rng.normal(5, 1, n),
                "population": rng.integers(500000, 5000000, n),
                "income": rng.normal(30000, 5000, n),
                "treated": [1 if ft[i] and r[1] >= ft[i] else 0 for i, r in enumerate(rows)][:n],
                "cohort": ft[:n],
            }
        )

    def _mock_load_divorce_laws(**kwargs):
        states = [f"S{i:02d}" for i in range(10)]
        years = list(range(1965, 1990))
        rows = [(s, y) for s in states for y in years]
        n = len(rows)
        ft = [0] * 125 + [1970] * 50 + [1975] * 50 + [1980] * 25
        return pd.DataFrame(
            {
                "state": [r[0] for r in rows],
                "year": [r[1] for r in rows],
                "first_treat": ft[:n],
                "divorce_rate": rng.normal(4, 1, n),
                "female_lfp": rng.normal(50, 5, n),
                "suicide_rate": rng.normal(5, 2, n),
                "treated": [1 if ft[i] and r[1] >= ft[i] else 0 for i, r in enumerate(rows)][:n],
                "cohort": ft[:n],
            }
        )

    def _mock_load_mpdta(**kwargs):
        counties = list(range(1, 21))
        years = list(range(2003, 2008))
        rows = [(c, y) for c in counties for y in years]
        n = len(rows)
        ft = ([0] * 25 + [2004] * 25 + [2006] * 25 + [2007] * 25)[:n]
        return pd.DataFrame(
            {
                "countyreal": [r[0] for r in rows],
                "year": [r[1] for r in rows],
                "lpop": rng.normal(10, 1, n),
                "lemp": rng.normal(8, 0.5, n),
                "first_treat": ft,
                "treat": [1 if f != 0 else 0 for f in ft],
            }
        )

    _dataset_dispatch = {
        "card_krueger": _mock_load_card_krueger,
        "castle_doctrine": _mock_load_castle_doctrine,
        "divorce_laws": _mock_load_divorce_laws,
        "mpdta": _mock_load_mpdta,
    }

    def _mock_load_dataset(name, **kwargs):
        if name not in _dataset_dispatch:
            raise ValueError(f"Unknown dataset: {name}")
        return _dataset_dispatch[name](**kwargs)

    def _mock_list_datasets():
        return {
            "card_krueger": "Card & Krueger (1994) minimum wage dataset",
            "castle_doctrine": "Castle Doctrine laws - staggered adoption",
            "divorce_laws": "Unilateral divorce laws - staggered adoption",
            "mpdta": "Minimum wage panel data - simulated CS example",
        }

    # Inject mocks into namespace so `from diff_diff.datasets import ...` works
    import types

    mock_datasets_mod = types.ModuleType("diff_diff.datasets")
    mock_datasets_mod.load_card_krueger = _mock_load_card_krueger
    mock_datasets_mod.load_castle_doctrine = _mock_load_castle_doctrine
    mock_datasets_mod.load_divorce_laws = _mock_load_divorce_laws
    mock_datasets_mod.load_mpdta = _mock_load_mpdta
    mock_datasets_mod.load_dataset = _mock_load_dataset
    mock_datasets_mod.list_datasets = _mock_list_datasets
    import sys

    sys.modules["diff_diff.datasets"] = mock_datasets_mod
    diff_diff.datasets = mock_datasets_mod

    # Also put loaders directly in namespace for bare-name usage
    ns["load_card_krueger"] = _mock_load_card_krueger
    ns["load_castle_doctrine"] = _mock_load_castle_doctrine
    ns["load_divorce_laws"] = _mock_load_divorce_laws
    ns["load_mpdta"] = _mock_load_mpdta
    ns["load_dataset"] = _mock_load_dataset
    ns["list_datasets"] = _mock_list_datasets

    return ns


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_datasets_module():
    """Restore diff_diff.datasets after each test to prevent mock leaking."""
    import sys as _sys
    import diff_diff as _dd

    orig_mod = _sys.modules.get("diff_diff.datasets")
    orig_attr = getattr(_dd, "datasets", None)
    yield
    if orig_mod is not None:
        _sys.modules["diff_diff.datasets"] = orig_mod
    elif "diff_diff.datasets" in _sys.modules:
        del _sys.modules["diff_diff.datasets"]
    if orig_attr is not None:
        _dd.datasets = orig_attr


# Snippets that reference variables from prior context blocks (e.g. ``results``
# from an earlier fit).  Generated by running discovery step — do not add
# entries without verifying the NameError is truly context-dependent.
_CONTEXT_DEPENDENT_SNIPPETS = {
    "api_bacon:block1",
    "api_bacon:block2",
    "api_visualization:block2",
    "api_visualization:block3",
    "api_visualization:block8",
    "python_comparison:block5",
    "quickstart:block3",
    "quickstart:block9",
    "r_comparison:block2",
    "r_comparison:block3",
    "r_comparison:block4",
    "r_comparison:block6",
    "troubleshooting:block8",
}


@pytest.mark.parametrize(
    "test_id, code, skip_reason",
    [pytest.param(tid, c, s, id=tid) for tid, c, s in _CASES],
)
def test_doc_snippet(test_id: str, code: str, skip_reason: Optional[str]):
    """Execute a documentation code snippet and assert no API/runtime errors."""
    if skip_reason:
        pytest.skip(skip_reason)

    ns = _build_namespace()
    try:
        exec(compile(code, f"<{test_id}>", "exec"), ns)
    except NameError as exc:
        if test_id not in _CONTEXT_DEPENDENT_SNIPPETS:
            pytest.fail(
                f"Snippet {test_id} raised unexpected NameError: {exc}\n\n"
                f"Code:\n{textwrap.indent(code, '  ')}"
            )
    except ImportError as exc:
        # Only suppress ImportError for known third-party packages that
        # comparison-page snippets import (or optional-dependency guards
        # that raise ImportError manually with the package name in the
        # message). In-package (diff_diff.*) and stdlib import failures
        # should still fail the test.
        mod_name = getattr(exc, "name", "") or ""
        top_level = mod_name.split(".")[0]
        msg = str(exc).lower()
        is_known = top_level in _THIRD_PARTY_MODULES or any(
            pkg in msg for pkg in _THIRD_PARTY_MODULES
        )
        if not is_known:
            pytest.fail(
                f"Snippet {test_id} raised ImportError for "
                f"'{mod_name}': {exc}\n\n"
                f"Code:\n{textwrap.indent(code, '  ')}"
            )
    except Exception as exc:
        pytest.fail(
            f"Snippet {test_id} raised {type(exc).__name__}: {exc}\n\n"
            f"Code:\n{textwrap.indent(code, '  ')}"
        )

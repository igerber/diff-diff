"""Exact confidence-level display: no summary truncates or rounds fractional coverage.

Every text surface that names a confidence level (``summary()`` headers, event-study
table headers, sup-t band labels, HonestDiD / placebo diagnostics, wild-bootstrap
summaries, Business/Diagnostic report prose and the serialized ``headline.ci_level``
field) routes through ``results_base._coverage_level`` / ``_coverage_pct``. Before
this module existed, ``alpha=0.025`` printed "97% Confidence Interval" (truncation)
or "98%" (rounding) over a 97.5% interval.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import diff_diff as dd
from diff_diff import prep_dgp
from diff_diff.results_base import _alpha_pct, _coverage_level, _coverage_pct

REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_FILES = sorted(
    p for p in (REPO_ROOT / "diff_diff").rglob("*.py") if "__pycache__" not in p.parts
)

# The three shapes the sweep retired. No required identifier prefix, so bare
# ``alpha``, ``_alpha``, ``hd.alpha`` and ``self._context.alpha`` all match; an
# optional ``float(...)`` wrapper is allowed; no required ``int(`` so the
# ``:.0f`` format-string variant matches too.
_ALPHA = r"(?:float\()?[A-Za-z0-9_.]*alpha\)?"
_RETIRED_PATTERNS = [
    re.compile(r"\(\s*1(\.0)?\s*-\s*" + _ALPHA + r"\s*\)\s*\*\s*100"),
    re.compile(r"100(\.0)?\s*\*\s*\(\s*1(\.0)?\s*-\s*" + _ALPHA + r"\s*\)"),
    re.compile(_ALPHA + r"\s*\*\s*100\b"),
]
# (relative path, substring) pairs that are the formatter itself or prose.
_EXEMPT = [
    ("results_base.py", "level = round(100.0 * (1.0 - float(alpha)), 6)"),
    ("results_base.py", 'return f"{round(100.0 * float(alpha), 6):g}"'),
    ("rdd.py", "rdrobust ``level = 100*(1-alpha)``"),
]


class TestFormatter:
    @pytest.mark.parametrize(
        "alpha, level, pct",
        [
            (0.05, 95, "95"),
            (0.10, 90, "90"),
            (0.01, 99, "99"),
            (0.025, 97.5, "97.5"),
            (0.001, 99.9, "99.9"),
            (np.float32(0.05), 95, "95"),
        ],
    )
    def test_coverage_exact(self, alpha, level, pct):
        got = _coverage_level(alpha)
        assert got == level
        # int when integral (schema byte-compatibility), float otherwise
        assert isinstance(got, int) == float(level).is_integer()
        assert _coverage_pct(alpha) == pct

    @pytest.mark.parametrize("alpha, pct", [(0.05, "5"), (0.025, "2.5"), (0.10, "10")])
    def test_alpha_pct(self, alpha, pct):
        assert _alpha_pct(alpha) == pct


class TestSourceGuard:
    def test_no_truncating_or_rounding_coverage_site_remains(self):
        offenders = []
        for path in _PKG_FILES:
            rel = path.relative_to(REPO_ROOT / "diff_diff").as_posix()
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if line.strip().startswith("#"):
                    continue
                if any(rel == f and s in line for f, s in _EXEMPT):
                    continue
                if any(rx.search(line) for rx in _RETIRED_PATTERNS):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, (
            "confidence-level percent computed inline instead of via "
            "results_base._coverage_pct / _coverage_level / _alpha_pct:\n" + "\n".join(offenders)
        )


@pytest.fixture(scope="module")
def did_data():
    return prep_dgp.generate_did_data(seed=1)


@pytest.fixture(scope="module")
def staggered_data():
    return prep_dgp.generate_staggered_data(seed=1, n_units=60, n_periods=6)


@pytest.fixture(scope="module")
def event_study_data():
    return prep_dgp.generate_event_study_data(seed=1, n_units=80)


class TestFractionalAlphaSurfaces:
    """alpha=0.025 renders 97.5, never 97 or 98, on one representative of each shape."""

    def test_did_summary(self, did_data):
        s = (
            dd.DifferenceInDifferences(alpha=0.025)
            .fit(did_data, "outcome", "treated", "post")
            .summary()
        )
        assert "97.5% Confidence Interval" in s
        assert "97% " not in s and "98% " not in s

    def test_callaway_santanna_summary_and_supt_band(self, staggered_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = dd.CallawaySantAnna(alpha=0.025, n_bootstrap=20, seed=1, cband=True).fit(
                staggered_data, "outcome", "unit", "period", "first_treat"
            )
            s = r.summary()
        assert "97.5% Confidence Interval" in s
        assert "97% " not in s and "98% " not in s

    def test_wooldridge_format_string_header(self, staggered_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = (
                dd.WooldridgeDiD(alpha=0.025)
                .fit(staggered_data, "outcome", "unit", "period", first_treat="first_treat")
                .summary()
            )
        assert "97.5%" in s
        assert "98%" not in s

    def test_event_study_table_header(self, event_study_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = (
                dd.TwoWayFixedEffects(alpha=0.025)
                .fit(
                    event_study_data,
                    "outcome",
                    "treated",
                    unit="unit",
                    time="period",
                    event_study=True,
                    post_periods=[5, 6, 7, 8, 9],
                )
                .summary()
            )
        assert "[97.5% CI]" in s
        assert "[97% CI]" not in s and "[98% CI]" not in s

    def test_dcdh_honest_significance_level(self, staggered_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = (
                dd.ChaisemartinDHaultfoeuille(alpha=0.025)
                .fit(
                    staggered_data,
                    "outcome",
                    unit="unit",
                    time="period",
                    treatment="treated",
                    honest_did=True,
                    L_max=2,
                )
                .summary()
            )
        assert "Robust 97.5% CI:" in s
        assert "Significant at 2.5%:" in s
        assert "Significant at 2%:" not in s

    def test_business_report_headline_ci_level_and_prose(self, did_data):
        r_frac = dd.DifferenceInDifferences(alpha=0.025).fit(did_data, "outcome", "treated", "post")
        br = dd.BusinessReport(r_frac, outcome_label="sales", outcome_unit="usd")
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 97.5 and isinstance(h["ci_level"], float)
        assert "97.5% CI:" in br.summary()
        # integral coverage stays an exact int (JSON byte-compatible with the
        # historical schema value)
        r_default = dd.DifferenceInDifferences().fit(did_data, "outcome", "treated", "post")
        h95 = dd.BusinessReport(r_default, outcome_label="sales", outcome_unit="usd").to_dict()[
            "headline"
        ]
        assert h95["ci_level"] == 95 and isinstance(h95["ci_level"], int)

    def test_diagnostic_report_prose(self, did_data):
        r_frac = dd.DifferenceInDifferences(alpha=0.025).fit(did_data, "outcome", "treated", "post")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dr = dd.DiagnosticReport(
                r_frac,
                data=did_data,
                outcome="outcome",
                treatment="treated",
                time="period",
                unit="unit",
            )
            text = dr.summary() + dr.full_report()
        assert "97.5% CI:" in text
        assert "98% CI:" not in text and "97% CI:" not in text

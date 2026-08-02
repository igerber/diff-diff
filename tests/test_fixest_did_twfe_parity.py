"""CI-locked fixest parity for the flagship 2x2 DiD and TWFE standard errors.

The core DiD / TWFE ``fixest::feols`` SE parity was previously only checked by
skip-guarded live-``Rscript`` tests that (a) never run in CI and (b) assert only
``att`` (rtol=1e-3), never the SE. This module pins Python's classical/iid ATT
**and SE** against a committed ``fixest`` golden, so the flagship paths are
SE-locked in CI without R at test time (SE-audit item G2).

The TWFE assertion also locks the SE-audit D4 full-K within-transform rescale
against fixest's iid vcov.

Regenerate the golden via
``Rscript benchmarks/R/generate_fixest_did_twfe_golden.R``.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from diff_diff import DifferenceInDifferences, TwoWayFixedEffects

_GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "benchmarks",
    "data",
    "fixest_did_twfe_golden.json",
)


def _load_golden():
    if not os.path.exists(_GOLDEN_PATH):
        return None
    with open(_GOLDEN_PATH) as fh:
        return json.load(fh)


def _build_df(block):
    return pd.DataFrame({k: v for k, v in block["data"].items()})


_SKIP = pytest.mark.skipif(
    _load_golden() is None,
    reason=(
        "fixest_did_twfe_golden.json not present; regenerate via "
        "`Rscript benchmarks/R/generate_fixest_did_twfe_golden.R`."
    ),
)


@_SKIP
class TestFixestDiDTWFEParity:
    """Python classical (iid) ATT + SE == fixest feols iid, machine precision."""

    def test_did_2x2_classical_se_matches_fixest_iid(self):
        golden = _load_golden()
        df = _build_df(golden["did"])
        res = DifferenceInDifferences(vcov_type="classical").fit(
            df, outcome="outcome", treatment="treated", post="post", unit="unit"
        )
        exp = golden["did"]["iid"]
        np.testing.assert_allclose(res.att, exp["att"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(res.se, exp["se"], atol=1e-10, rtol=0)

    def test_twfe_classical_se_matches_fixest_iid(self):
        """Also locks the SE-audit D4 full-K within-transform rescale."""
        golden = _load_golden()
        df = _build_df(golden["twfe"])
        res = TwoWayFixedEffects(vcov_type="classical").fit(
            df, outcome="outcome", treatment="treated", time="post", unit="unit"
        )
        exp = golden["twfe"]["iid"]
        np.testing.assert_allclose(res.att, exp["att"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(res.se, exp["se"], atol=1e-10, rtol=0)

    def test_did_cluster_se_matches_fixest_exactly(self):
        """Plain-OLS CR1 == fixest CR1 to machine precision on the DiD path
        (balanced AND heteroskedastic/unbalanced scenarios). With no absorbed
        FE, both apply the same (G/(G-1))*((n-1)/(n-k)) small-sample factor —
        the documented fixest-CR1 DOF-convention deviation is an absorbed-FE
        (within-transform) phenomenon only, so the former DiD band-pin is
        tightened to a machine-precision lock (SE-audit G2)."""
        golden = _load_golden()
        for key in ("did", "did_hetero"):
            assert key in golden, f"required golden block {key!r} missing — regenerate the fixture"
            df = _build_df(golden[key])
            res = DifferenceInDifferences(vcov_type="hc1", cluster="unit").fit(
                df, outcome="outcome", treatment="treated", post="post", unit="unit"
            )
            exp = golden[key]["cluster_unit"]
            np.testing.assert_allclose(res.att, exp["att"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(res.se, exp["se"], atol=1e-10, rtol=0)

    def test_twfe_cluster_att_matches_fixest(self):
        """TWFE cluster-robust ATT and SE both match fixest.

        The historical ~0.25% SE band was defect D2 (the within-transform CR1
        factor used k_visible, omitting the non-nested time FE fixest counts);
        under the 3.9 K_reference convergence the ``twfe`` arm matches at
        machine precision (measured 0.0 relative here, 1.9e-16 elsewhere) and
        is locked exactly. The ``twfe_hetero`` arm carries a measured 5.2e-11
        relative residual (MAP-demean / BLAS-order dependent), pinned at
        rtol=1e-9 (~20x headroom) rather than a machine-epsilon literal."""
        golden = _load_golden()
        tolerances = {"twfe": dict(atol=1e-10, rtol=0), "twfe_hetero": dict(rtol=1e-9)}
        for key, tol in tolerances.items():
            assert key in golden, f"required golden block {key!r} missing — regenerate the fixture"
            df = _build_df(golden[key])
            res = TwoWayFixedEffects(vcov_type="hc1", cluster="unit").fit(
                df, outcome="outcome", treatment="treated", time="post", unit="unit"
            )
            exp = golden[key]["cluster_unit"]
            np.testing.assert_allclose(res.att, exp["att"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(res.se, exp["se"], **tol)


@_SKIP
class TestFixestHeteroskedasticParity:
    """SE-audit G2 machine-precision hetero lock on an unbalanced,
    heteroskedastic DGP (error sd varies by arm and period; ~15% of rows
    dropped), where fixest's ``hetero`` (HC1) no longer collapses to iid.

    Scope: the plain-OLS DiD path, where Python exposes an unclustered HC1
    (``DifferenceInDifferences(vcov_type="hc1")``). ``TwoWayFixedEffects``
    deliberately auto-clusters at unit on hc1 (documented convention), so it
    has no public unclustered-hetero surface to lock — its scenario locks
    iid (which also exercises the D4 full-K rescale on an UNBALANCED panel)
    and the cluster ATT via the class above.
    """

    def test_did_hetero_hc1_matches_fixest_machine_precision(self):
        golden = _load_golden()
        assert (
            "did_hetero" in golden
        ), "required golden block 'did_hetero' missing — regenerate the fixture"
        df = _build_df(golden["did_hetero"])
        res = DifferenceInDifferences(vcov_type="hc1").fit(
            df, outcome="outcome", treatment="treated", post="post", unit="unit"
        )
        exp = golden["did_hetero"]["hetero"]
        np.testing.assert_allclose(res.att, exp["att"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(res.se, exp["se"], atol=1e-10, rtol=0)
        # Discriminating: hetero must NOT collapse to iid on this DGP.
        assert abs(exp["se"] - golden["did_hetero"]["iid"]["se"]) > 0.01

    def test_did_hetero_iid_matches_fixest_machine_precision(self):
        golden = _load_golden()
        assert (
            "did_hetero" in golden
        ), "required golden block 'did_hetero' missing — regenerate the fixture"
        df = _build_df(golden["did_hetero"])
        res = DifferenceInDifferences(vcov_type="classical").fit(
            df, outcome="outcome", treatment="treated", post="post", unit="unit"
        )
        exp = golden["did_hetero"]["iid"]
        np.testing.assert_allclose(res.att, exp["att"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(res.se, exp["se"], atol=1e-10, rtol=0)

    def test_twfe_hetero_iid_matches_fixest_machine_precision(self):
        """Unbalanced-panel iid lock — the D4 full-K within-transform rescale
        must hold off the balanced design too."""
        golden = _load_golden()
        assert (
            "twfe_hetero" in golden
        ), "required golden block 'twfe_hetero' missing — regenerate the fixture"
        df = _build_df(golden["twfe_hetero"])
        res = TwoWayFixedEffects(vcov_type="classical").fit(
            df, outcome="outcome", treatment="treated", time="post", unit="unit"
        )
        exp = golden["twfe_hetero"]["iid"]
        np.testing.assert_allclose(res.att, exp["att"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(res.se, exp["se"], atol=1e-10, rtol=0)


@_SKIP
class TestFixestTailDfParity:
    """External p-value/CI gates against the stored fixest values (3.9 / M-127).

    The golden has carried fixest's `p_value`/`ci_lower`/`ci_upper`/`t_stat`
    since it was generated, but nothing asserted them until the tail-df
    consolidation. Two lanes:

    - iid arms under the DEFAULT `df_convention="residual"`: fixest's iid
      t-reference is the residual df, so Python matches directly (measured
      during plan review: Python residual df == fixest implied df on all four
      arms — did 396, twfe 148, did_hetero 209, twfe_hetero 125; p/CI agree
      to ~1e-13 relative).
    - cluster arms under `df_convention="cluster"`: fixest's clustered
      t-reference is G−1, exactly the knob's value.

    Tolerances are PER-ARM, inheriting the SE-band structure: the hetero
    arms' 5.2e-11 relative SE residual amplifies ~12x into the p-value at
    t≈3.5/df=125, so their p/CI pin at rtol=1e-8 (≥20x headroom on the
    measured ~6.3e-10); the exact-SE arms pin at rtol=1e-9.
    """

    _P_TOL = {"did": 1e-9, "twfe": 1e-9, "did_hetero": 1e-8, "twfe_hetero": 1e-8}

    def _fit(self, key, golden, **est_kw):
        df = _build_df(golden[key])
        cls = TwoWayFixedEffects if key.startswith("twfe") else DifferenceInDifferences
        return cls(**est_kw).fit(
            df, outcome="outcome", treatment="treated", time="post", unit="unit"
        )

    def test_iid_p_and_ci_match_fixest_under_residual_default(self):
        golden = _load_golden()
        for key, tol in self._P_TOL.items():
            res = self._fit(key, golden, vcov_type="classical")
            exp = golden[key]["iid"]
            np.testing.assert_allclose(res.t_stat, exp["t_stat"], rtol=tol)
            np.testing.assert_allclose(res.p_value, exp["p_value"], rtol=tol)
            np.testing.assert_allclose(res.conf_int[0], exp["ci_lower"], rtol=tol)
            np.testing.assert_allclose(res.conf_int[1], exp["ci_upper"], rtol=tol)

    def test_cluster_p_and_ci_match_fixest_under_cluster_knob(self):
        golden = _load_golden()
        for key, tol in self._P_TOL.items():
            res = self._fit(key, golden, vcov_type="hc1", cluster="unit", df_convention="cluster")
            exp = golden[key]["cluster_unit"]
            np.testing.assert_allclose(res.t_stat, exp["t_stat"], rtol=tol)
            np.testing.assert_allclose(res.p_value, exp["p_value"], rtol=tol)
            np.testing.assert_allclose(res.conf_int[0], exp["ci_lower"], rtol=tol)
            np.testing.assert_allclose(res.conf_int[1], exp["ci_upper"], rtol=tol)

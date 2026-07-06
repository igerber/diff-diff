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
            df, outcome="outcome", treatment="treated", time="post", unit="unit"
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

    def test_cluster_att_matches_fixest(self):
        """The cluster-robust ATT matches fixest exactly (the SE carries the
        documented CR1 DOF-convention difference and is deferred)."""
        golden = _load_golden()
        for key, est in (
            ("did", DifferenceInDifferences(vcov_type="hc1", cluster="unit")),
            ("twfe", TwoWayFixedEffects(vcov_type="hc1", cluster="unit")),
        ):
            df = _build_df(golden[key])
            res = est.fit(df, outcome="outcome", treatment="treated", time="post", unit="unit")
            np.testing.assert_allclose(
                res.att, golden[key]["cluster_unit"]["att"], atol=1e-10, rtol=0
            )
            # SE-audit G2: ratio-band pin on the cluster-robust SE. The exact
            # value carries the documented ~0.25% fixest-CR1 small-sample
            # DOF-convention deviation (SE_AUDIT.md), so it is not machine-
            # precision lockable here; this pins that we never regress BEYOND
            # the known band (catches an unintended CR1 SE-formula change). The
            # machine-precision hetero/cluster lock is the deferred G2 golden
            # regeneration (needs an unbalanced/heteroskedastic DGP).
            assert res.se == pytest.approx(golden[key]["cluster_unit"]["se"], rel=0.005)

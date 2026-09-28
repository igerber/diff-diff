"""Drift detection for Tutorial 23 (``docs/tutorials/23_spillover_tva.ipynb``).

The tutorial narrative quotes seed-specific numbers (naive vs.
SpilloverDiD comparison, sensitivity-grid endpoints, HC1 vs. Conley
SE). If library numerics drift (estimator changes, RNG path changes,
BLAS path changes), the prose can go stale silently while
``pytest --nbmake`` still passes — it only checks that the cells
execute without error.

These asserts re-derive the same numbers using the locked T23 DGP
duplicated below (verbatim from the notebook §2 code cell), then check
them against the values quoted in the tutorial markdown. If a future
change moves any number outside its tolerance band, this test fails
and a maintainer is forced to either update the prose or investigate
the methodology shift before merge.

T23 is the first SpilloverDiD tutorial. It demonstrates the
``SpilloverDiD`` estimator on a TVA-style synthetic panel reproducing
the Butts (2021) §4 Table 1 Panel A ~40% understatement direction
(naive multi-period TWFE significantly understates the direct effect
when near-controls absorb spillover). The DGP-builder constants below
MUST stay in sync with the corresponding constants in the notebook §2
code cell; the ``test_dgp_true_parameters_match_quoted`` test catches
silent drift on those values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import MultiPeriodDiD, SpilloverDiD

# Locked DGP parameters — must stay in sync with the notebook §2 cell.
MAIN_SEED = 23
N_TREATED = 25
N_NEAR = 120
N_FAR = 55
T_PERIODS = 4
FIRST_TREAT = 3
TAU_TOTAL = -7.4
DELTA_1 = -4.5
D_BAR_KM = 100.0
NOISE_SD = 0.5


def _build_t23_panel(seed: int = MAIN_SEED) -> pd.DataFrame:
    """Duplicated verbatim from the notebook §2 code cell. Keep in sync."""
    rng = np.random.default_rng(seed)
    n_units = N_TREATED + N_NEAR + N_FAR
    units = [f"u{i:04d}" for i in range(n_units)]
    alpha = rng.normal(0.0, 1.0, size=n_units)
    lambda_t = np.array([0.0, 0.5, 1.0, 1.5])[:T_PERIODS]

    coords = np.empty((n_units, 2))
    is_treated_unit = np.zeros(n_units, dtype=bool)
    is_near_unit = np.zeros(n_units, dtype=bool)
    for i in range(N_TREATED):
        coords[i] = (rng.normal(0, 0.05), rng.normal(0, 0.05))
        is_treated_unit[i] = True
    for i in range(N_TREATED, N_TREATED + N_NEAR):
        coords[i] = (rng.uniform(0.1, 0.7), rng.uniform(-0.3, 0.3))
        is_near_unit[i] = True
    for i in range(N_TREATED + N_NEAR, n_units):
        coords[i] = (rng.uniform(2.0, 3.0), rng.uniform(-0.5, 0.5))

    rows = []
    for i, u in enumerate(units):
        for t in range(1, T_PERIODS + 1):
            D_it = int(is_treated_unit[i] and t >= FIRST_TREAT)
            Ring1_it = int(is_near_unit[i] and t >= FIRST_TREAT)
            y = (
                alpha[i]
                + lambda_t[t - 1]
                + TAU_TOTAL * D_it
                + DELTA_1 * Ring1_it * (1 - D_it)
                + rng.normal(0, NOISE_SD)
            )
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": coords[i, 0],
                    "lon": coords[i, 1],
                    "ever_treated": int(is_treated_unit[i]),
                    "D": D_it,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _build_t23_panel()


@pytest.fixture(scope="module")
def naive_fit(panel):
    est = MultiPeriodDiD()
    with warnings.catch_warnings():
        # absorb=['unit'] makes the unit-invariant 'ever_treated' indicator
        # perfectly collinear with the unit FE; MultiPeriodDiD drops it
        # (with a UserWarning) and identifies the ATT through the
        # ever_treated x post interaction columns. This is the expected
        # TWFE specification; the rank-deficient drop is benign.
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Rank-deficient design matrix"
        )
        return est.fit(
            panel,
            outcome="y",
            treatment="ever_treated",
            time="time",
            post_periods=[3, 4],
            unit="unit",
            absorb=["unit"],
            reference_period=2,  # explicit pre-period; matches the current MPD default
        )


def _silence_spillover_matmul_warnings():
    """Apply the notebook's narrow ``.*encountered in matmul``
    ``RuntimeWarning`` filter. The three matmul warnings ("divide by
    zero" / "overflow" / "invalid value") are an Apple Silicon M4 +
    macOS Sequoia + numpy<2.3 Accelerate BLAS artifact documented at
    ``docs/dev-status.md`` under "RuntimeWarnings — Apple Silicon M4 BLAS
    bug"
    (root cause: Apple BLAS SME kernels corrupt the FP status register;
    tracked as numpy#28687, fixed in numpy>=2.3). They DO NOT fire on
    M3 / Intel / Linux or numpy>=2.3 — so this filter is a no-op there,
    and any platform-specific noise it does silence does not affect
    result correctness.

    The post-filter warning surface (zero remaining warnings on the
    T23 DGP) is pinned by ``test_spillover_fit_warning_policy_post_filter_clean``
    and ``test_spillover_conley_fit_warning_policy_post_filter_clean``.
    A new RuntimeWarning with a different message, or any UserWarning /
    FutureWarning, fails those tests."""
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul")


@pytest.fixture(scope="module")
def spillover_fit(panel):
    est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"))
    with warnings.catch_warnings():
        _silence_spillover_matmul_warnings()
        return est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")


@pytest.fixture(scope="module")
def spillover_conley_lag0_fit(panel):
    est = SpilloverDiD(
        rings=[0.0, D_BAR_KM],
        conley_coords=("lat", "lon"),
        vcov_type="conley",
        conley_cutoff_km=D_BAR_KM,
        conley_lag_cutoff=0,
    )
    with warnings.catch_warnings():
        _silence_spillover_matmul_warnings()
        return est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")


@pytest.fixture(scope="module")
def spillover_conley_lag1_fit(panel):
    est = SpilloverDiD(
        rings=[0.0, D_BAR_KM],
        conley_coords=("lat", "lon"),
        vcov_type="conley",
        conley_cutoff_km=D_BAR_KM,
        conley_lag_cutoff=1,
    )
    with warnings.catch_warnings():
        _silence_spillover_matmul_warnings()
        return est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_panel_composition(panel):
    """800 rows = 200 units × 4 periods; 25 treated, 120 near, 55 far."""
    assert len(panel) == (N_TREATED + N_NEAR + N_FAR) * T_PERIODS == 800
    assert panel["unit"].nunique() == N_TREATED + N_NEAR + N_FAR == 200
    assert panel["time"].nunique() == T_PERIODS == 4
    treated_units = panel.loc[panel["ever_treated"] == 1, "unit"].nunique()
    assert treated_units == N_TREATED == 25


def test_panel_geographic_bands(panel):
    """Treated cluster within ~10 km of origin (5 sigma * 2 lat/lon),
    near-controls in [0.1, 0.7] lat degrees, far-controls in [2.0, 3.0]."""
    units = panel.drop_duplicates("unit")
    treated = units[units["ever_treated"] == 1]
    untreated = units[units["ever_treated"] == 0]
    near = untreated[untreated["lat"] <= 1.0]
    far = untreated[untreated["lat"] > 1.0]

    assert len(near) == N_NEAR == 120
    assert len(far) == N_FAR == 55

    # Treated cluster geometry
    assert treated["lat"].abs().max() < 0.3
    assert treated["lon"].abs().max() < 0.3
    # Near-control band geometry
    assert near["lat"].min() >= 0.1
    assert near["lat"].max() <= 0.7
    # Far-control band geometry
    assert far["lat"].min() >= 2.0
    assert far["lat"].max() <= 3.0


def test_dgp_true_parameters_match_quoted():
    """True parameters quoted in the tutorial narrative (§2)."""
    assert TAU_TOTAL == -7.4
    assert DELTA_1 == -4.5
    assert D_BAR_KM == 100.0
    assert NOISE_SD == 0.5
    assert MAIN_SEED == 23


def test_estimator_construction_matches_quoted():
    """The §5 fit instantiation parameters must match the docstring narrative."""
    est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"))
    params = est.get_params()
    assert params["rings"] == [0.0, 100.0]
    assert params["d_bar"] is None  # auto-default to max(rings)
    assert params["conley_coords"] == ("lat", "lon")
    assert params["vcov_type"] == "hc1"


def test_naive_twfe_understates_tau_total(naive_fit):
    """§3 quoted: naive ATT ≈ -4.29, ~58% of true tau_total (~42% understatement)."""
    ratio = naive_fit.att / TAU_TOTAL
    assert 0.55 <= ratio <= 0.62, f"naive ratio={ratio:.3f} outside [0.55, 0.62] band"


def test_naive_att_endpoint_matches_quoted(naive_fit):
    """§3 quoted endpoint: 2-decimal pin matching the published `-4.29`
    in the notebook, README, and CHANGELOG. The well-conditioned naive
    MultiPeriodDiD fit is stable across BLAS paths to better than 0.005,
    so 2-decimal pinning is safe (in contrast to the borderline-rank-
    deficient `rings=[0,50]` sensitivity point, which we keep at
    round-to-1)."""
    assert round(naive_fit.att, 2) == -4.29


def test_spillover_did_recovers_tau_total(spillover_fit):
    """§5 quoted: SpilloverDiD tau_total = -7.34, recovers true -7.4
    (within 0.5 tolerance bound). Endpoint pinned to 2 decimals
    matching the published `-7.34` in the notebook, README, and
    CHANGELOG — well-conditioned fit, BLAS-stable at 2 decimals."""
    assert abs(spillover_fit.att - TAU_TOTAL) < 0.5
    assert round(spillover_fit.att, 2) == -7.34


def test_spillover_did_recovers_delta_1(spillover_fit):
    """§5 quoted: SpilloverDiD delta_1 = -4.53, recovers true -4.5
    (within 0.5 tolerance bound). Endpoint pinned to 2 decimals
    matching the published `-4.53` in the notebook, README, and
    CHANGELOG — well-conditioned fit, BLAS-stable at 2 decimals."""
    delta_1 = float(spillover_fit.spillover_effects.iloc[0]["coef"])
    assert abs(delta_1 - DELTA_1) < 0.5
    assert round(delta_1, 2) == -4.53


def test_rings_sensitivity_grid_endpoints(panel):
    """§4 quoted: d_bar=50 → tau=-5.4, others (100/150/200) → tau=-7.3.

    Per the plan and reviewer guidance, round-to-1 tolerance is safer
    than round-to-2 against BLAS divergence on the borderline-rank-deficient
    smallest grid point.
    """
    expected_tau = {50.0: -5.4, 100.0: -7.3, 150.0: -7.3, 200.0: -7.3}
    expected_delta = {50.0: -2.6, 100.0: -4.5, 150.0: -4.5, 200.0: -4.5}
    for outer in (50.0, 100.0, 150.0, 200.0):
        est = SpilloverDiD(rings=[0.0, outer], conley_coords=("lat", "lon"))
        with warnings.catch_warnings():
            _silence_spillover_matmul_warnings()
            res = est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
        assert res.spillover_effects is not None
        delta_1 = float(res.spillover_effects.iloc[0]["coef"])
        assert round(res.att, 1) == expected_tau[outer], (
            f"d_bar={outer}: tau={res.att:.4f} (rounded {round(res.att, 1)}) "
            f"vs expected {expected_tau[outer]}"
        )
        assert round(delta_1, 1) == expected_delta[outer], (
            f"d_bar={outer}: delta_1={delta_1:.4f} (rounded {round(delta_1, 1)}) "
            f"vs expected {expected_delta[outer]}"
        )


def test_rings_grid_d_bar_100_to_200_identical_delta_1(panel):
    """§4 narrative claim covers BOTH coefficients: `d_bar in {100, 150,
    200}` produces identical tau_total AND delta_1 (the test of
    tau_total identity lives in `test_rings_grid_d_bar_100_to_200_identical`;
    this companion test pins the delta_1 identity so a future drift
    that affects only the spillover coefficient can't leave the
    'identical' notebook claim stale)."""
    deltas = []
    for outer in (100.0, 150.0, 200.0):
        est = SpilloverDiD(rings=[0.0, outer], conley_coords=("lat", "lon"))
        with warnings.catch_warnings():
            _silence_spillover_matmul_warnings()
            res = est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
        assert res.spillover_effects is not None
        deltas.append(float(res.spillover_effects.iloc[0]["coef"]))
    np.testing.assert_allclose(deltas, deltas[0] * np.ones(3), atol=1e-10)


def test_rings_grid_d_bar_100_to_200_identical(panel):
    """§4 narrative claim: once d_bar covers the true spillover horizon
    (which here ends at ~78 km), widening past 100 km adds zero
    observations to the ring and the estimates are identical."""
    results = []
    for outer in (100.0, 150.0, 200.0):
        est = SpilloverDiD(rings=[0.0, outer], conley_coords=("lat", "lon"))
        with warnings.catch_warnings():
            _silence_spillover_matmul_warnings()
            res = est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
        results.append(res.att)
    np.testing.assert_allclose(results, results[0] * np.ones(3), atol=1e-10)


def test_conley_se_differs_from_hc1(spillover_fit, spillover_conley_lag0_fit):
    """§6 sanity: Conley vcov produces a different SE than HC1 by more than
    floating-point noise. Pairs with `test_conley_se_less_than_hc1` which
    pins the direction of the difference for this specific DGP."""
    assert abs(spillover_conley_lag0_fit.se - spillover_fit.se) > 1e-6


def test_conley_se_less_than_hc1(spillover_fit, spillover_conley_lag0_fit):
    """§6 prose claim: 'on this DGP, the Conley spatial-HAC SE comes in
    *lower* than HC1'. Pin the direction so the narrative doesn't go
    stale if a future library change flips the sign of the per-pair
    score covariance and reverses the inequality."""
    assert spillover_conley_lag0_fit.se < spillover_fit.se


def test_conley_se_point_estimates_invariant(
    spillover_fit, spillover_conley_lag0_fit, spillover_conley_lag1_fit
):
    """§6 narrative claim: variance-type choice doesn't move the point
    estimates. tau_total is bit-identical across HC1 / Conley lag=0 /
    Conley lag=1 (all paths use the same OLS solve; only the meat
    differs)."""
    np.testing.assert_allclose(
        [spillover_conley_lag0_fit.att, spillover_conley_lag1_fit.att],
        spillover_fit.att,
        atol=1e-10,
    )


def test_conley_lag_cutoff_changes_se_vs_lag_zero(
    spillover_conley_lag0_fit, spillover_conley_lag1_fit
):
    """§6 sanity: adding the serial term (lag=1) changes the SE relative
    to spatial-only (lag=0). Direction-agnostic — on this DGP it
    shrinks, on others it can grow."""
    assert abs(spillover_conley_lag1_fit.se - spillover_conley_lag0_fit.se) > 1e-6


def test_summary_renders_without_warning(spillover_fit):
    """§5 smoke: the summary() call runs clean on the headline fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = spillover_fit.summary()
    assert isinstance(out, str)
    assert len(out) > 0


def test_notebook_dgp_constants_match_test_module_constants():
    """P2 sync guard: codex R6 caught that `test_notebook_dgp_ast_matches_test_fixture`
    only compares the function body, and `test_dgp_true_parameters_match_quoted`
    only reasserts the test module's own constants. A notebook-only edit
    to `MAIN_SEED`, `N_TREATED`, `N_NEAR`, `N_FAR`, `T_PERIODS`,
    `FIRST_TREAT`, `TAU_TOTAL`, `DELTA_1`, `D_BAR_KM`, or `NOISE_SD`
    would change tutorial behavior without failing either of those
    tests.

    Parses the notebook §2 cell, walks the top-level
    ``Assign`` nodes, and asserts the value of each expected constant
    matches the test module's value. Any notebook-only constant edit
    now fails this guard.

    **CI isolation note:** the project's pure-Python and Rust CI jobs
    copy ``tests/`` to an isolated location WITHOUT the ``docs/``
    tree (to verify the installed package doesn't depend on the
    source tree). When the notebook source isn't reachable, this
    test skips gracefully — the sync-guard contract is meaningful
    in local dev (where edits happen) and the nbmake job separately
    verifies the notebook executes end-to-end."""
    import ast
    import json
    from pathlib import Path

    nb_path = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "23_spillover_tva.ipynb"
    if not nb_path.exists():
        pytest.skip(
            f"Notebook source not found at {nb_path}; this drift guard "
            f"requires source-tree access and is meaningful only in local "
            f"dev. The nbmake CI job separately verifies the notebook "
            f"executes end-to-end."
        )
    with nb_path.open() as f:
        nb = json.load(f)

    matches = [
        c
        for c in nb["cells"]
        if c["cell_type"] == "code" and any("def build_t23_panel" in s for s in c["source"])
    ]
    assert len(matches) == 1, (
        f"Expected exactly one notebook code cell defining `build_t23_panel`; "
        f"found {len(matches)}."
    )
    nb_cell_src = "".join(matches[0]["source"])

    # Walk top-level Assigns in the cell and collect a constant -> value dict.
    nb_consts: dict = {}
    tree = ast.parse(nb_cell_src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    nb_consts[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass  # non-literal RHS; skip

    expected = {
        "MAIN_SEED": MAIN_SEED,
        "N_TREATED": N_TREATED,
        "N_NEAR": N_NEAR,
        "N_FAR": N_FAR,
        "T_PERIODS": T_PERIODS,
        "FIRST_TREAT": FIRST_TREAT,
        "TAU_TOTAL": TAU_TOTAL,
        "DELTA_1": DELTA_1,
        "D_BAR_KM": D_BAR_KM,
        "NOISE_SD": NOISE_SD,
    }
    missing = [k for k in expected if k not in nb_consts]
    assert not missing, (
        f"Notebook §2 cell missing expected constant assignments: {missing}. "
        f"If a constant was renamed or moved, update both notebook and test."
    )
    mismatched = {k: (nb_consts[k], expected[k]) for k in expected if nb_consts[k] != expected[k]}
    assert not mismatched, (
        f"Notebook §2 constants drifted from test module constants: {mismatched}. "
        f"Each entry is (notebook_value, test_value). Update both to match."
    )


def test_notebook_dgp_ast_matches_test_fixture():
    """P2 sync guard: enforces the "verbatim" duplication claim by
    parsing the notebook's §2 ``build_t23_panel`` definition and
    asserting that, modulo function name (notebook: ``build_t23_panel``;
    test: ``_build_t23_panel``) and docstring, its AST matches the test
    fixture's. Catches silent drift in non-constant DGP logic (coordinate
    ranges, lambda_t, row construction) that the numerical-value pins
    don't see — codex R4 P2 flagged this gap.

    Uses ``ast.dump`` for a whitespace-/comment-agnostic comparison:
    semantically identical code matches, cosmetic edits don't trigger
    spurious failures.

    **CI isolation note:** like
    ``test_notebook_dgp_constants_match_test_module_constants``, this
    test skips gracefully when CI's isolated-tests-copy step strips
    the ``docs/`` tree. The sync-guard contract is meaningful in
    local dev where edits happen."""
    import ast
    import inspect
    import json
    from pathlib import Path

    nb_path = Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "23_spillover_tva.ipynb"
    if not nb_path.exists():
        pytest.skip(
            f"Notebook source not found at {nb_path}; this drift guard "
            f"requires source-tree access and is meaningful only in local "
            f"dev. The nbmake CI job separately verifies the notebook "
            f"executes end-to-end."
        )
    with nb_path.open() as f:
        nb = json.load(f)

    matches = [
        c
        for c in nb["cells"]
        if c["cell_type"] == "code" and any("def build_t23_panel" in s for s in c["source"])
    ]
    assert len(matches) == 1, (
        f"Expected exactly one notebook code cell defining `build_t23_panel`; "
        f"found {len(matches)}. If you renamed or split the §2 DGP cell, "
        f"update this test's cell-locator."
    )
    nb_cell_src = "".join(matches[0]["source"])

    def _extract_normalized_fn(src: str, fn_name: str) -> str:
        """Parse `src`, find FunctionDef `fn_name`, strip its docstring,
        rename it to the canonical `build_t23_panel`, and return the
        normalized AST dump."""
        tree = ast.parse(src)
        fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name),
            None,
        )
        assert fn is not None, f"Could not find FunctionDef `{fn_name}` in source"
        if (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        ):
            fn.body = fn.body[1:]
        fn.name = "build_t23_panel"
        return ast.dump(fn, annotate_fields=True, include_attributes=False)

    nb_norm = _extract_normalized_fn(nb_cell_src, "build_t23_panel")
    test_norm = _extract_normalized_fn(inspect.getsource(_build_t23_panel), "_build_t23_panel")

    assert nb_norm == test_norm, (
        f"Notebook §2 DGP cell drifted from test fixture `_build_t23_panel`.\n"
        f"--- notebook AST ---\n{nb_norm[:400]}...\n"
        f"--- test AST ---\n{test_norm[:400]}...\n"
        f"Update one or both so the function bodies match modulo name + docstring."
    )


def test_seed_specific_geometry_pins_match_quoted(panel):
    """P3 sync guard: the §2 panel-layout table and §6 within-cutoff
    enumeration quote seed-specific geometry numbers (max distance from
    origin, cluster diameter, band extents, far×far / near×near
    pair-within-100km percentages). The drift test pins all the values
    quoted in the notebook so prose can't go stale even if the headline
    estimates remain within tolerance — codex R4 P3 flagged this gap."""
    treated = panel[panel["ever_treated"] == 1].drop_duplicates("unit")
    near = panel[(panel["ever_treated"] == 0) & (panel["lat"] <= 1.0)].drop_duplicates("unit")
    far = panel[(panel["ever_treated"] == 0) & (panel["lat"] > 1.0)].drop_duplicates("unit")
    deg_to_km = 111.0

    def _max_dist_from_origin_km(d):
        return float(np.sqrt(d["lat"] ** 2 + d["lon"] ** 2).max() * deg_to_km)

    def _min_dist_from_origin_km(d):
        return float(np.sqrt(d["lat"] ** 2 + d["lon"] ** 2).min() * deg_to_km)

    def _band_diameter_km(d):
        lats = d["lat"].values
        lons = d["lon"].values
        diffs = np.sqrt((lats[:, None] - lats[None, :]) ** 2 + (lons[:, None] - lons[None, :]) ** 2)
        return float(diffs.max() * deg_to_km)

    def _pct_pairs_within_100km(d):
        lats = d["lat"].values
        lons = d["lon"].values
        n = len(lats)
        dist = (
            np.sqrt((lats[:, None] - lats[None, :]) ** 2 + (lons[:, None] - lons[None, :]) ** 2)
            * deg_to_km
        )
        triu = np.triu(np.ones((n, n), dtype=bool), k=1)
        pair_d = dist[triu]
        return float((pair_d <= 100.0).sum() / len(pair_d) * 100.0)

    def _cross_band_max_pair_km(d1, d2):
        lats1, lons1 = d1["lat"].values, d1["lon"].values
        lats2, lons2 = d2["lat"].values, d2["lon"].values
        cross = (
            np.sqrt((lats1[:, None] - lats2[None, :]) ** 2 + (lons1[:, None] - lons2[None, :]) ** 2)
            * deg_to_km
        )
        return float(cross.max())

    def _within_band_median_pair_km(d):
        lats = d["lat"].values
        lons = d["lon"].values
        n = len(lats)
        dist = (
            np.sqrt((lats[:, None] - lats[None, :]) ** 2 + (lons[:, None] - lons[None, :]) ** 2)
            * deg_to_km
        )
        triu = np.triu(np.ones((n, n), dtype=bool), k=1)
        return float(np.median(dist[triu]))

    # §2 quoted: "clustered around (0,0); max ~12 km from origin, cluster diameter ~22 km at seed 23"
    assert round(_max_dist_from_origin_km(treated)) == 12, _max_dist_from_origin_km(treated)
    assert round(_band_diameter_km(treated)) == 22, _band_diameter_km(treated)
    # §2 quoted: "~12-82 km north"
    assert round(_min_dist_from_origin_km(near)) == 12, _min_dist_from_origin_km(near)
    assert round(_max_dist_from_origin_km(near)) == 82, _max_dist_from_origin_km(near)
    # §2 quoted: "~224-331 km north"
    assert round(_min_dist_from_origin_km(far)) == 224, _min_dist_from_origin_km(far)
    assert round(_max_dist_from_origin_km(far)) == 331, _max_dist_from_origin_km(far)
    # §6 quoted: "max within-band pairwise distance is ~131 km" for far band
    # (NOT "lat extent" — geometrically the lat extent is ~111 km; 131 km is
    # the pairwise band diameter accounting for lon dispersion too)
    assert round(_band_diameter_km(far)) == 131, _band_diameter_km(far)
    # §6 quoted: "100% of within-band pairs are within 100 km" for near band
    assert round(_pct_pairs_within_100km(near)) == 100, _pct_pairs_within_100km(near)
    # §6 quoted: "~95% of within-band pair distances are within 100 km" for far band
    assert round(_pct_pairs_within_100km(far)) == 95, _pct_pairs_within_100km(far)
    # §6 quoted: "treated × near-control pairs (max pairwise separation ~90 km)"
    assert round(_cross_band_max_pair_km(treated, near)) == 90, _cross_band_max_pair_km(
        treated, near
    )
    # §6 quoted: "median far/far pairwise distance is ~56 km"
    assert round(_within_band_median_pair_km(far)) == 56, _within_band_median_pair_km(far)


def _assert_post_filter_warning_surface_is_clean(captured) -> None:
    """Shared T19-style platform-agnostic warning-policy assertion.

    The notebook's narrow ``.*encountered in matmul`` filter (see
    `_silence_spillover_matmul_warnings`) silences three Apple Silicon
    M4 + numpy<2.3 Accelerate BLAS warnings that are emitted on the
    affected platform but DO NOT fire on M3 / Intel / Linux or
    numpy>=2.3 (per ``docs/dev-status.md`` "RuntimeWarnings — Apple
    Silicon M4 BLAS bug"). The drift contract this assertion locks is
    platform-agnostic:

    - on platforms where the matmul warnings fire, they get filtered
      and never reach the captured list;
    - on platforms where they don't fire, the filter is a no-op;

    EITHER WAY the post-filter captured list must be empty. Any
    UserWarning, FutureWarning, DeprecationWarning, or RuntimeWarning
    with a non-matmul message will fail this assertion and force the
    maintainer to either update the notebook narrative or fix the
    underlying cause."""
    if not captured:
        return
    details = [(msg.category.__name__, str(msg.message)) for msg in captured]
    assert False, (
        f"Unexpected post-filter warnings on the T23 DGP: {details}. "
        f"If a new warning is genuinely expected, broaden "
        f"`_silence_spillover_matmul_warnings()` and update the §5/§6 "
        f"notebook narrative accordingly."
    )


def test_spillover_fit_warning_policy_post_filter_clean(panel):
    """§5 warning-policy guard (T19-pattern, platform-agnostic).

    Mirrors the notebook's narrow ``.*encountered in matmul`` filter
    inside the capture block, then asserts the post-filter warning
    surface is empty on the T23 DGP. On Apple Silicon M4 + numpy<2.3
    the three known BLAS matmul warnings fire and are filtered; on
    M3 / Intel / Linux or numpy>=2.3 the filter is a no-op. EITHER
    WAY a fresh ``UserWarning`` / ``FutureWarning`` or any non-matmul
    ``RuntimeWarning`` will fail this guard."""
    est = SpilloverDiD(rings=[0.0, D_BAR_KM], conley_coords=("lat", "lon"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _silence_spillover_matmul_warnings()  # mirror notebook §5 filter
        est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
    _assert_post_filter_warning_surface_is_clean(w)


def test_spillover_conley_fit_warning_policy_post_filter_clean(panel):
    """§6 warning-policy guard, parallel to §5 but on the Conley path
    (vcov_type="conley", conley_cutoff_km=d_bar, conley_lag_cutoff in {0, 1}).
    Same T19-style platform-agnostic contract: mirror the notebook
    filter inside the capture, assert no remaining warning escaped."""
    for lag in (0, 1):
        est = SpilloverDiD(
            rings=[0.0, D_BAR_KM],
            conley_coords=("lat", "lon"),
            vcov_type="conley",
            conley_cutoff_km=D_BAR_KM,
            conley_lag_cutoff=lag,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _silence_spillover_matmul_warnings()  # mirror notebook §6 filter
            est.fit(panel, outcome="y", unit="unit", time="time", treatment="D")
        _assert_post_filter_warning_surface_is_clean(w)

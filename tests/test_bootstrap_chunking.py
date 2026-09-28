"""Tests for memory-bounded multiplier-bootstrap weight chunking.

The chunking in :mod:`diff_diff.bootstrap_chunking` tiles the bootstrap *draw*
dimension to cap peak memory at ``O(block x n_units)`` instead of
``O(n_bootstrap x n_units)``. Its load-bearing guarantee is that tiling
reproduces the un-chunked weight *stream* exactly (bit-identical), on whichever
backend is active (Rust absolute per-row seeding; NumPy in-order stream). These
tests lock the weight-stream bit-identity at the helper level and end-to-end
chunk-invariance (to floating-point reassociation) through CallawaySantAnna,
under whatever ``DIFF_DIFF_BACKEND`` the CI matrix selects.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, EfficientDiD, HeterogeneousAdoptionDiD
from diff_diff.bootstrap_chunking import (
    ReplayableWeightStream,
    compute_block_size,
    iter_survey_multiplier_weight_blocks,
    iter_weight_blocks,
    tiled_if_matmul,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch,
    generate_survey_multiplier_weights_batch,
)

WEIGHT_TYPES = ["rademacher", "mammen", "webb"]


def _stack(n_bootstrap, n_gen, weight_type, seed, block_size, expand_index=None):
    """Concatenate all weight blocks from a fresh, identically-seeded rng."""
    rng = np.random.default_rng(seed)
    blocks = list(
        iter_weight_blocks(
            n_bootstrap,
            n_gen,
            weight_type,
            rng,
            expand_index=expand_index,
            block_size=block_size,
        )
    )
    starts = [cs for cs, _ in blocks]
    mat = np.vstack([b for _, b in blocks])
    return starts, mat


class TestComputeBlockSize:
    def test_always_in_bounds(self):
        assert compute_block_size(1000, 200) <= 200
        assert compute_block_size(1000, 200) >= 1

    def test_huge_n_units_floors_to_one_row(self):
        assert compute_block_size(10**9, 200) == 1

    def test_tiny_n_units_fits_all_in_one_block(self):
        assert compute_block_size(1, 200) == 200

    def test_respects_target_bytes(self):
        # 100 units x 8 bytes = 800 B/row; an 8000 B budget -> 10 rows/block
        assert compute_block_size(100, 500, target_bytes=8000) == 10


class TestWeightStreamBitIdentity:
    """Tiling the draw dimension reproduces the single-block stream exactly."""

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    @pytest.mark.parametrize("block_size", [1, 7, 33, 198])
    def test_chunked_equals_single_block(self, weight_type, block_size):
        n_bootstrap, n_gen, seed = 199, 53, 12345
        _, single = _stack(n_bootstrap, n_gen, weight_type, seed, block_size=n_bootstrap)
        starts, chunked = _stack(n_bootstrap, n_gen, weight_type, seed, block_size=block_size)
        assert single.shape == (n_bootstrap, n_gen)
        assert chunked.shape == (n_bootstrap, n_gen)
        # exact: the chunking promise is bit-identity, not approximate equality
        np.testing.assert_array_equal(chunked, single)
        # blocks cover every draw exactly once, in order
        assert starts == list(range(0, n_bootstrap, block_size))

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_expand_index_is_chunk_invariant(self, weight_type):
        # cluster/PSU fan-out: generate at n_gen, expand to unit width per block
        n_bootstrap, n_clusters, n_units, seed = 100, 9, 40, 7
        expand = np.array([i % n_clusters for i in range(n_units)])
        _, single = _stack(
            n_bootstrap,
            n_clusters,
            weight_type,
            seed,
            block_size=n_bootstrap,
            expand_index=expand,
        )
        _, chunked = _stack(
            n_bootstrap,
            n_clusters,
            weight_type,
            seed,
            block_size=11,
            expand_index=expand,
        )
        assert single.shape == (n_bootstrap, n_units)
        np.testing.assert_array_equal(chunked, single)

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_single_block_matches_legacy_generator(self, weight_type):
        # iter_weight_blocks in single-block mode must reproduce the legacy
        # generate_bootstrap_weights_batch wrapper exactly (matched seeds), so the
        # chunked path is anchored to the pre-existing generator, not just to its
        # own single-block mode.
        n_bootstrap, n_gen, seed = 199, 53, 999
        legacy = generate_bootstrap_weights_batch(
            n_bootstrap, n_gen, weight_type, np.random.default_rng(seed)
        )
        _, chunked = _stack(n_bootstrap, n_gen, weight_type, seed, block_size=n_bootstrap)
        assert chunked.shape == legacy.shape
        np.testing.assert_array_equal(chunked, legacy)


class _CountingStream:
    """Re-iterable wrapper that counts how many passes were made."""

    def __init__(self, stream):
        self.stream = stream
        self.n_passes = 0

    def __iter__(self):
        self.n_passes += 1
        return iter(self.stream)


class TestTiledIFMatmul:
    """Oracle + replay + tiling contracts for the fused scatter-GEMM kernel.

    ``tiled_if_matmul`` replaces per-column ``W[:, idx] @ values`` GEMVs with
    one GEMM per (weight block, column tile). The oracle below computes the OLD
    per-column semantics on the materialized weight matrix and requires the
    kernel to match at reassociation tolerance. Replay determinism of
    ``ReplayableWeightStream`` is what makes multi-tile passes sound.
    """

    N_BOOT, N_UNITS, BLOCK = 61, 220, 17

    def _stream(self, weight_type="rademacher", seed=42):
        rng = np.random.default_rng(seed)
        return ReplayableWeightStream(
            lambda r: iter_weight_blocks(
                self.N_BOOT, self.N_UNITS, weight_type, r, block_size=self.BLOCK
            ),
            rng,
        )

    def _weight_matrix(self, weight_type="rademacher", seed=42):
        _, mat = _stack(self.N_BOOT, self.N_UNITS, weight_type, seed, block_size=self.BLOCK)
        return mat

    def _columns(self):
        rng = np.random.default_rng(0)
        n = self.N_UNITS
        sparse_idx = rng.choice(n, size=40, replace=False)
        sparse_vals = rng.standard_normal(40)
        dense_vals = rng.standard_normal(n)
        # two mutually disjoint contributions in one column (the CS cell shape)
        treated_idx = np.arange(0, 30)
        control_idx = np.arange(50, 130)
        treated_vals = rng.standard_normal(30)
        control_vals = rng.standard_normal(80)
        lazy_vals = rng.standard_normal(n)
        nan_idx = np.array([3, 7])
        columns = [
            [(sparse_idx, sparse_vals)],
            [(None, dense_vals)],
            [(treated_idx, treated_vals), (control_idx, control_vals)],
            lambda: [(None, lazy_vals)],
            [(nan_idx, np.array([np.nan, np.nan]))],
            [],  # all-zero column
        ]
        return columns

    @staticmethod
    def _old_semantics(W, columns):
        """The pre-rewrite per-column GEMV semantics: sum of W[:, idx] @ vals."""
        out = np.zeros((W.shape[0], len(columns)))
        for c, spec in enumerate(columns):
            if callable(spec):
                spec = spec()
            for idx, values in spec:
                if idx is None:
                    out[:, c] += W @ values
                else:
                    out[:, c] += W[:, idx] @ values
        return out

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_oracle_matches_old_gemv_semantics(self, weight_type):
        columns = self._columns()
        W = self._weight_matrix(weight_type)
        ref = self._old_semantics(W, columns)
        out = tiled_if_matmul(self._stream(weight_type), self.N_BOOT, self.N_UNITS, columns)
        finite_cols = [0, 1, 2, 3, 5]
        np.testing.assert_allclose(out[:, finite_cols], ref[:, finite_cols], rtol=1e-12, atol=1e-14)
        # NaN influence values poison exactly their own column
        assert np.all(np.isnan(out[:, 4]))
        assert not np.any(np.isnan(out[:, finite_cols]))
        # the empty column is exactly zero
        np.testing.assert_array_equal(out[:, 5], np.zeros(self.N_BOOT))

    def test_single_tile_equals_multi_tile(self):
        columns = self._columns()
        one = tiled_if_matmul(self._stream(), self.N_BOOT, self.N_UNITS, columns, tile_bytes=2**62)
        # 2 columns per tile -> 3 tiles
        tiny = tiled_if_matmul(
            self._stream(),
            self.N_BOOT,
            self.N_UNITS,
            columns,
            tile_bytes=2 * self.N_UNITS * 8,
        )
        np.testing.assert_allclose(tiny, one, rtol=1e-12, atol=1e-14, equal_nan=True)

    def test_multi_tile_actually_replays_the_stream(self):
        # Guards against a silent single-tile no-op: the multi-tile arm must
        # iterate the weight stream once per tile, and still match the oracle.
        columns = self._columns()
        counting = _CountingStream(self._stream())
        out = tiled_if_matmul(
            counting,
            self.N_BOOT,
            self.N_UNITS,
            columns,
            tile_bytes=2 * self.N_UNITS * 8,
        )
        assert counting.n_passes == 3  # 6 columns, 2 per tile
        ref = self._old_semantics(self._weight_matrix(), columns)
        np.testing.assert_allclose(out[:, :4], ref[:, :4], rtol=1e-12, atol=1e-14)

    def test_single_pass_iterator_raises_when_tiled(self):
        # A plain generator would silently exhaust after the first tile.
        gen = iter_weight_blocks(self.N_BOOT, self.N_UNITS, "rademacher", np.random.default_rng(1))
        with pytest.raises(ValueError, match="single-pass"):
            tiled_if_matmul(
                gen,
                self.N_BOOT,
                self.N_UNITS,
                self._columns(),
                tile_bytes=2 * self.N_UNITS * 8,
            )

    def test_empty_inputs(self):
        out = tiled_if_matmul(self._stream(), self.N_BOOT, self.N_UNITS, [])
        assert out.shape == (self.N_BOOT, 0)

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_replay_determinism(self, weight_type):
        # Two passes over a ReplayableWeightStream yield byte-identical blocks
        # on whichever backend is active, and the rng ends exactly where one
        # plain pass would leave it.
        rng = np.random.default_rng(7)
        stream = ReplayableWeightStream(
            lambda r: iter_weight_blocks(
                self.N_BOOT, self.N_UNITS, weight_type, r, block_size=self.BLOCK
            ),
            rng,
        )
        first = [(cs, b.copy()) for cs, b in stream]
        second = [(cs, b.copy()) for cs, b in stream]
        assert [cs for cs, _ in first] == [cs for cs, _ in second]
        for (_, a), (_, b) in zip(first, second):
            np.testing.assert_array_equal(a, b)

        # single plain pass from an identically-seeded rng: same end state
        rng_ref = np.random.default_rng(7)
        for _ in iter_weight_blocks(
            self.N_BOOT, self.N_UNITS, weight_type, rng_ref, block_size=self.BLOCK
        ):
            pass
        assert rng.bit_generator.state == rng_ref.bit_generator.state

    def test_survey_stratified_replay_determinism(self):
        # The stratified survey branch generates its FULL PSU matrix eagerly at
        # factory-call time -- the riskiest replay mechanism. Recreating the
        # factory from restored state must reproduce it bit-identically.
        psu = np.arange(6)
        strata = np.array([0, 0, 0, 1, 1, 1])
        design = _design(psu=psu, strata=strata, weights=np.ones(6))
        rng = np.random.default_rng(11)

        def make_iter(r):
            _, blocks = iter_survey_multiplier_weight_blocks(
                50, design, "rademacher", r, block_size=9
            )
            return blocks

        stream = ReplayableWeightStream(make_iter, rng)
        first = np.vstack([b.copy() for _, b in stream])
        second = np.vstack([b.copy() for _, b in stream])
        np.testing.assert_array_equal(first, second)


class TestCSBootstrapChunkInvariance:
    """CallawaySantAnna bootstrap output is invariant to the chunk size.

    The generated weight stream is bit-identical across chunk sizes (locked by
    ``TestWeightStreamBitIdentity``). The downstream ``weights @ influence``
    matmuls go through BLAS, whose reduction order depends on the operand
    row-count, so the resulting statistics match to within floating-point
    reassociation (~1 ULP) rather than bit-for-bit -- far below bootstrap
    Monte-Carlo error. This mirrors the repo's assert_allclose convention for
    float linalg.
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(0)
        nu, nt = 120, 8
        units = np.repeat(np.arange(nu), nt)
        periods = np.tile(np.arange(nt), nu)
        n = nu * nt
        cohort = rng.integers(0, 3, nu)
        ft_unit = np.where(cohort == 0, 0, np.where(cohort == 1, 3, 5))
        ft = np.repeat(ft_unit, nt)
        post = (periods >= ft) & (ft > 0)
        y = rng.standard_normal(n) + 0.1 * periods + 2.0 * post + 0.5 * np.repeat(cohort, nt)
        return pd.DataFrame({"unit": units, "period": periods, "y": y, "first_treat": ft})

    def _fit(self):
        return CallawaySantAnna(
            control_group="never_treated",
            estimation_method="dr",
            cluster="unit",
            n_bootstrap=200,
            seed=42,
        ).fit(
            self._panel(),
            outcome="y",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="all",
        )

    def test_tiny_chunks_match_single_chunk(self, monkeypatch):
        # Default path: this small panel fits in a single block.
        base = self._fit()
        # Force many tiny blocks on every weight path (unit + survey).
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr("diff_diff.staggered_bootstrap.compute_block_size", lambda *a, **k: 7)
        tiny = self._fit()

        # Continuous bootstrap statistics match to within BLAS reassociation.
        assert tiny.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        assert tiny.cband_crit_value == pytest.approx(base.cband_crit_value, rel=1e-10, abs=1e-12)
        # p-values are discrete proportions over draws; a borderline draw can
        # flip under reassociation, shifting a p-value by O(1/n_bootstrap).
        assert tiny.overall_p_value == pytest.approx(base.overall_p_value, abs=0.02)

        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiny.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        for col in ["se", "conf_int_lower", "conf_int_upper"]:
            np.testing.assert_allclose(t[col].to_numpy(), b[col].to_numpy(), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(t["p_value"].to_numpy(), b["p_value"].to_numpy(), atol=0.02)

        # Event-study and group aggregate effects/SEs/CIs also match under chunking.
        for level in ("event_study", "group"):
            bl = base.to_dataframe(level=level).reset_index(drop=True)
            tl = tiny.to_dataframe(level=level).reset_index(drop=True)
            num_cols = [c for c in bl.columns if bl[c].dtype.kind in "fi" and c != "p_value"]
            assert num_cols, f"no numeric columns to compare for level={level}"
            for col in num_cols:
                np.testing.assert_allclose(
                    tl[col].to_numpy(), bl[col].to_numpy(), rtol=1e-9, atol=1e-10
                )

    def test_cluster_none_default_chunks_match_single(self, monkeypatch):
        # cluster=None is the public default (auto-clusters at unit); confirm the
        # default path is chunk-invariant end-to-end.
        def fit():
            return CallawaySantAnna(
                control_group="never_treated",
                estimation_method="dr",
                cluster=None,
                n_bootstrap=200,
                seed=42,
            ).fit(
                self._panel(),
                outcome="y",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="simple",
            )

        base = fit()
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr("diff_diff.staggered_bootstrap.compute_block_size", lambda *a, **k: 7)
        tiny = fit()
        assert tiny.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)

    @staticmethod
    def _clustered_panel():
        # Units grouped into states (5 units/state) -> the unit->PSU map is a
        # genuine many-units-to-one-PSU fan-out (non-identity expansion), unlike
        # cluster="unit" above.
        rng = np.random.default_rng(1)
        n_states, units_per_state, nt = 10, 5, 8
        nu = n_states * units_per_state
        units = np.repeat(np.arange(nu), nt)
        periods = np.tile(np.arange(nt), nu)
        n = nu * nt
        cohort = rng.integers(0, 3, nu)
        ft_unit = np.where(cohort == 0, 0, np.where(cohort == 1, 3, 5))
        ft = np.repeat(ft_unit, nt)
        post = (periods >= ft) & (ft > 0)
        y = rng.standard_normal(n) + 0.1 * periods + 2.0 * post + 0.5 * np.repeat(cohort, nt)
        state = np.repeat(np.repeat(np.arange(n_states), units_per_state), nt)
        return pd.DataFrame(
            {"unit": units, "period": periods, "y": y, "first_treat": ft, "state": state}
        )

    def _fit_clustered(self):
        return CallawaySantAnna(
            control_group="never_treated",
            estimation_method="dr",
            cluster="state",
            n_bootstrap=200,
            seed=42,
        ).fit(
            self._clustered_panel(),
            outcome="y",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="all",
        )

    def test_nonidentity_cluster_chunks_match_single(self, monkeypatch):
        # Exercises the non-identity PSU fan-out expansion under tiny chunks.
        base = self._fit_clustered()
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr("diff_diff.staggered_bootstrap.compute_block_size", lambda *a, **k: 7)
        tiny = self._fit_clustered()

        assert tiny.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiny.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        for col in ["se", "conf_int_lower", "conf_int_upper"]:
            np.testing.assert_allclose(t[col].to_numpy(), b[col].to_numpy(), rtol=1e-10, atol=1e-12)


class TestCSBootstrapTileInvariance:
    """CallawaySantAnna bootstrap output is invariant to the COLUMN tile size.

    The fused perturbation GEMM tiles the influence-column axis under
    ``_TARGET_TILE_BYTES``; each tile replays the weight stream via
    ``ReplayableWeightStream``. Forcing one column per tile makes every fit
    replay the stream n_cols times -- results must match the default
    single-tile fit at reassociation tolerance on every weight path (unit,
    non-identity cluster, survey-PSU, and STRATIFIED survey, whose per-pass
    factory recreation regenerates the full PSU matrix from restored state).
    """

    @staticmethod
    def _force_one_column_tiles(monkeypatch):
        # tiled_if_matmul resolves the cap at call time, so patching the module
        # attribute is effective (no def-time default binding).
        monkeypatch.setattr("diff_diff.bootstrap_chunking._TARGET_TILE_BYTES", 1)

    def test_unit_path_tiles_match_single_tile(self, monkeypatch):
        fit = TestCSBootstrapChunkInvariance()._fit
        base = fit()
        self._force_one_column_tiles(monkeypatch)
        tiled = fit()

        assert tiled.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        assert tiled.cband_crit_value == pytest.approx(base.cband_crit_value, rel=1e-10, abs=1e-12)
        assert tiled.overall_p_value == pytest.approx(base.overall_p_value, abs=0.02)
        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiled.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        for col in ["se", "conf_int_lower", "conf_int_upper"]:
            np.testing.assert_allclose(t[col].to_numpy(), b[col].to_numpy(), rtol=1e-10, atol=1e-12)
        for level in ("event_study", "group"):
            bl = base.to_dataframe(level=level).reset_index(drop=True)
            tl = tiled.to_dataframe(level=level).reset_index(drop=True)
            num_cols = [c for c in bl.columns if bl[c].dtype.kind in "fi" and c != "p_value"]
            for col in num_cols:
                np.testing.assert_allclose(
                    tl[col].to_numpy(), bl[col].to_numpy(), rtol=1e-9, atol=1e-10
                )

    def test_cluster_path_tiles_match_single_tile(self, monkeypatch):
        # Non-identity PSU fan-out: the per-pass expansion copy must replay too.
        fit = TestCSBootstrapChunkInvariance()._fit_clustered
        base = fit()
        self._force_one_column_tiles(monkeypatch)
        tiled = fit()

        assert tiled.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiled.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        np.testing.assert_allclose(t["se"].to_numpy(), b["se"].to_numpy(), rtol=1e-10, atol=1e-12)

    @staticmethod
    def _survey_panel(stratified):
        rng = np.random.default_rng(3)
        n_psu, units_per_psu, nt = 12, 4, 8
        nu = n_psu * units_per_psu
        units = np.repeat(np.arange(nu), nt)
        periods = np.tile(np.arange(nt), nu)
        n = nu * nt
        cohort = rng.integers(0, 3, nu)
        ft_unit = np.where(cohort == 0, 0, np.where(cohort == 1, 3, 5))
        ft = np.repeat(ft_unit, nt)
        post = (periods >= ft) & (ft > 0)
        y = rng.standard_normal(n) + 0.1 * periods + 2.0 * post + 0.5 * np.repeat(cohort, nt)
        psu = np.repeat(np.repeat(np.arange(n_psu), units_per_psu), nt)
        w = np.repeat(np.exp(rng.normal(0, 0.3, nu)), nt)
        df = pd.DataFrame(
            {"unit": units, "period": periods, "y": y, "first_treat": ft, "psu": psu, "w": w}
        )
        if stratified:
            # 3 strata x 4 PSUs -> routes the STRATIFIED (eager full-generation)
            # branch of iter_survey_multiplier_weight_blocks
            df["stratum"] = df["psu"] % 3
        return df

    def _fit_survey(self, stratified):
        from diff_diff import SurveyDesign

        df = self._survey_panel(stratified)
        design = (
            SurveyDesign(weights="w", strata="stratum", psu="psu")
            if stratified
            else SurveyDesign(weights="w", psu="psu")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CallawaySantAnna(
                control_group="never_treated",
                estimation_method="dr",
                n_bootstrap=200,
                seed=42,
            ).fit(
                df,
                outcome="y",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="all",
                survey_design=design,
            )

    @pytest.mark.parametrize("stratified", [False, True], ids=["psu", "stratified"])
    def test_survey_path_tiles_match_single_tile(self, monkeypatch, stratified):
        base = self._fit_survey(stratified)
        self._force_one_column_tiles(monkeypatch)
        tiled = self._fit_survey(stratified)

        assert tiled.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiled.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        for col in ["se", "conf_int_lower", "conf_int_upper"]:
            np.testing.assert_allclose(t[col].to_numpy(), b[col].to_numpy(), rtol=1e-10, atol=1e-12)

    @staticmethod
    def _rcs_frame():
        # Repeated cross sections: every row one observation with a unique unit
        # id; treatment timing assigned at the observation level (panel=False
        # contract). The bootstrap then perturbs observation-level IFs through
        # the same tiled kernel.
        rng = np.random.default_rng(5)
        n_obs, n_periods = 600, 6
        t = rng.integers(1, n_periods + 1, size=n_obs)
        never = rng.random(n_obs) < 0.4
        g = np.where(never, 0, rng.choice([3, 5], size=n_obs))
        treated = (g > 0) & (t >= g)
        y = 0.2 * t + 2.0 * treated + rng.standard_normal(n_obs)
        return pd.DataFrame({"unit": np.arange(n_obs), "period": t, "y": y, "first_treat": g})

    def test_rcs_path_tiles_match_single_tile(self, monkeypatch):
        # panel=False routes observation-level IFs through the same kernel.
        def fit():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return CallawaySantAnna(
                    estimation_method="reg", panel=False, n_bootstrap=200, seed=42
                ).fit(
                    self._rcs_frame(),
                    "y",
                    "unit",
                    "period",
                    "first_treat",
                    aggregate="all",
                )

        base = fit()
        self._force_one_column_tiles(monkeypatch)
        tiled = fit()

        assert tiled.overall_se == pytest.approx(base.overall_se, rel=1e-10, abs=1e-12)
        b = base.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        t = tiled.to_dataframe().sort_values(["group", "time"]).reset_index(drop=True)
        for col in ["se", "conf_int_lower", "conf_int_upper"]:
            np.testing.assert_allclose(t[col].to_numpy(), b[col].to_numpy(), rtol=1e-10, atol=1e-12)


def _design(psu=None, strata=None, fpc=None, weights=None, lonely_psu="adjust"):
    """Minimal duck-typed ResolvedSurveyDesign for the survey weight generators."""
    return SimpleNamespace(psu=psu, strata=strata, fpc=fpc, weights=weights, lonely_psu=lonely_psu)


class TestSurveyWeightBlocks:
    """`iter_survey_multiplier_weight_blocks` reproduces the full survey generator.

    The chunked survey path must yield the exact same PSU-weight matrix and
    psu_ids as `generate_survey_multiplier_weights_batch`, so the
    CallawaySantAnna survey/cluster bootstrap is bit-identical regardless of
    chunking. Covers unstratified generation (tiled), `psu=None`, the FPC
    scalar, and the stratified fallback (sliced).
    """

    def _assert_matches_full(self, design, weight_type, seed, block_size):
        full_w, full_ids = generate_survey_multiplier_weights_batch(
            199, design, weight_type, np.random.default_rng(seed)
        )
        ids, blocks = iter_survey_multiplier_weight_blocks(
            199, design, weight_type, np.random.default_rng(seed), block_size=block_size
        )
        chunked = np.vstack([b for _, b in blocks])
        np.testing.assert_array_equal(ids, full_ids)
        assert chunked.shape == full_w.shape
        np.testing.assert_array_equal(chunked, full_w)

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_unstratified_psu_tiled_matches_full(self, weight_type):
        # 8 PSUs, 3 units each; tiled generation (block_size << n_bootstrap)
        psu = np.repeat(np.arange(8), 3)
        design = _design(psu=psu, weights=np.ones(len(psu)))
        self._assert_matches_full(design, weight_type, seed=5, block_size=7)

    @pytest.mark.parametrize("weight_type", WEIGHT_TYPES)
    def test_psu_none_matches_full(self, weight_type):
        # psu=None -> each observation is its own PSU
        design = _design(psu=None, weights=np.ones(20))
        self._assert_matches_full(design, weight_type, seed=9, block_size=11)

    def test_unstratified_fpc_scaling_matches_full(self):
        # f = n_psu / fpc = 10/100 = 0.1 -> sqrt(0.9) scaling on every weight
        psu = np.arange(10)
        design = _design(psu=psu, fpc=np.full(10, 100.0), weights=np.ones(10))
        self._assert_matches_full(design, "rademacher", seed=3, block_size=13)

    def test_stratified_fallback_matches_full(self):
        # 2 strata x 3 PSUs -> falls back to full generation + sliced blocks
        psu = np.arange(6)
        strata = np.array([0, 0, 0, 1, 1, 1])
        design = _design(psu=psu, strata=strata, weights=np.ones(6))
        self._assert_matches_full(design, "rademacher", seed=7, block_size=9)


class TestEfficientDiDBootstrapChunkInvariance:
    """EfficientDiD multiplier bootstrap is invariant to the chunk size.

    Mirrors ``TestCSBootstrapChunkInvariance``: the weight stream is bit-identical
    across chunk sizes; the ``weights @ eif`` matmuls reassociate under BLAS, so
    SEs match to ~1 ULP (assert_allclose, not bit-for-bit). Covers all four
    bootstrap paths: unit (cluster=None), cluster (genuine many-units-to-cluster
    fan-out), survey-PSU, and weights-only ``SurveyDesign`` -- the last exercises
    the unit_level_weights / weight-path decoupling (unit weight generation but
    eif_scaled perturbation), which a "survey vs non-survey" mis-keying would
    silently break.
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(2)
        n_states, units_per_state, nt = 10, 6, 6
        nu = n_states * units_per_state
        units = np.repeat(np.arange(nu), nt)
        periods = np.tile(np.arange(nt), nu)
        n = nu * nt
        cohort = rng.integers(0, 3, nu)
        ft_unit = np.where(cohort == 0, 0, np.where(cohort == 1, 3, 4))
        ft = np.repeat(ft_unit, nt)
        post = (periods >= ft) & (ft > 0)
        y = rng.standard_normal(n) + 0.1 * periods + 2.0 * post + 0.5 * np.repeat(cohort, nt)
        state = np.repeat(np.repeat(np.arange(n_states), units_per_state), nt)
        w = np.repeat(1.0 + 0.3 * np.abs(rng.standard_normal(nu)), nt)
        return pd.DataFrame(
            {"unit": units, "period": periods, "y": y, "first_treat": ft, "state": state, "w": w}
        )

    def _fit(self, cluster=None, survey_design=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return EfficientDiD(n_bootstrap=200, seed=42, cluster=cluster).fit(
                self._panel(),
                "y",
                "unit",
                "period",
                "first_treat",
                aggregate="all",
                survey_design=survey_design,
            )

    @staticmethod
    def _ses(r):
        # Flatten every bootstrap SE (overall + group_time + event_study + group)
        # into one vector, ordered by sorted keys, for an nan-safe comparison.
        gt = [r.group_time_effects[k]["se"] for k in sorted(r.group_time_effects)]
        es = (
            [r.event_study_effects[k]["se"] for k in sorted(r.event_study_effects)]
            if r.event_study_effects
            else []
        )
        gp = [r.group_effects[k]["se"] for k in sorted(r.group_effects)] if r.group_effects else []
        return np.array([r.overall_se, *gt, *es, *gp], dtype=float)

    def _run(self, monkeypatch, **fit_kwargs):
        base = self._fit(**fit_kwargs)
        base_ses = self._ses(base)
        # Guard the equal_nan comparison below: require the path actually produced
        # finite bootstrap inference (overall SE + at least one cell SE), so a
        # regression that NaN-outs both base and tiny chunk paths cannot pass.
        assert np.isfinite(base_ses[0]) and np.isfinite(base_ses[1:]).any()
        # Force many tiny blocks on every weight path: bootstrap_chunking covers
        # iter_weight_blocks' internal sizing (unit/cluster); the module-level
        # efficient_did_bootstrap target covers the survey-path block_size call.
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 7)
        monkeypatch.setattr(
            "diff_diff.efficient_did_bootstrap.compute_block_size", lambda *a, **k: 7
        )
        tiny = self._fit(**fit_kwargs)
        np.testing.assert_allclose(self._ses(tiny), base_ses, rtol=1e-9, atol=1e-12, equal_nan=True)

    def test_unit_path(self, monkeypatch):
        self._run(monkeypatch)

    def test_cluster_path(self, monkeypatch):
        self._run(monkeypatch, cluster="state")

    def test_survey_psu_path(self, monkeypatch):
        from diff_diff.survey import SurveyDesign

        self._run(monkeypatch, survey_design=SurveyDesign(psu="state", weights="w"))

    def test_weights_only_survey_path(self, monkeypatch):
        # weights-only SurveyDesign: _use_survey_bootstrap is False (unit weight
        # generation) but unit_level_weights is set (eif_scaled perturbation).
        from diff_diff.survey import SurveyDesign

        self._run(monkeypatch, survey_design=SurveyDesign(weights="w"))


class TestEfficientDiDBootstrapTileInvariance:
    """EfficientDiD bootstrap output is invariant to the COLUMN tile size.

    Tile-forced twins of all four ``TestEfficientDiDBootstrapChunkInvariance``
    weight paths: one column per tile makes the fused perturbation GEMM replay
    the weight stream once per (g,t) column (unit, cluster/expand_index,
    survey-PSU, and weights-only survey -- the last exercising the LAZY
    scaled-EIF columns under multi-pass replay).
    """

    def _run(self, monkeypatch, **fit_kwargs):
        harness = TestEfficientDiDBootstrapChunkInvariance()
        base = harness._fit(**fit_kwargs)
        base_ses = harness._ses(base)
        assert np.isfinite(base_ses[0]) and np.isfinite(base_ses[1:]).any()
        # tiled_if_matmul resolves the cap at call time -> patching the module
        # attribute forces one column per tile.
        monkeypatch.setattr("diff_diff.bootstrap_chunking._TARGET_TILE_BYTES", 1)
        tiled = harness._fit(**fit_kwargs)
        np.testing.assert_allclose(
            harness._ses(tiled), base_ses, rtol=1e-9, atol=1e-12, equal_nan=True
        )

    def test_unit_path(self, monkeypatch):
        self._run(monkeypatch)

    def test_cluster_path(self, monkeypatch):
        self._run(monkeypatch, cluster="state")

    def test_survey_psu_path(self, monkeypatch):
        from diff_diff.survey import SurveyDesign

        self._run(monkeypatch, survey_design=SurveyDesign(psu="state", weights="w"))

    def test_weights_only_survey_path(self, monkeypatch):
        from diff_diff.survey import SurveyDesign

        self._run(monkeypatch, survey_design=SurveyDesign(weights="w"))


class TestHADBootstrapChunkInvariance:
    """HAD event-study sup-t bootstrap is invariant to the chunk size.

    The ``weights @ influence`` perturbations are tiled over draws into the small
    ``(B, n_horizons)`` matrix; the sup-t reduction (nanmax over horizons, then
    quantile) runs post-loop. The weight stream is bit-identical across chunk
    sizes; the simultaneous-band critical value matches to ~1 ULP. Covers the
    non-survey (iter_weight_blocks) and survey (iter_survey_multiplier_weight_blocks)
    paths.
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(73)
        G, T = 150, 4
        d_post = rng.uniform(0.0, 1.0, G)
        rows = []
        for t in range(T):
            for g in range(G):
                dose = d_post[g] if t == T - 1 else 0.0
                y = 0.2 * t + (2.0 * dose if t == T - 1 else 0.0) + 0.5 * rng.standard_normal()
                rows.append((g, t, dose, y))
        panel = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome"])
        # HAD's continuous path requires unit-CONSTANT sampling weights.
        w_unit = 1.0 + 0.3 * np.abs(rng.standard_normal(G))
        panel["w"] = panel["unit"].map(lambda g: w_unit[g])
        return panel

    def _fit(self):
        from diff_diff.survey import SurveyDesign

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return HeterogeneousAdoptionDiD(
                design="continuous_at_zero", seed=42, n_bootstrap=400
            ).fit(
                self._panel(),
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
                survey_design=SurveyDesign(weights="w"),
            )

    def test_survey_path(self, monkeypatch):
        # Public event-study cband always routes through the survey-aware branch
        # (iter_survey_multiplier_weight_blocks); a weights-only design makes
        # n_psu == n_units, the large-allocation case the chunking targets.
        base = self._fit()
        assert np.isfinite(base.cband_crit_value)
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 9)
        monkeypatch.setattr("diff_diff.had.compute_block_size", lambda *a, **k: 9)
        tiny = self._fit()
        assert tiny.cband_crit_value == pytest.approx(base.cband_crit_value, rel=1e-8, abs=1e-10)

    def test_nonsurvey_branch_chunk_invariant(self, monkeypatch):
        # The iid (resolved_survey=None) else-branch is unreachable end-to-end --
        # the cband path always builds a (possibly synthetic) survey design, even
        # for the weights= shortcut -- so the refactored iter_weight_blocks path is
        # exercised by a direct call.
        from diff_diff.had import _sup_t_multiplier_bootstrap

        rng = np.random.default_rng(5)
        n_units, n_h = 80, 4
        infl = rng.standard_normal((n_units, n_h))
        att = rng.standard_normal(n_h) * 0.1
        se = np.abs(rng.standard_normal(n_h)) + 0.5

        def _crit():
            return _sup_t_multiplier_bootstrap(
                infl,
                att,
                se,
                None,
                n_bootstrap=400,
                alpha=0.05,
                seed=42,
                bootstrap_weights="rademacher",
            )[0]

        base = _crit()
        assert np.isfinite(base)
        monkeypatch.setattr("diff_diff.bootstrap_chunking.compute_block_size", lambda *a, **k: 9)
        tiny = _crit()
        assert tiny == pytest.approx(base, rel=1e-8, abs=1e-10)


class TestEffectiveWeightBackend:
    """The backend discriminator behind the post-fit bootstrap replay guard.

    Its predicate must stay IDENTICAL to iter_weight_blocks' own rust_gen
    resolution — a degenerate helper (e.g. always "numpy") would silently
    disarm the fail-closed backend-mismatch guard in
    CallawaySantAnnaResults.aggregate(), so the return value is pinned
    directly here rather than only through the guard.
    """

    def test_matches_the_iter_weight_blocks_predicate(self):
        import diff_diff.bootstrap_chunking as bc

        expected = (
            "rust" if (bc.HAS_RUST_BACKEND and bc._rust_bootstrap_weights is not None) else "numpy"
        )
        assert bc.effective_weight_backend() == expected

    def test_numpy_when_rust_unavailable(self, monkeypatch):
        import diff_diff.bootstrap_chunking as bc

        monkeypatch.setattr(bc, "HAS_RUST_BACKEND", False)
        monkeypatch.setattr(bc, "_rust_bootstrap_weights", None)
        assert bc.effective_weight_backend() == "numpy"

    def test_rust_requires_both_flag_and_symbol(self, monkeypatch):
        import diff_diff.bootstrap_chunking as bc

        # A stale-extension environment can have the flag without the
        # symbol; the generator falls back to NumPy there, so the
        # discriminator must too.
        monkeypatch.setattr(bc, "HAS_RUST_BACKEND", True)
        monkeypatch.setattr(bc, "_rust_bootstrap_weights", None)
        assert bc.effective_weight_backend() == "numpy"

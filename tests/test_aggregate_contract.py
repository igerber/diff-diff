"""Behavioral contract for post-fit ``results.aggregate()`` (spec section 6).

The ``test_ref`` for every aggregate-postfit ledger row this file pins:
M-020/M-023/M-021/M-022 (the CS / EfficientDiD / Imputation / TwoStage
``fit(aggregate=)`` shims), M-024/M-026 (the Stacked / dCDH shims + view
relays), M-025 (the ContinuousDiD shim + MIXED view/recompute aggregate() -
'simple'/'dose' views, 'event_study' pruned-IF-payload kit),
M-117/M-120/M-118/M-119 (``balance_e`` moves onto ``aggregate()``),
M-027/M-139 (the HAD fit + pretest-workflow mode-selector shims with
panel-shape inference, and the two PURE-VIEW results classes)
and M-122 (``AggregationResult``).

The headline gate is NUMERICAL INERTNESS: for every supported type,
``fit(aggregate=T)`` and ``fit(); .aggregate(T)`` must agree to 1e-14. The
refactor that enabled post-fit aggregation touched the influence-function
path, so drift here means a regression, not a design change.
"""

import copy
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, StaggeredTripleDifference
from diff_diff.aggregation import AGGREGATION_SCHEMA, AggregationResult
from diff_diff.results_base import EventStudyResults

FIT_KW = dict(outcome="y", unit="unit", time="time", first_treat="first_treat")


def _panel(seed=11, n_units=80, n_periods=7):
    rng = np.random.default_rng(seed)
    cohorts = [0, 3, 4, 5, 6]
    rows = []
    for u in range(n_units):
        g = cohorts[u % len(cohorts)]
        ui = rng.normal(0, 0.5)
        for t in range(1, n_periods + 1):
            treated = g != 0 and t >= g
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "first_treat": g,
                    "y": ui
                    + 0.4 * t
                    + (1.5 + 0.3 * (t - g) if treated else 0.0)
                    + rng.normal(0, 0.25),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def fitted(panel):
    """A plain fit - no aggregate= argument, so no deprecation warning."""
    return CallawaySantAnna().fit(panel, **FIT_KW)


@pytest.fixture(scope="module")
def fit_time(panel):
    """The DEPRECATED fit-time aggregation, for the inertness comparison."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return CallawaySantAnna().fit(panel, aggregate="all", **FIT_KW)


# --------------------------------------------------------------------------- #
# Numerical inertness (the headline gate)
# --------------------------------------------------------------------------- #


class TestInertness:
    def test_simple_matches_fit_time(self, fitted, fit_time):
        got = fitted.aggregate("simple")
        assert np.allclose(got.att[0], fit_time.overall_att, rtol=1e-14, atol=1e-14)
        assert np.allclose(got.se[0], fit_time.overall_se, rtol=1e-14, atol=1e-14)
        assert np.allclose(got.p_value[0], fit_time.overall_p_value, rtol=1e-14, atol=1e-14)

    def test_group_matches_fit_time(self, fitted, fit_time):
        frame = fitted.aggregate("group").to_dataframe()
        assert len(frame) == len(fit_time.group_effects)
        for _, row in frame.iterrows():
            native = fit_time.group_effects[row["label"]]
            for col, key in (
                ("att", "effect"),
                ("se", "se"),
                ("t_stat", "t_stat"),
                ("p_value", "p_value"),
            ):
                assert np.allclose(
                    row[col], native[key], rtol=1e-14, atol=1e-14, equal_nan=True
                ), f"group[{row['label']}].{col} drifted"

    def test_event_study_matches_fit_time(self, fitted, fit_time):
        frame = fitted.aggregate("event_study").to_dataframe()
        compared = 0
        for _, row in frame.iterrows():
            if bool(row["is_reference"]):
                continue
            native = fit_time.event_study_effects[row["event_time"]]
            for col, key in (
                ("att", "effect"),
                ("se", "se"),
                ("t_stat", "t_stat"),
                ("p_value", "p_value"),
            ):
                assert np.allclose(
                    row[col], native[key], rtol=1e-14, atol=1e-14, equal_nan=True
                ), f"event_study[{row['event_time']}].{col} drifted"
            compared += 1
        assert compared > 0, "no non-reference event times compared"

    @pytest.mark.parametrize("balance_e", [0, 1, 2])
    def test_balance_e_matches_fit_time(self, panel, fitted, balance_e):
        """M-117: balance_e on aggregate() reproduces fit(balance_e=)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna().fit(
                panel, aggregate="event_study", balance_e=balance_e, **FIT_KW
            )
        frame = fitted.aggregate("event_study", balance_e=balance_e).to_dataframe()
        non_ref = frame[~frame["is_reference"].astype(bool)]
        assert len(non_ref) == len(
            [e for e in native.event_study_effects if e in set(non_ref["event_time"])]
        )
        for _, row in non_ref.iterrows():
            got = native.event_study_effects[row["event_time"]]
            assert np.allclose(row["att"], got["effect"], rtol=1e-14, atol=1e-14, equal_nan=True)
            assert np.allclose(row["se"], got["se"], rtol=1e-14, atol=1e-14, equal_nan=True)


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #


class TestImmutability:
    #: The attributes the aggregators used to mutate on their host.
    MUTATED = (
        "event_study_effects",
        "group_effects",
        "event_study_vcov",
        "event_study_vcov_index",
        "event_study_df",
    )

    def test_parent_unchanged_across_mixed_calls(self, fitted):
        before = {f: copy.deepcopy(getattr(fitted, f, None)) for f in self.MUTATED}
        for level in ("event_study", "group", "simple", "group", "event_study"):
            fitted.aggregate(level)
        for f in self.MUTATED:
            after = getattr(fitted, f, None)
            if before[f] is None:
                assert after is None, f"{f} was populated by aggregate()"
            else:
                assert str(before[f]) == str(after), f"{f} changed"

    def test_repeated_calls_agree(self, fitted):
        a = fitted.aggregate("group").to_dataframe()
        b = fitted.aggregate("group").to_dataframe()
        # assert_frame_equal treats corresponding NaNs as equal and handles the
        # mixed object/float dtypes; a bare np.array_equal does neither.
        pd.testing.assert_frame_equal(a, b)

    def test_order_independent(self, fitted):
        es_first = fitted.aggregate("event_study").to_dataframe()
        fitted.aggregate("group")
        es_after = fitted.aggregate("event_study").to_dataframe()
        assert np.allclose(es_first["att"].to_numpy(), es_after["att"].to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# Retention kit
# --------------------------------------------------------------------------- #


class TestRetentionKit:
    def test_kit_holds_no_dataframe(self, fitted):
        """The source panel must never be retained on a results object."""
        kit = fitted._aggregation_kit
        frames = [k for k, v in kit.bookkeeping.items() if isinstance(v, pd.DataFrame)]
        assert frames == [], f"kit retained DataFrame(s): {frames}"
        assert not isinstance(kit.influence, pd.DataFrame)

    def test_no_dataframe_anywhere_on_results(self, fitted):
        frames = [
            f
            for f in fitted.__dataclass_fields__
            if isinstance(getattr(fitted, f, None), pd.DataFrame)
        ]
        assert frames == [], f"results retained DataFrame(s): {frames}"

    def test_aggregate_survives_pickle(self, fitted):
        """No live reference to the estimator or its frame."""
        revived = pickle.loads(pickle.dumps(fitted))
        got = revived.aggregate("group").to_dataframe()
        want = fitted.aggregate("group").to_dataframe()
        assert np.allclose(got["att"].to_numpy(), want["att"].to_numpy(), equal_nan=True)

    def test_no_raw_unit_identifiers_are_retained(self):
        """Data minimization: results objects are picklable and get shared, so
        the kit must not become a carrier for unit identifiers - which are
        routinely names, emails or administrative IDs. The kit needs only
        POSITION (influence arrays index by ``treated_idx``/``control_idx``),
        so it stores canonical 0..n-1 codes.

        Searches the whole serialized artifact, not just the kit: a recursive
        check is what makes this a real guarantee rather than a spot check.
        """
        sentinel = "SENTINEL-ID-{}@example.invalid"
        rng = np.random.default_rng(4)
        rows = []
        for i in range(45):
            g = [0, 3, 4][i % 3]
            for t in range(1, 7):
                treated = g != 0 and t >= g
                rows.append(
                    {
                        "unit": sentinel.format(i),
                        "time": t,
                        "first_treat": g,
                        "y": 0.4 * t + (1.5 if treated else 0.0) + rng.normal(0, 0.25),
                    }
                )
        res = CallawaySantAnna().fit(pd.DataFrame(rows), **FIT_KW)
        assert b"SENTINEL-ID" not in pickle.dumps(res), "raw unit ids reached the artifact"
        # And the minimization is inert: aggregation still produces numbers.
        for level in ("simple", "group", "event_study"):
            assert np.isfinite(np.asarray(res.aggregate(level).att)).any()

    def test_identifier_minimization_is_numerically_inert(self, panel, fitted):
        """Canonical codes must not perturb any aggregate: compared at
        atol=rtol=0, since substituting positions for labels should not change
        a single floating-point operation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna().fit(panel, aggregate="all", **FIT_KW)
        got = fitted.aggregate("simple")
        assert got.att[0] == native.overall_att
        assert got.se[0] == native.overall_se


# --------------------------------------------------------------------------- #
# Fail-closed contract
# --------------------------------------------------------------------------- #


def _assert_bootstrap_simple_relay(res, n_expected=None):
    """Shared per-level-policy pin (converged with M-027): on a bootstrapped
    fit, aggregate('simple') relays the STORED overall quintet verbatim -
    percentile se/p/CI beside the finite ``safe_inference`` t - and only the
    df COLUMN is NaN (no df governs percentile inference)."""
    agg = res.aggregate("simple")
    assert float(agg.att[0]) == res.overall_att
    assert float(agg.se[0]) == res.overall_se
    assert float(agg.t_stat[0]) == res.overall_t_stat
    assert np.isfinite(agg.t_stat[0])
    assert float(agg.p_value[0]) == res.overall_p_value
    assert float(agg.conf_int_lower[0]) == res.overall_conf_int[0]
    assert float(agg.conf_int_upper[0]) == res.overall_conf_int[1]
    assert np.isnan(agg.df[0])
    if n_expected is not None:
        assert float(agg.n[0]) == float(n_expected)
    return agg


class TestFailClosed:
    def test_calendar_unsupported_by_cs(self, fitted):
        with pytest.raises(ValueError, match="calendar"):
            fitted.aggregate("calendar")

    def test_all_is_not_a_post_fit_type(self, fitted):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            fitted.aggregate("all")

    def test_error_names_supported_types(self, fitted):
        with pytest.raises(ValueError) as exc:
            fitted.aggregate("nonsense")
        for level in ("simple", "event_study", "group", "total"):
            assert level in str(exc.value)

    def test_weights_rejected(self, fitted):
        """CS exposes no weighting selector, so anything but None fails closed."""
        with pytest.raises(ValueError, match="weights"):
            fitted.aggregate("simple", "cohort_share")

    @pytest.mark.parametrize("level", ["simple", "group", "total"])
    def test_balance_e_rejected_where_inert(self, fitted, level):
        """balance_e applies to event-study aggregation ONLY - silently
        ignoring it elsewhere would accept a user argument that does nothing."""
        with pytest.raises(ValueError, match="balance_e"):
            fitted.aggregate(level, balance_e=2)

    def test_bootstrap_simple_relays_stored_quintet(self, panel):
        """Per-level policy (converged with M-027): 'simple' is a bit-exact
        relay of the stored overall row, faithful under the bootstrap regime,
        so it stays available with a NaN df column."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = CallawaySantAnna(n_bootstrap=49, seed=42).fit(panel, **FIT_KW)
        _assert_bootstrap_simple_relay(boot, n_expected=boot.n_treated_units + boot.n_control_units)

    def test_bootstrap_survey_simple_relay_df_nan(self, panel):
        """Survey-PSU multiplier bootstrap: the finite survey metadata rides
        the fit, but the relay's df column is still NaN - percentile p is not
        governed by a t-reference, so publishing the survey df beside it
        would misstate provenance."""
        d = panel.copy()
        rng = np.random.default_rng(7)
        wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
        d["w"] = d["unit"].map(wmap)
        d["psu"] = d["unit"] % 7
        from diff_diff import SurveyDesign

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = CallawaySantAnna(n_bootstrap=49, seed=42).fit(
                d,
                **FIT_KW,
                survey_design=SurveyDesign(weights="w", psu="psu"),
            )
        assert boot.survey_metadata is not None
        assert boot.survey_metadata.df_survey is not None
        _assert_bootstrap_simple_relay(boot)


# --------------------------------------------------------------------------- #
# Bootstrap replay: recompute levels on bootstrapped fits
# --------------------------------------------------------------------------- #


def _boot_fit(data, *, seed=42, n_bootstrap=50, fit_kwargs=None, **ctor):
    """A plain bootstrapped fit (modern route: no fit-time aggregate=)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CallawaySantAnna(n_bootstrap=n_bootstrap, seed=seed, **ctor).fit(
            data, **FIT_KW, **(fit_kwargs or {})
        )


def _boot_fit_time(data, *, seed=42, n_bootstrap=50, aggregate="all", fit_kwargs=None, **ctor):
    """The parity REFERENCE: the deprecated fit-time aggregation route.

    The reference is the NATIVE fit-time surface (`event_study_effects` /
    `group_effects` dict entries + `cband_crit_value`), never a second
    aggregate() call — the kit is attached unconditionally at the end of
    fit(), so a fit-time-aggregated result's own aggregate() would ALSO
    replay, and replay-vs-replay proves nothing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CallawaySantAnna(n_bootstrap=n_bootstrap, seed=seed, **ctor).fit(
            data, aggregate=aggregate, **FIT_KW, **(fit_kwargs or {})
        )


def _assert_es_replay_parity(es, ref, n_bootstrap=50):
    """Replayed EventStudyResults vs the native fit-time bootstrap surface.

    Tolerance note (reconciling with this file's 1e-14 analytical-parity
    convention): the replayed weight stream is bit-identical, and only the
    fused GEMM's tile-boundary/column-count reassociation differs between
    the fit-time and replay passes — ~1 ULP relative — so 1e-13 is the same
    contract with one order of headroom for quantile-interpolation
    arithmetic; any real desynchronization is O(1), not O(1e-13). p-values
    are COUNT statistics (a draw within a ULP of the point estimate can
    flip one count), hence the additional 2/n_bootstrap atol.
    """
    ref_es = ref.event_study_effects
    p_atol = 2.0 / n_bootstrap
    for i, e in enumerate(es.event_time):
        if bool(es.is_reference[i]):
            continue
        r = ref_es[int(e)]
        assert float(es.att[i]) == float(r["effect"])  # same analytical point path
        np.testing.assert_allclose(es.se[i], r["se"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(es.t_stat[i], r["t_stat"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(es.p_value[i], r["p_value"], rtol=1e-13, atol=p_atol)
        np.testing.assert_allclose(
            [es.conf_int_lower[i], es.conf_int_upper[i]],
            list(r["conf_int"]),
            rtol=1e-13,
            atol=1e-13,
        )
        cb = r.get("cband_conf_int")
        if cb is not None:
            np.testing.assert_allclose(
                [es.cband_lower[i], es.cband_upper[i]], list(cb), rtol=1e-13, atol=1e-13
            )
    # Percentile provenance: no joint covariance, no analytical df.
    assert es.vcov is None
    assert np.all(np.isnan(np.asarray(es.df, dtype=float)))
    if ref.cband_crit_value is not None:
        np.testing.assert_allclose(es.cband_crit_value, ref.cband_crit_value, rtol=1e-13)


def _assert_group_replay_parity(agg, ref, n_bootstrap=50):
    gdf = agg.to_dataframe()
    p_atol = 2.0 / n_bootstrap
    for _, row in gdf.iterrows():
        r = ref.group_effects[row["label"]]
        assert float(row["att"]) == float(r["effect"])
        if np.isnan(r["se"]):
            assert np.isnan(row["se"])
            continue
        np.testing.assert_allclose(row["se"], r["se"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(row["t_stat"], r["t_stat"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(row["p_value"], r["p_value"], rtol=1e-13, atol=p_atol)
        # df cleared: percentile inference never used the analytical df.
        assert np.isnan(row["df"])


class TestBootstrapReplay:
    """Recompute levels replay the fit-time multiplier bootstrap.

    Warning-coverage note (recorded decision, not a deferral): of the
    warnings the wholesale re-run can re-emit, the two cheaply reachable
    representatives — n_bootstrap<50 and G<2-PSU — are pinned below; the
    degenerate-path warnings ("No post-treatment effects for bootstrap
    aggregation", "Too few valid sup-t bootstrap samples") have no cheap
    fixture (the first needs a fit whose every surviving cell is
    pre-treatment, the second >50% non-finite sup-t draws) and are covered
    by the convention, not pinned individually.
    """

    def test_event_study_parity_with_fit_time(self, panel):
        ref = _boot_fit_time(panel, aggregate="event_study")
        plain = _boot_fit(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = plain.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        _assert_es_replay_parity(es, ref)

    def test_group_parity_with_fit_time(self, panel):
        ref = _boot_fit_time(panel, aggregate="group")
        plain = _boot_fit(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg = plain.aggregate("group")
        _assert_group_replay_parity(agg, ref)

    def test_balance_e_parity_with_fit_time(self, panel):
        ref = _boot_fit_time(panel, aggregate="event_study", fit_kwargs={"balance_e": 1})
        plain = _boot_fit(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = plain.aggregate("event_study", balance_e=1)
        _assert_es_replay_parity(es, ref)

    def test_seedless_fit_replays_and_is_idempotent(self, panel):
        """seed=None fits replay too — the spec captures the actual RNG
        state by value — and repeated calls restore the same stream."""
        plain = _boot_fit(panel, seed=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = plain.aggregate("event_study")
            b = plain.aggregate("event_study")
        np.testing.assert_array_equal(a.se, b.se)
        np.testing.assert_array_equal(a.p_value, b.p_value)

    def test_set_params_and_mutation_immunity(self, panel):
        """The spec is by-value: post-fit set_params / direct attribute
        mutation of the estimator cannot change the replay."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = CallawaySantAnna(n_bootstrap=50, seed=42)
            res = est.fit(panel, **FIT_KW)
            before = res.aggregate("event_study")
            est.set_params(n_bootstrap=5, bootstrap_weights="webb")
            est.seed = 7
            after = res.aggregate("event_study")
        np.testing.assert_array_equal(before.se, after.se)

    def test_pickle_round_trip_replays(self, panel):
        """The PCG64 state dict pickles; an unpickled result replays
        identically in the same environment (same weight backend)."""
        plain = _boot_fit(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = plain.aggregate("event_study")
            rt = pickle.loads(pickle.dumps(plain))
            es_rt = rt.aggregate("event_study")
        np.testing.assert_array_equal(es.se, es_rt.se)

    def test_relays_unchanged_and_order_independent(self, panel):
        """'simple' still relays the stored quintet bit-exactly AFTER an ES
        replay call (the replay writes nothing back to the results)."""
        plain = _boot_fit(panel, n_bootstrap=49)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain.aggregate("event_study")
        _assert_bootstrap_simple_relay(plain)

    def test_legacy_kit_without_spec_fails_closed(self, panel):
        plain = _boot_fit(panel)
        plain._aggregation_kit.bootstrap = None  # simulate a pre-replay pickle
        for level in ("event_study", "group"):
            with pytest.raises(NotImplementedError, match="predates") as exc:
                plain.aggregate(level)
            assert "refit" in str(exc.value)

    def test_backend_mismatch_fails_closed(self, panel):
        """A different weight backend regenerates different draws from the
        same RNG state — the replay must refuse, deterministically under
        either installed backend."""
        import dataclasses

        from diff_diff.bootstrap_chunking import effective_weight_backend

        plain = _boot_fit(panel)
        spec = plain._aggregation_kit.bootstrap
        assert spec.backend == effective_weight_backend()
        other = "numpy" if spec.backend == "rust" else "rust"
        plain._aggregation_kit.bootstrap = dataclasses.replace(spec, backend=other)
        with pytest.raises(NotImplementedError, match="weight backend"):
            plain.aggregate("event_study")
        # None means UNKNOWN and also fails closed (fail-open default on a
        # safety discriminator would disarm the guard for future callers).
        plain._aggregation_kit.bootstrap = dataclasses.replace(spec, backend=None)
        with pytest.raises(NotImplementedError, match="weight backend"):
            plain.aggregate("group")

    def test_low_bootstrap_warning_re_emitted_on_replay(self, panel):
        """The replay re-runs the fit-time engine, so its warnings re-fire
        (the recompute-level re-warn convention; relays stay silent)."""
        plain = _boot_fit(panel, n_bootstrap=49)
        with pytest.warns(UserWarning, match="n_bootstrap=49 is low"):
            plain.aggregate("event_study")

    def test_no_post_treatment_cells_group_returns_zero_rows(self):
        """A fit whose every treated cohort's onset lies beyond the observed
        panel has NO post-treatment cells: overall inference is NaN, only
        pre-treatment cells exist, and the group aggregation is empty. The
        bootstrapped group path must return the supported zero-row result on
        BOTH routes (they share `_run_multiplier_bootstrap`, whose group
        stats block would otherwise np.column_stack an empty list)."""
        rng = np.random.default_rng(0)
        rows = []
        for u in range(30):
            g = 8 if u % 3 else 0  # onset beyond the 5-period panel
            for t in range(1, 6):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": g,
                        "y": 0.3 * t + rng.normal(0, 0.3),
                    }
                )
        data = pd.DataFrame(rows)
        plain = _boot_fit(data, n_bootstrap=25, seed=1)
        assert np.isnan(plain.overall_att)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agg = plain.aggregate("group")
        assert len(agg.to_dataframe()) == 0
        assert np.isnan(plain.overall_att)  # unchanged by the replay
        # The deprecated fit-time route rides the same helper — it must
        # survive too (the crash was reachable there pre-replay).
        ref = _boot_fit_time(data, n_bootstrap=25, seed=1, aggregate="group")
        assert ref.group_effects in (None, {})


class TestBootstrapReplayDesigns:
    """Replay parity on the engine branches the plain panel never reaches."""

    def test_bare_cluster_psu_expansion(self, panel):
        d = panel.copy()
        d["psu"] = d["unit"] % 7
        ref = _boot_fit_time(d, cluster="psu")
        plain = _boot_fit(d, cluster="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_es_replay_parity(plain.aggregate("event_study"), ref)
            _assert_group_replay_parity(plain.aggregate("group"), ref)

    def test_stratified_survey_is_portable_and_matches(self, panel):
        from diff_diff import SurveyDesign

        d = panel.copy()
        rng = np.random.default_rng(5)
        wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
        d["w"] = d["unit"].map(wmap)
        d["psu"] = d["unit"] % 7
        d["stratum"] = d["unit"] % 2
        sd = SurveyDesign(weights="w", strata="stratum", psu="psu", nest=True)
        ref = _boot_fit_time(d, fit_kwargs={"survey_design": sd})
        plain = _boot_fit(d, fit_kwargs={"survey_design": sd})
        # The stratified survey generator draws through NumPy regardless of
        # the installed backend — provably backend-independent, stamped
        # "portable" so the artifact replays anywhere.
        assert plain._aggregation_kit.bootstrap.backend == "portable"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_es_replay_parity(plain.aggregate("event_study"), ref)
            _assert_group_replay_parity(plain.aggregate("group"), ref)

    def test_fpc_survey_matches(self, panel):
        from diff_diff import SurveyDesign

        d = panel.copy()
        rng = np.random.default_rng(6)
        wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
        d["w"] = d["unit"].map(wmap)
        d["psu"] = d["unit"] % 7
        d["fpc"] = 100  # non-census: the fpc_scale branch
        sd = SurveyDesign(weights="w", psu="psu", fpc="fpc")
        ref = _boot_fit_time(d, fit_kwargs={"survey_design": sd})
        plain = _boot_fit(d, fit_kwargs={"survey_design": sd})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_es_replay_parity(plain.aggregate("event_study"), ref)

    def test_repeated_cross_sections_match(self):
        data = _rcs()
        ref = _boot_fit_time(data, panel=False)
        plain = _boot_fit(data, panel=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_es_replay_parity(plain.aggregate("event_study"), ref)
            _assert_group_replay_parity(plain.aggregate("group"), ref)

    def test_unbalanced_panel_matches(self, panel):
        thinned = panel.drop(panel.index[::13]).reset_index(drop=True)
        ref = _boot_fit_time(thinned, allow_unbalanced_panel=True)
        plain = _boot_fit(thinned, allow_unbalanced_panel=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_es_replay_parity(plain.aggregate("event_study"), ref)

    def test_single_psu_nan_surfaces_and_warning(self, panel):
        from diff_diff import SurveyDesign

        d = panel.copy()
        rng = np.random.default_rng(8)
        wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
        d["w"] = d["unit"].map(wmap)
        d["one_psu"] = 1
        single = _boot_fit(
            d, fit_kwargs={"survey_design": SurveyDesign(weights="w", psu="one_psu")}
        )
        # Single-PSU generation is backend-independent (degenerate branch).
        assert single._aggregation_kit.bootstrap.backend == "portable"
        with pytest.warns(UserWarning, match="PSU"):
            es = single.aggregate("event_study")
        nonref = ~np.asarray(es.is_reference, dtype=bool)
        assert np.all(~np.isfinite(np.asarray(es.se, dtype=float)[nonref]))
        assert es.cband_crit_value is None


class TestBootstrapReplayConsumers:
    """The newly-reachable public path: a bootstrapped-CS derived container
    (vcov=None) flowing into the event-study consumers."""

    def test_pretrends_power_rides_diag_fallback(self, panel):
        from diff_diff import compute_pretrends_power

        plain = _boot_fit(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = plain.aggregate("event_study")
            power = compute_pretrends_power(es)
        # Silent by design on the pretrends side: the diagonal fallback is
        # the container contract for vcov=None surfaces.
        assert power is not None

    def test_honest_did_warns_then_uses_diagonal(self, panel):
        from diff_diff import compute_honest_did

        plain = _boot_fit(panel)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = plain.aggregate("event_study")
        with pytest.warns(UserWarning, match="no full covariance"):
            honest = compute_honest_did(es, method="relative_magnitude", M=1.0)
        assert honest is not None


# --------------------------------------------------------------------------- #
# Container schema (M-122)
# --------------------------------------------------------------------------- #


class TestAggregationResult:
    def test_pinned_schema(self, fitted):
        assert tuple(fitted.aggregate("group").to_dataframe().columns) == AGGREGATION_SCHEMA

    def test_event_study_returns_the_unified_container(self, fitted):
        """M-092's container finally gets a public producer."""
        assert isinstance(fitted.aggregate("event_study"), EventStudyResults)

    def test_non_event_study_returns_aggregation_result(self, fitted):
        assert isinstance(fitted.aggregate("group"), AggregationResult)
        assert isinstance(fitted.aggregate("simple"), AggregationResult)

    def test_serializers_present(self, fitted):
        got = fitted.aggregate("group")
        assert isinstance(got.to_dict(), dict)
        assert isinstance(got.to_dataframe(), pd.DataFrame)
        assert isinstance(got.summary(), str)

    def test_group_weight_is_none_not_fabricated(self, fitted):
        """_aggregate_by_group weights (g,t) cells equally WITHIN a cohort and
        forms no cross-cohort mass, so there is no per-row weight to report."""
        assert fitted.aggregate("group").weight is None
        assert fitted.aggregate("group").to_dict()["weight"] is None

    def test_n_kind_is_declared(self, fitted):
        """`n` means different things per level; n_kind is what disambiguates."""
        assert fitted.aggregate("group").n_kind == "cells"
        assert fitted.aggregate("simple").n_kind == "units"

    def test_target_is_per_row(self, fitted):
        got = fitted.aggregate("group")
        assert got.target.shape == got.label.shape

    def test_zero_row_is_a_supported_boundary(self):
        empty = AggregationResult(
            level="group",
            label=np.array([], dtype=object),
            target=np.array([], dtype=object),
            att=np.array([]),
            se=np.array([]),
            t_stat=np.array([]),
            p_value=np.array([]),
            conf_int_lower=np.array([]),
            conf_int_upper=np.array([]),
            n=np.array([]),
            df=np.array([]),
        )
        assert len(empty.to_dataframe()) == 0
        assert isinstance(empty.summary(), str)

    def test_non_estimable_row_keeps_point_estimate(self):
        """safe_inference NaNs t/p/CI only - att and se are inputs, not
        outputs, so NaN-ing the whole quintet would erase valid estimates."""
        got = AggregationResult(
            level="group",
            label=np.array([1], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([2.0]),
            se=np.array([np.nan]),
            t_stat=np.array([np.nan]),
            p_value=np.array([np.nan]),
            conf_int_lower=np.array([np.nan]),
            conf_int_upper=np.array([np.nan]),
            n=np.array([3.0]),
            df=61.0,
        )
        assert got.att[0] == 2.0
        assert np.isnan(got.t_stat[0])
        assert np.isnan(got.df[0]), "df must be NaN where p_value is non-finite"


# --------------------------------------------------------------------------- #
# Deprecation shim (M-020 / M-117)
# --------------------------------------------------------------------------- #


class TestFitShim:
    def test_plain_fit_does_not_warn(self, panel):
        """The sentinel default is what makes this true - a bare None default
        could not tell 'not passed' from 'passed None'."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CallawaySantAnna().fit(panel, **FIT_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    @pytest.mark.parametrize("kwargs", [{"aggregate": "group"}, {"balance_e": 1}])
    def test_deprecated_args_warn(self, panel, kwargs):
        with pytest.warns(FutureWarning, match="aggregate"):
            CallawaySantAnna().fit(panel, **kwargs, **FIT_KW)

    def test_deprecated_path_still_populates_legacy_surface(self, fit_time):
        """Downstream consumers (honest_did, pretrends,
        build_event_study_surface) read these off the results object."""
        assert fit_time.event_study_effects is not None
        assert fit_time.group_effects is not None


def _rcs(seed=17, n_per_cell=6, n_periods=6):
    """Repeated cross-sections: a DIFFERENT set of units each period, so unit
    ids must be globally unique rather than recycled across periods."""
    rng = np.random.default_rng(seed)
    cohorts = [0, 3, 4, 5]
    rows = []
    uid = 0
    for t in range(1, n_periods + 1):
        for g in cohorts:
            for _ in range(n_per_cell):
                treated = g != 0 and t >= g
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "first_treat": g,
                        "y": 0.4 * t + (1.5 if treated else 0.0) + rng.normal(0, 0.25),
                    }
                )
                uid += 1
    return pd.DataFrame(rows)


class TestInertnessAcrossDesigns:
    """The inertness gate on designs the balanced-panel fixture does not reach.

    The refactor made the aggregators pure by threading bookkeeping that the
    panel path and the repeated-cross-section path populate differently (RCS
    makes several kit keys observation-length rather than unit-length), and
    ``anticipation`` shifts which ``(g, t)`` cells are eligible - so both need
    their own round-trip, not just the panel.
    """

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_repeated_cross_sections_match_fit_time(self, level):
        data = _rcs()
        post = CallawaySantAnna(panel=False).fit(data, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(panel=False).fit(data, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study", "total"])
    def test_anticipation_matches_fit_time(self, panel, level):
        post = CallawaySantAnna(anticipation=1).fit(panel, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(anticipation=1).fit(panel, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study", "total"])
    def test_universal_base_period_matches_fit_time(self, panel, level):
        """REGISTRY gives universal bases their own reference-cell and
        VCV-index semantics (a zero reference cell per cohort), which the
        default 'varying' fixture never exercises."""
        kw = dict(base_period="universal")
        post = CallawaySantAnna(**kw).fit(panel, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(**kw).fit(panel, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study", "total"])
    def test_not_yet_treated_with_anticipation_matches_fit_time(self, panel, level):
        """The not-yet-treated control group interacts with anticipation in
        picking each (g, t) comparison set - a different code path from the
        never-treated default."""
        kw = dict(control_group="not_yet_treated", anticipation=1)
        post = CallawaySantAnna(**kw).fit(panel, **FIT_KW)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            native = CallawaySantAnna(**kw).fit(panel, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_unbalanced_panel_matches_fit_time(self, panel, level):
        """allow_unbalanced_panel changes which units survive into the
        influence vectors, so the retained bookkeeping must agree with what
        the fit actually used."""
        thinned = panel.drop(panel.index[::13]).reset_index(drop=True)
        kw = dict(allow_unbalanced_panel=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            post = CallawaySantAnna(**kw).fit(thinned, **FIT_KW)
            native = CallawaySantAnna(**kw).fit(thinned, aggregate="all", **FIT_KW)
        _assert_level_matches(post, native, level)

    @pytest.mark.parametrize("level", ["simple", "group", "event_study"])
    def test_survey_numbers_match_fit_time(self, survey_fit, level):
        """Survey parity on the ESTIMATES, not just the df metadata."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            native = CallawaySantAnna().fit(
                _survey_panel(),
                survey_design=_survey_design(),
                aggregate="all",
                **FIT_KW,
            )
        _assert_level_matches(survey_fit, native, level)


def _assert_level_matches(post, native, level):
    """Compare a post-fit aggregation against the fit-time surface at 1e-14."""
    if level == "total":
        # Identity form (total == n x overall on the SAME fit) AND the
        # independent-mass form (n pinned against a frame-derived oracle from
        # the OTHER fit's public cells) - the identity alone passes for ANY C.
        got = post.aggregate("total")
        antic = getattr(native, "anticipation", 0)
        n_exp = sum(
            d["n_treated"]
            for (g, t), d in native.group_time_effects.items()
            if t >= g - antic and np.isfinite(d["effect"])
        )
        assert got.n[0] == float(n_exp)
        assert np.allclose(got.att[0], got.n[0] * native.overall_att, rtol=1e-12)
        assert np.allclose(got.se[0], got.n[0] * native.overall_se, rtol=1e-12)
        assert np.allclose(got.t_stat[0], native.overall_t_stat, rtol=1e-14, atol=0, equal_nan=True)
        return
    if level == "simple":
        got = post.aggregate("simple")
        assert np.allclose(got.att[0], native.overall_att, rtol=1e-14, atol=1e-14)
        assert np.allclose(got.se[0], native.overall_se, rtol=1e-14, atol=1e-14)
        return

    frame = post.aggregate(level).to_dataframe()
    if level == "group":
        expected, key_col = native.group_effects, "label"
    else:
        expected, key_col = native.event_study_effects, "event_time"
    compared = 0
    for _, row in frame.iterrows():
        if level == "event_study" and bool(row["is_reference"]):
            continue
        ref = expected[row[key_col]]
        for col, key in (("att", "effect"), ("se", "se"), ("p_value", "p_value")):
            assert np.allclose(
                row[col], ref[key], rtol=1e-14, atol=1e-14, equal_nan=True
            ), f"{level}[{row[key_col]}].{col} drifted"
        compared += 1
    assert compared > 0, f"no {level} rows compared"


def _survey_panel():
    rng = np.random.default_rng(3)
    rows = []
    for u in range(80):
        g = [0, 4, 6][u % 3]
        for t in range(1, 9):
            treated = g != 0 and t >= g
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "first_treat": g,
                    "psu": u % 20,
                    "stratum": u % 4,
                    "w": 1.0 + (u % 5) * 0.1,
                    "y": 0.3 * t + (2.0 if treated else 0.0) + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def _survey_design():
    from diff_diff import SurveyDesign

    return SurveyDesign(weights="w", strata="stratum", psu="psu")


@pytest.fixture(scope="module")
def survey_fit():
    """An EXPLICIT survey design - the branch where ``df_inference`` is
    intentionally None and ``survey_metadata.df_survey`` is the real carrier."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return CallawaySantAnna().fit(_survey_panel(), survey_design=_survey_design(), **FIT_KW)


class TestInferenceDfProvenance:
    """``df`` is per-row PROVENANCE - the df that actually produced the stored
    p-value/CI - so it must come from the carrier that governed inference.

    Regression: ``simple`` read ``df_inference`` (documented to stay None on
    explicit ``survey_design=`` fits) and ``group`` read ``event_study_df`` (a
    DIFFERENT aggregation's df, and None after a plain fit). Both reported
    ``df=NaN`` - implying normal/undefined inference - while the interval they
    carried was built on a finite t-reference.
    """

    def test_explicit_survey_df_reaches_every_level(self, survey_fit):
        """The canonical carrier is ``survey_metadata.df_survey``; on an
        explicit survey design ``df_inference`` is intentionally None."""
        expected = float(survey_fit.survey_metadata.df_survey)
        assert survey_fit.df_inference is None, "fixture no longer exercises the bug"
        for level in ("simple", "group", "event_study"):
            df_col = np.asarray(survey_fit.aggregate(level).df, dtype=float)
            finite = df_col[np.isfinite(df_col)]
            assert finite.size > 0, f"{level} reported no df at all"
            assert np.all(finite == expected), f"{level} df {finite} != {expected}"

    @pytest.mark.parametrize("survey", [False, True])
    def test_bootstrap_clears_group_df_provenance(self, survey):
        """When bootstrap replaces a group row's se/p/CI with percentile values,
        the retained analytical df described inference that no longer exists.
        Leaving it finite would claim a t-reference governed percentile-bootstrap
        numbers - the same false provenance the event-study path already clears.
        """
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(7)
        rows = []
        for u in range(90):
            g = [0, 4, 6][u % 3]
            for t in range(1, 9):
                treated = g != 0 and t >= g
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": g,
                        "psu": u % 18,
                        "stratum": u % 3,
                        "w": 1.0 + (u % 4) * 0.1,
                        "y": 0.3 * t + (2.0 if treated else 0.0) + rng.normal(0, 0.5),
                    }
                )
        # cluster= is a CONSTRUCTOR argument; survey_design= is a fit() argument.
        ctor = {} if survey else {"cluster": "psu"}
        fit_extra = (
            {"survey_design": SurveyDesign(weights="w", strata="stratum", psu="psu")}
            if survey
            else {}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(n_bootstrap=50, seed=1, **ctor).fit(
                pd.DataFrame(rows), aggregate="all", **fit_extra, **FIT_KW
            )
        assert res.group_effects, "fixture produced no group rows"
        for g, eff in res.group_effects.items():
            assert eff.get("df_used") is None, f"group {g} kept analytical df on a bootstrap fit"
        # The event-study path's equivalent clearing must not have regressed.
        assert res.event_study_df is None

    @pytest.mark.parametrize("survey", [False, True])
    def test_sddd_bootstrap_clears_group_df_provenance(self, survey):
        """StaggeredTripleDifference carries its OWN copy of the bootstrap
        group-replacement loop, so the clearing rule has to hold there too -
        fixing one twin and testing only the other is how this class of bug
        survived in the first place.

        Coverage is ASYMMETRIC and deliberately so: only ``survey=True``
        exercises the clearing (verified by reverting the fix - just that case
        fails). SDDD's plain-``cluster`` path never resolves a finite analytical
        df, so ``df_used`` is already None there and nothing needs clearing. The
        non-survey case is kept as a cheap guard in case that ever changes, not
        because it currently reproduces the bug. The CallawaySantAnna
        equivalent above DOES fail in both configurations.
        """
        from diff_diff import SurveyDesign

        rng = np.random.default_rng(9)
        rows = []
        for u in range(96):
            g = [0, 3, 4][u % 3]
            elig = u % 2
            for t in range(1, 7):
                treated = g != 0 and t >= g and elig == 1
                rows.append(
                    {
                        "unit": u,
                        "period": t,
                        "first_treat": g,
                        "eligibility": elig,
                        "psu": u % 16,
                        "stratum": u % 4,
                        "w": 1.0 + (u % 3) * 0.1,
                        "outcome": 0.3 * t + (1.2 if treated else 0.0) + rng.normal(0, 0.2),
                    }
                )
        ctor = {} if survey else {"cluster": "psu"}
        fit_extra = (
            {"survey_design": SurveyDesign(weights="w", strata="stratum", psu="psu")}
            if survey
            else {}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = StaggeredTripleDifference(n_bootstrap=40, seed=3, **ctor).fit(
                pd.DataFrame(rows),
                "outcome",
                "unit",
                "period",
                "first_treat",
                "eligibility",
                aggregate="all",
                **fit_extra,
            )
        assert res.group_effects, "fixture produced no SDDD group rows"
        for g, eff in res.group_effects.items():
            assert eff.get("df_used") is None, f"SDDD group {g} kept analytical df on bootstrap"

    def test_group_df_is_not_the_event_study_df(self, fitted):
        """``group`` must not borrow ``event_study_df``: on a plain fit that
        field is None, and it is a different aggregation's denominator."""
        assert fitted.event_study_df is None
        grp = fitted.aggregate("group")
        assert np.asarray(grp.df).shape == np.asarray(grp.att).shape


class TestContainerNormalization:
    """``AggregationResult`` normalizes its own columns; doing so must not
    reach back into caller-owned memory or mask a shape error."""

    @staticmethod
    def _kw():
        return dict(
            level="group",
            label=np.array(["a", "b"], dtype=object),
            target=np.array(["att", "att"], dtype=object),
            att=np.array([1.0, 2.0]),
            se=np.array([0.1, 0.2]),
            t_stat=np.array([10.0, np.nan]),
            p_value=np.array([0.01, np.nan]),
            conf_int_lower=np.array([0.8, np.nan]),
            conf_int_upper=np.array([1.2, np.nan]),
            n=np.array([5.0, 6.0]),
        )

    def test_df_input_is_not_mutated(self):
        """The NaN-out of non-estimable rows wrote THROUGH to the caller's
        array when df was normalized with ``asarray`` instead of ``array``."""
        caller = np.array([10.0, 20.0])
        AggregationResult(df=caller, **self._kw())
        assert np.array_equal(caller, [10.0, 20.0]), "caller's df array was mutated"

    def test_read_only_df_is_accepted(self):
        frozen = np.array([10.0, 20.0])
        frozen.flags.writeable = False
        got = AggregationResult(df=frozen, **self._kw())
        assert np.isnan(got.df[1]), "non-estimable row should still be NaN'd"

    def test_zero_dim_label_raises_value_error(self):
        """Not IndexError: the shape check must precede the shape read."""
        with pytest.raises(ValueError, match="one-dimensional"):
            AggregationResult(
                level="simple",
                label=np.array("x", dtype=object),
                target=np.array(["att"], dtype=object),
                att=np.array([1.0]),
                se=np.array([0.1]),
                t_stat=np.array([1.0]),
                p_value=np.array([0.1]),
                conf_int_lower=np.array([0.0]),
                conf_int_upper=np.array([2.0]),
                n=np.array([5.0]),
                df=None,
            )

    def test_off_vocabulary_n_kind_raises(self):
        """``n_kind`` is a routing key shared with EventStudyResults, so an
        unknown value is a contract break rather than a free-form label."""
        with pytest.raises(ValueError, match="vocabulary"):
            AggregationResult(df=None, n_kind="widgets", **self._kw())

    @pytest.mark.parametrize(
        "level,expected", [("simple", "units"), ("group", "cells"), ("total", "obs")]
    )
    def test_cs_n_kinds_are_in_the_shared_vocabulary(self, fitted, level, expected):
        from diff_diff.results_base import N_KIND_VOCABULARY

        got = fitted.aggregate(level)
        assert got.n_kind == expected
        assert got.n_kind in N_KIND_VOCABULARY

    def test_rcs_simple_reports_obs_not_units(self):
        """On panel=False, fit() counts ROWS (there is no unit tracking), so
        labelling that total "units" would misdescribe the sample - the exact
        conflation the shared vocabulary forbids."""
        data = _rcs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = CallawaySantAnna(panel=False).fit(data, **FIT_KW)
        got = res.aggregate("simple")
        assert got.n_kind == "obs"
        assert got.n[0] == float(res.n_treated_units + res.n_control_units)
        # The panel fixture must still say units, or this would be a blanket rename.
        panel_res = CallawaySantAnna().fit(_panel(), **FIT_KW)
        assert panel_res.aggregate("simple").n_kind == "units"

    def test_shared_vocabulary_is_enforced_on_both_containers(self, fitted):
        """The vocabulary is declared SHARED, so validating it on only one
        container would let an unknown value reach a consumer through the
        unchecked side. Every value a real producer emits must still pass.
        """
        from diff_diff.results_base import N_KIND_VOCABULARY

        base = dict(
            event_time=np.array([-1, 0, 1]),
            att=np.array([0.0, 1.0, 2.0]),
            se=np.array([0.1, 0.2, 0.3]),
            t_stat=np.array([0.0, 5.0, 6.0]),
            p_value=np.array([1.0, 0.01, 0.01]),
            conf_int_lower=np.array([-0.2, 0.6, 1.4]),
            conf_int_upper=np.array([0.2, 1.4, 2.6]),
            is_reference=np.array([True, False, False]),
            n=np.array([np.nan, 4.0, 4.0]),
        )
        with pytest.raises(ValueError, match="vocabulary"):
            EventStudyResults(n_kind="widgets", **base)
        # Every value shipped producers actually emit stays constructible.
        for kind in ("groups", "switcher_cells", "cells", "units", "obs", None):
            assert EventStudyResults(n_kind=kind, **base).n_kind == kind
        assert set(N_KIND_VOCABULARY) >= {"groups", "switcher_cells", "cells", "units", "obs"}

    def test_summary_refuses_to_relabel_a_stored_interval(self, fitted):
        """summary(alpha=) never recomputes, so printing the passed alpha would
        assert a confidence level the stored interval was not built at. Raises
        instead, matching EventStudyResults.summary."""
        got = fitted.aggregate("group")
        assert f"alpha={got.alpha}" in got.summary()
        assert f"alpha={got.alpha}" in got.summary(alpha=got.alpha)
        with pytest.raises(ValueError, match="never recomputes"):
            got.summary(alpha=0.10)


# --------------------------------------------------------------------------- #
# Cross-estimator regression
# --------------------------------------------------------------------------- #


def test_staggered_triple_diff_overall_att_es_still_works():
    """StaggeredTripleDifference inherits the CS aggregation mixin and is the
    ONLY reader of the Eq. 4.14 overall, which the purity refactor converted
    from a self-attribute to a returned value."""
    rng = np.random.default_rng(5)
    rows = []
    for u in range(80):
        g = [0, 3, 4][u % 3]
        elig = u % 2
        for t in range(1, 6):
            treated = g != 0 and t >= g and elig == 1
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "first_treat": g,
                    "eligibility": elig,
                    "outcome": 0.3 * t + (1.2 if treated else 0.0) + rng.normal(0, 0.2),
                }
            )
    data = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = StaggeredTripleDifference().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            "eligibility",
            aggregate="event_study",
        )
    assert res.overall_att_es is not None
    assert np.isfinite(res.overall_att_es)


# --------------------------------------------------------------------------- #
# dCDH (row M-026): fit(aggregate=) shim + the VIEW-based aggregate()
# --------------------------------------------------------------------------- #

DCDH_KW = dict(outcome="outcome", unit="unit", time="period", treatment="treat")


def _dcdh_panel(seed=5, n_units=40, n_periods=6, switch_t=4):
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(n_units):
        s_t = switch_t if u < n_units // 2 else 10**6
        for t in range(1, n_periods + 1):
            d = 1 if t >= s_t else 0
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "outcome": u / 10 + 0.2 * t + 1.5 * d + rng.randn() * 0.3,
                    "treat": d,
                }
            )
    return pd.DataFrame(rows)


def _dcdh_survey_panel(seed=7, n_units=60, n_periods=6, switch_t=4):
    df = _dcdh_panel(seed=seed, n_units=n_units, n_periods=n_periods, switch_t=switch_t)
    df["survey_weights"] = 1.0 + 0.1 * (df["unit"] % 5)
    df["strata"] = df["unit"] % 4
    df["psu"] = df["unit"]
    return df


def _fit_dcdh(data, *, est_kw=None, **fit_kw):
    from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ChaisemartinDHaultfoeuille(**(est_kw or {})).fit(data, **DCDH_KW, **fit_kw)


@pytest.fixture(scope="module")
def dcdh_panel():
    return _dcdh_panel()


@pytest.fixture(scope="module")
def dcdh_fitted(dcdh_panel):
    """Phase-1 fit (L_max=None)."""
    return _fit_dcdh(dcdh_panel)


class TestDcdhShim:
    def test_plain_fit_does_not_warn(self, dcdh_panel):
        from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ChaisemartinDHaultfoeuille().fit(dcdh_panel, **DCDH_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, dcdh_panel):
        from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            ChaisemartinDHaultfoeuille().fit(dcdh_panel, aggregate=None, **DCDH_KW)

    def test_non_none_value_warns_then_raises(self, dcdh_panel):
        from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille

        with pytest.warns(FutureWarning, match="aggregate"):
            with pytest.raises(ValueError, match=r"results\.aggregate"):
                ChaisemartinDHaultfoeuille().fit(dcdh_panel, aggregate="event_study", **DCDH_KW)

    def test_wrapper_forwarded_aggregate_warns(self, dcdh_panel):
        # chaisemartin_dhaultfoeuille() splits **kwargs by signature and
        # forwards non-__init__ names into fit(), so the shim is reachable
        # through the wrapper too.
        from diff_diff.chaisemartin_dhaultfoeuille import chaisemartin_dhaultfoeuille

        # Both warnings fire since 2(d) PR-A: the wrapper deprecation
        # (M-077) plus the forwarded fit(aggregate=) shim.
        with pytest.warns(FutureWarning) as record:
            chaisemartin_dhaultfoeuille(
                dcdh_panel,
                outcome="outcome",
                group="unit",
                time="period",
                treatment="treat",
                aggregate=None,
            )
        msgs = [str(w.message) for w in record]
        assert any("chaisemartin_dhaultfoeuille() is deprecated" in m for m in msgs), msgs
        assert any(re.search(r"fit\(aggregate=\) is deprecated", m) for m in msgs), msgs


class TestDcdhAggregate:
    def _assert_surface_matches_builder(self, res):
        from diff_diff.results_base import build_event_study_surface

        es = res.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        built = build_event_study_surface(res)
        # The dataclass's generated == raises on ndarray fields; compare
        # to_dataframe rows per the file's precedent.
        a, b = es.to_dataframe(), built.to_dataframe()
        assert list(a.columns) == list(b.columns)
        assert a.shape == b.shape
        for col in a.columns:
            av, bv = a[col].to_numpy(), b[col].to_numpy()
            if av.dtype.kind in "fc":
                assert np.allclose(av.astype(float), bv.astype(float), equal_nan=True)
            else:
                assert list(av) == list(bv)
        return es

    def test_simple_view_bit_exact_phase1(self, dcdh_fitted):
        agg = dcdh_fitted.aggregate("simple")
        assert isinstance(agg, AggregationResult)
        assert agg.level == "simple"
        assert list(agg.label) == ["overall"]
        assert list(agg.target) == ["DID_M"]
        assert float(agg.att[0]) == dcdh_fitted.overall_att
        assert float(agg.se[0]) == dcdh_fitted.overall_se
        assert float(agg.t_stat[0]) == dcdh_fitted.overall_t_stat
        assert float(agg.p_value[0]) == dcdh_fitted.overall_p_value
        assert float(agg.conf_int_lower[0]) == dcdh_fitted.overall_conf_int[0]
        assert float(agg.conf_int_upper[0]) == dcdh_fitted.overall_conf_int[1]
        assert float(agg.n[0]) == float(dcdh_fitted.n_switcher_cells)
        assert agg.n_kind == "switcher_cells"
        # Non-survey analytical inference is z-based: no df.
        assert np.isnan(agg.df[0])
        assert agg.estimator == "ChaisemartinDHaultfoeuille"

    def test_simple_view_lmax1_groups(self, dcdh_panel):
        res = _fit_dcdh(dcdh_panel, L_max=1)
        agg = res.aggregate("simple")
        assert list(agg.target) == ["DID_1"]
        assert agg.n_kind == "groups"
        assert float(agg.n[0]) == float(res.n_switcher_cells)

    def test_simple_view_lmax2_delta(self, dcdh_panel):
        res = _fit_dcdh(dcdh_panel, L_max=2)
        agg = res.aggregate("simple")
        assert list(agg.target) == ["delta"]
        # The delta averages horizon-specific N_l: no truthful scalar count.
        assert np.isnan(agg.n[0])
        assert agg.n_kind is None
        assert float(agg.att[0]) == res.overall_att

    def test_simple_view_trends_linear_all_nan_relay(self, dcdh_panel):
        # trends_linear + L_max>=2 suppresses the delta by design: every
        # overall_* field is NaN and the estimand label points at
        # linear_trends_effects. The view relays the all-NaN row honestly.
        res = _fit_dcdh(dcdh_panel, L_max=2, trends_linear=True)
        agg = res.aggregate("simple")
        assert "fd" in str(agg.target[0])
        assert np.isnan(agg.att[0])
        assert np.isnan(agg.se[0])
        assert np.isnan(agg.p_value[0])
        assert np.isnan(agg.conf_int_lower[0]) and np.isnan(agg.conf_int_upper[0])
        assert np.isnan(agg.df[0])

    def test_simple_view_bootstrap_percentile_relay(self, dcdh_panel):
        # Bootstrap fits are PERMITTED (pure view): the row relays the
        # stored percentile-bootstrap inference; df is NaN (no df used).
        res = _fit_dcdh(dcdh_panel, est_kw=dict(n_bootstrap=49, seed=3))
        agg = res.aggregate("simple")
        assert float(agg.att[0]) == res.overall_att
        assert float(agg.se[0]) == res.overall_se
        assert np.isnan(agg.df[0])

    def test_event_study_container_threads_survey_df(self):
        # CQ1 (local review R3): the dCDH builder threads the scalar
        # df_survey provenance too - a survey fit's container must carry
        # survey_metadata.df_survey, not None.
        from diff_diff.survey import SurveyDesign

        df = _dcdh_survey_panel()
        sd = SurveyDesign(weights="survey_weights", strata="strata", psu="psu")
        res = _fit_dcdh(df, L_max=2, survey_design=sd)
        surface = res.aggregate("event_study")
        assert res.survey_metadata is not None
        assert surface.df_survey == float(res.survey_metadata.df_survey)

    def test_simple_view_survey_analytical_df(self):
        # Analytical survey fit: the stored p/CI used the survey df; the
        # view relays it (event_study_df carries it here).
        from diff_diff.survey import SurveyDesign

        df = _dcdh_survey_panel()
        sd = SurveyDesign(weights="survey_weights", strata="strata", psu="psu")
        res = _fit_dcdh(df, L_max=2, survey_design=sd)
        agg = res.aggregate("simple")
        assert res.survey_metadata is not None
        expected = res.survey_metadata.df_survey
        assert expected is not None and np.isfinite(agg.df[0])
        assert float(agg.df[0]) == float(expected)

    def test_simple_view_lmax2_survey_bootstrap_finite_df(self):
        # THE df-provenance pin: under n_bootstrap>0 the event_study_df
        # channel is cleared, but the L_max>=2 delta's stored p/CI still
        # came from analytical safe_inference with the survey df (REGISTRY
        # Note, Phase 2 cost-benefit delta SE). The view must report that
        # finite df, not NaN.
        from diff_diff.survey import SurveyDesign

        df = _dcdh_survey_panel()
        sd = SurveyDesign(weights="survey_weights", strata="strata", psu="psu")
        res = _fit_dcdh(df, L_max=2, survey_design=sd, est_kw=dict(n_bootstrap=49, seed=3))
        assert res.event_study_df is None  # cleared under bootstrap
        assert np.isfinite(res.overall_p_value)  # delta stayed analytical
        agg = res.aggregate("simple")
        assert res.survey_metadata is not None
        assert float(agg.df[0]) == float(res.survey_metadata.df_survey)

    def test_event_study_view_phase1_two_rows(self, dcdh_fitted):
        es = self._assert_surface_matches_builder(dcdh_fitted)
        # Phase-1 (L_max=None): the 2-row l=1 view, l1 convention - NOT an
        # error (fit populates event_study_effects={1: ...} on this path).
        assert es.event_time.tolist() == [0, 1]
        assert es.event_time_convention == "l1_first_switch"
        assert es.n_kind == "switcher_cells"

    def test_event_study_view_multi_horizon(self, dcdh_panel):
        res = _fit_dcdh(dcdh_panel, L_max=2)
        es = self._assert_surface_matches_builder(res)
        assert 2 in es.event_time.tolist()
        assert es.n_kind == "groups"

    def test_balance_e_rejected_empty_vocabulary(self, dcdh_fitted):
        with pytest.raises(ValueError, match="no aggregation type on this estimator"):
            dcdh_fitted.aggregate("event_study", balance_e=1)

    def test_weights_rejected(self, dcdh_fitted):
        with pytest.raises(ValueError, match="does not accept a weights selector"):
            dcdh_fitted.aggregate("simple", weights="cell")

    @pytest.mark.parametrize("bad", ["group", "calendar", "all", "nonsense"])
    def test_unsupported_types_fail_closed(self, dcdh_fitted, bad):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            dcdh_fitted.aggregate(bad)

    def test_mixin_hooks_are_not_dataclass_fields(self):
        # Regression: on a dataclass results class, annotating the mixin
        # routing hooks without ClassVar turns them into __init__ fields,
        # widening the public constructor/repr/equality surface. Enforced
        # dynamically for EVERY dataclass that mixes AggregationMixin in,
        # so later 2(b) waves are enrolled automatically.
        import dataclasses
        import inspect

        import diff_diff
        from diff_diff.aggregation import AggregationMixin

        hooks = ("_AGGREGATE_SUPPORTED", "_AGGREGATE_BALANCE_E_TYPES")
        checked = []
        for name in dir(diff_diff):
            obj = getattr(diff_diff, name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, AggregationMixin)
                and obj is not AggregationMixin
                and dataclasses.is_dataclass(obj)
            ):
                checked.append(name)
                # dataclasses.fields() (not __dataclass_fields__, which
                # also lists ClassVar pseudo-fields) = the real
                # init/repr/eq surface.
                fields = {f.name for f in dataclasses.fields(obj)}
                params = inspect.signature(obj.__init__).parameters
                for hook in hooks:
                    assert hook not in fields, f"{name}.{hook} leaked into fields"
                    assert hook not in params, f"{name}.{hook} leaked into __init__"
        # The roster must at least cover the shipped mixin adopters.
        assert "CallawaySantAnnaResults" in checked
        assert "ChaisemartinDHaultfoeuilleResults" in checked
        assert "StackedDiDResults" in checked
        assert "EfficientDiDResults" in checked
        assert "ImputationDiDResults" in checked
        assert "TwoStageDiDResults" in checked
        assert "ContinuousDiDResults" in checked
        assert "HeterogeneousAdoptionDiDResults" in checked
        assert "HeterogeneousAdoptionDiDEventStudyResults" in checked


# --------------------------------------------------------------------------- #
# StackedDiD (row M-024): fit(aggregate=) shim + the VIEW-based aggregate()
# --------------------------------------------------------------------------- #

STACKED_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _stacked_panel(seed=42, n_units=120, n_periods=12):
    from diff_diff.prep_dgp import generate_staggered_data

    return generate_staggered_data(
        n_units=n_units, n_periods=n_periods, cohort_periods=[4, 6, 8], seed=seed
    )


def _fit_stacked(data, *, est_kw=None, **fit_kw):
    from diff_diff import StackedDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StackedDiD(**(est_kw or {"kappa_pre": 2, "kappa_post": 2})).fit(
            data, **STACKED_KW, **fit_kw
        )


@pytest.fixture(scope="module")
def stacked_panel():
    return _stacked_panel()


@pytest.fixture(scope="module")
def stacked_fitted(stacked_panel):
    """Plain hc1 fit - the surface is always materialized (M-024)."""
    return _fit_stacked(stacked_panel)


class TestStackedShim:
    def test_plain_fit_does_not_warn(self, stacked_panel):
        from diff_diff import StackedDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StackedDiD(kappa_pre=2, kappa_post=2).fit(stacked_panel, **STACKED_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, stacked_panel):
        from diff_diff import StackedDiD

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            StackedDiD(kappa_pre=2, kappa_post=2).fit(stacked_panel, aggregate=None, **STACKED_KW)

    def test_deprecated_value_warns_and_still_works(self, stacked_panel, stacked_fitted):
        # CS-style warn-and-still-work (the param genuinely worked here,
        # unlike dCDH's raise): the deprecated path returns an object whose
        # surface equals a plain fit's - the surface is always computed.
        from diff_diff import StackedDiD

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            res = StackedDiD(kappa_pre=2, kappa_post=2).fit(
                stacked_panel, aggregate="event_study", **STACKED_KW
            )
        assert res.overall_att == stacked_fitted.overall_att
        assert res.event_study_effects is not None
        assert sorted(res.event_study_effects) == sorted(stacked_fitted.event_study_effects)
        np.testing.assert_array_equal(res.event_study_vcov, stacked_fitted.event_study_vcov)

    def test_wrapper_forwarded_aggregate_warns(self, stacked_panel):
        # stacked_did() declares aggregate explicitly with its own sentinel
        # default and forwards verbatim into fit().
        from diff_diff.stacked_did import stacked_did

        # Both warnings fire since 2(d) PR-A (M-072 + the forwarded shim).
        with pytest.warns(FutureWarning) as record:
            stacked_did(
                stacked_panel,
                "outcome",
                "unit",
                "period",
                "first_treat",
                kappa_pre=2,
                kappa_post=2,
                aggregate="simple",
            )
        msgs = [str(w.message) for w in record]
        assert any("stacked_did() is deprecated" in m for m in msgs), msgs
        assert any(re.search(r"fit\(aggregate=\) is deprecated", m) for m in msgs), msgs

    def test_plain_wrapper_call_fires_only_wrapper_warning(self, stacked_panel):
        # Flipped BY DESIGN in the 2(d) PR-A (M-072): the wrapper itself
        # now emits its deprecation FutureWarning, but the sentinel
        # forwarding is unchanged - a plain wrapper call must never fire
        # the fit-time aggregate warning, so EXACTLY ONE FutureWarning.
        from diff_diff.stacked_did import stacked_did

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = stacked_did(
                stacked_panel,
                "outcome",
                "unit",
                "period",
                "first_treat",
                kappa_pre=2,
                kappa_post=2,
            )
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1, [str(w.message) for w in fw]
        assert "stacked_did() is deprecated" in str(fw[0].message)
        assert "StackedDiD.fit" not in str(fw[0].message)
        assert res.event_study_effects is not None

    def test_group_warns_then_raises_educational_error(self, stacked_panel):
        from diff_diff import StackedDiD

        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            with pytest.raises(ValueError, match="not supported by StackedDiD"):
                StackedDiD(kappa_pre=2, kappa_post=2).fit(
                    stacked_panel, aggregate="group", **STACKED_KW
                )


class TestStackedAggregate:
    def _assert_surface_matches_builder(self, res):
        from diff_diff.results_base import build_event_study_surface

        es = res.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        built = build_event_study_surface(res)
        a, b = es.to_dataframe(), built.to_dataframe()
        assert list(a.columns) == list(b.columns)
        assert a.shape == b.shape
        for col in a.columns:
            av, bv = a[col].to_numpy(), b[col].to_numpy()
            if av.dtype.kind in "fc":
                assert np.allclose(av.astype(float), bv.astype(float), equal_nan=True)
            else:
                assert list(av) == list(bv)
        return es

    def test_event_study_view_matches_builder(self, stacked_fitted):
        es = self._assert_surface_matches_builder(stacked_fitted)
        # kappa 2/2 grid: {-2, ref -1, 0, 1, 2}; n_kind from n_obs cells.
        assert sorted(es.event_time.tolist()) == [-2, -1, 0, 1, 2]
        assert es.n_kind == "obs"
        assert es.base_period == "universal"
        assert es.reference_event_times == (-1,)

    def test_event_study_view_does_not_alias_fit_vcov(self, stacked_fitted):
        es = stacked_fitted.aggregate("event_study")
        assert not np.shares_memory(es.vcov, stacked_fitted.event_study_vcov)
        # int labels survive to_dict (no float coercion of the index)
        assert es.to_dict()["vcov_index"] == [-2, 0, 1, 2]

    def test_simple_bit_exact_relay_hc1(self, stacked_fitted):
        agg = stacked_fitted.aggregate("simple")
        assert isinstance(agg, AggregationResult)
        assert agg.level == "simple"
        assert list(agg.label) == ["overall"]
        # target is "att": overall_att is the equally-weighted post-period
        # average, NOT the per-event trimmed aggregate ATT (M-024 Note).
        assert list(agg.target) == ["att"]
        assert float(agg.att[0]) == stacked_fitted.overall_att
        assert float(agg.se[0]) == stacked_fitted.overall_se
        assert float(agg.t_stat[0]) == stacked_fitted.overall_t_stat
        assert float(agg.p_value[0]) == stacked_fitted.overall_p_value
        assert (float(agg.conf_int_lower[0]), float(agg.conf_int_upper[0])) == tuple(
            stacked_fitted.overall_conf_int
        )
        # Treated-unit count: control units OVERLAP treated across
        # sub-experiments, so no disjoint total exists (M-024 Note).
        assert float(agg.n[0]) == float(stacked_fitted.n_treated_units)
        assert agg.n_kind == "units"
        assert float(agg.df[0]) == float(stacked_fitted.inference_df)
        assert agg.estimator == "StackedDiD"

    @pytest.mark.parametrize("weighting", ["aggregate", "population", "sample_share"])
    def test_simple_target_att_on_all_weighting_schemes(self, stacked_panel, weighting):
        fit_kw = {}
        panel = stacked_panel
        if weighting == "population":
            panel = stacked_panel.copy()
            panel["pop"] = 100.0 + (panel["unit"] % 7)
            fit_kw["population"] = "pop"
        res = _fit_stacked(
            panel,
            est_kw={"kappa_pre": 2, "kappa_post": 2, "weighting": weighting},
            **fit_kw,
        )
        agg = res.aggregate("simple")
        assert list(agg.target) == ["att"]
        assert agg.n_kind == "units"

    def test_simple_relay_hc2_bm_df_is_overall_bm_dof(self, stacked_panel):
        res = _fit_stacked(
            stacked_panel,
            est_kw={
                "kappa_pre": 2,
                "kappa_post": 2,
                "vcov_type": "hc2_bm",
                "cluster": "unit",
            },
        )
        agg = res.aggregate("simple")
        assert np.isfinite(agg.df[0])
        assert float(agg.df[0]) == float(res.inference_df)
        # per-row BM dfs are present WITHOUT fit-time aggregate (M-024)
        assert res.event_study_df is not None
        assert all(np.isfinite(v) for v in res.event_study_df.values())

    def test_simple_relay_bm_failure_nan_inference(self, stacked_panel, monkeypatch):
        # The hc2_bm fail-closed state (finite att/se, jointly-NaN t/p/CI,
        # inference_df None) must RELAY through the simple view - the df
        # comparison is np.isnan, never df == inference_df (nan == None).
        import diff_diff.linalg as dl

        def _nan_dof(X, cluster_ids, bread_matrix, contrasts, weights=None):
            return np.full(contrasts.shape[1], np.nan)

        monkeypatch.setattr(dl, "_compute_cr2_bm_contrast_dof", _nan_dof)
        res = _fit_stacked(
            stacked_panel,
            est_kw={
                "kappa_pre": 2,
                "kappa_post": 2,
                "vcov_type": "hc2_bm",
                "cluster": "unit",
            },
        )
        assert res.inference_df is None
        agg = res.aggregate("simple")
        assert np.isfinite(agg.att[0]) and np.isfinite(agg.se[0])
        assert np.isnan(agg.t_stat[0])
        assert np.isnan(agg.p_value[0])
        assert np.isnan(agg.conf_int_lower[0]) and np.isnan(agg.conf_int_upper[0])
        assert np.isnan(agg.df[0])

    def test_simple_relay_survey_tsl_df(self, stacked_panel):
        from diff_diff.survey import SurveyDesign

        panel = stacked_panel.copy()
        panel["w"] = 1.0 + 0.1 * (panel["unit"] % 5)
        panel["strata"] = panel["unit"] % 4
        panel["psu"] = panel["unit"]
        res = _fit_stacked(
            panel,
            survey_design=SurveyDesign(weights="w", strata="strata", psu="psu"),
        )
        agg = res.aggregate("simple")
        assert float(agg.att[0]) == res.overall_att
        # The stored overall inference used the survey df; the relay
        # carries exactly that provenance.
        assert float(agg.df[0]) == float(res.inference_df)

    def test_bm_dof_batch_parity_and_overall_reconstruction(self, stacked_panel, monkeypatch):
        # Pin (a): per-contrast Satterthwaite dof is column-independent in
        # VALUE on a well-conditioned design - the m=1 evaluation of each
        # contrast equals its column in the batched call (the batch-relative
        # noise floor changes only the DEGENERACY GUARD's scale, documented
        # in the REGISTRY M-024 Note).
        # Pin (b): the pre-change PLAIN-fit overall inference (m=1 batch) is
        # reconstructed in-process from the spy-captured fit-time locals and
        # must match the post-change stored overall inference at 1e-14.
        import diff_diff.linalg as dl
        from diff_diff.utils import safe_inference

        real = dl._compute_cr2_bm_contrast_dof
        captured = {}

        def spy(X, cluster_ids, bread_matrix, contrasts, weights=None):
            captured.update(
                X=X,
                cluster_ids=cluster_ids,
                bread=bread_matrix,
                contrasts=contrasts,
                weights=weights,
            )
            return real(X, cluster_ids, bread_matrix, contrasts, weights=weights)

        monkeypatch.setattr(dl, "_compute_cr2_bm_contrast_dof", spy)
        res = _fit_stacked(
            stacked_panel,
            est_kw={
                "kappa_pre": 2,
                "kappa_post": 2,
                "vcov_type": "hc2_bm",
                "cluster": "unit",
            },
        )
        assert captured, "spy never fired"
        contrasts = captured["contrasts"]
        batched = real(
            captured["X"],
            captured["cluster_ids"],
            captured["bread"],
            contrasts,
            weights=captured["weights"],
        )
        for j in range(contrasts.shape[1]):
            single = real(
                captured["X"],
                captured["cluster_ids"],
                captured["bread"],
                contrasts[:, [j]],
                weights=captured["weights"],
            )
            np.testing.assert_allclose(single[0], batched[j], rtol=1e-14)
        # (b) overall contrast is appended LAST at the fit site; its m=1
        # dof + safe_inference reproduce the stored overall inference.
        overall_dof_m1 = float(
            real(
                captured["X"],
                captured["cluster_ids"],
                captured["bread"],
                contrasts[:, [-1]],
                weights=captured["weights"],
            )[0]
        )
        np.testing.assert_allclose(res.inference_df, overall_dof_m1, rtol=1e-14)
        t, p, ci = safe_inference(
            res.overall_att, res.overall_se, alpha=res.alpha, df=overall_dof_m1
        )
        np.testing.assert_allclose(
            [t, ci[0], ci[1]],
            [
                res.overall_t_stat,
                res.overall_conf_int[0],
                res.overall_conf_int[1],
            ],
            rtol=1e-14,
        )
        # The p-value gets a looser rtol: batched-vs-m=1 BLAS kernels differ
        # at ~1 ULP in the dof (platform-dependent - observed on Linux
        # x86 OpenBLAS), and the deep tail amplifies that relative noise by
        # ~t^2 (d ln p ~ -t dt; t ~ 29 here => ~1e3x), so p carries ~1e-13
        # relative noise while t/CI stay at 1e-14.
        np.testing.assert_allclose(p, res.overall_p_value, rtol=1e-12)

    def test_legacy_pickle_absent_surface_hint(self, stacked_fitted):
        import dataclasses

        legacy = dataclasses.replace(
            stacked_fitted,
            event_study_effects=None,
            event_study_vcov=None,
            event_study_vcov_index=None,
            event_study_df=None,
        )
        with pytest.raises(ValueError, match=r"diff-diff >= 3\.9"):
            legacy.aggregate("event_study")

    def test_balance_e_rejected_empty_vocabulary(self, stacked_fitted):
        with pytest.raises(ValueError, match="no aggregation type on this estimator"):
            stacked_fitted.aggregate("event_study", balance_e=1)

    def test_weights_rejected(self, stacked_fitted):
        with pytest.raises(ValueError, match="does not accept a weights selector"):
            stacked_fitted.aggregate("simple", weights="cell")

    @pytest.mark.parametrize("bad", ["group", "calendar", "all", "nonsense"])
    def test_unsupported_types_fail_closed(self, stacked_fitted, bad):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            stacked_fitted.aggregate(bad)


# --------------------------------------------------------------------------- #
# EfficientDiD (rows M-023/M-120): fit(aggregate=/balance_e=) shim + the
# lazy-KIT recompute aggregate() (the CallawaySantAnna class)
# --------------------------------------------------------------------------- #

EFFICIENT_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _efficient_panel(seed=42, n_units=120, n_periods=8):
    from diff_diff.prep_dgp import generate_staggered_data

    return generate_staggered_data(
        n_units=n_units, n_periods=n_periods, cohort_periods=[4, 6], seed=seed
    )


def _efficient_clustered_panel(seed=11):
    # Units nested 3-per-cluster (the sibling test_efficient_did.py shape).
    d = _efficient_panel(seed=seed)
    d = d.copy()
    d["cl"] = (d["unit"] // 3).astype(int)
    return d


def _efficient_survey_panel(seed=5, replicate=False, degenerate=None):
    """Unit-constant pweights; optionally a JK replicate design.

    degenerate: None | "dropped" (2 all-zero replicate columns ->
    n_valid = n_replicates - 2) | "undefined" (all but one column all-zero
    -> n_valid <= 1 -> working df None, the load-bearing degenerate state).
    """
    import numpy as np

    d = _efficient_panel(seed=seed).copy()
    rng = np.random.default_rng(seed)
    wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
    d["w"] = d["unit"].map(wmap)
    rep_cols = []
    if replicate:
        n_rep = 8
        for r in range(n_rep):
            col = f"rw{r}"
            rep_cols.append(col)
            if degenerate == "dropped" and r >= n_rep - 2:
                d[col] = 0.0
            elif degenerate == "undefined" and r >= 1:
                d[col] = 0.0
            else:
                jitter = {u: rng.uniform(0.0, 2.0) for u in d["unit"].unique()}
                d[col] = d["unit"].map(jitter) * d["w"]
    return d, rep_cols


def _efficient_survey_design(rep_cols=None):
    from diff_diff import SurveyDesign

    if rep_cols:
        return SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
    return SurveyDesign(weights="w")


def _fit_efficient(data, *, est_kw=None, **fit_kw):
    from diff_diff import EfficientDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EfficientDiD(**(est_kw or {})).fit(data, **EFFICIENT_KW, **fit_kw)


@pytest.fixture(scope="module")
def efficient_panel():
    return _efficient_panel()


@pytest.fixture(scope="module")
def efficient_fitted(efficient_panel):
    """Plain fit - computes NOTHING extra; the kit powers aggregate()."""
    return _fit_efficient(efficient_panel)


@pytest.fixture(scope="module")
def efficient_fit_time(efficient_panel):
    """Deprecated fit-time aggregate="all" - the inertness reference."""
    return _fit_efficient(efficient_panel, aggregate="all")


class TestEfficientShim:
    def test_plain_fit_does_not_warn(self, efficient_panel):
        from diff_diff import EfficientDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EfficientDiD().fit(efficient_panel, **EFFICIENT_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, efficient_panel):
        from diff_diff import EfficientDiD

        with pytest.warns(FutureWarning, match=r"EfficientDiD\.fit\(aggregate=\)"):
            EfficientDiD().fit(efficient_panel, **EFFICIENT_KW, aggregate=None)

    def test_balance_e_kwarg_warns_alone(self, efficient_panel):
        from diff_diff import EfficientDiD

        with pytest.warns(FutureWarning, match=r"EfficientDiD\.fit\(balance_e=\)"):
            EfficientDiD().fit(efficient_panel, **EFFICIENT_KW, balance_e=None)

    def test_joint_supply_warns_once_naming_both(self, efficient_panel):
        from diff_diff import EfficientDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EfficientDiD().fit(
                efficient_panel, **EFFICIENT_KW, aggregate="event_study", balance_e=0
            )
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1
        msg = str(fw[0].message)
        assert "aggregate=" in msg and "balance_e=" in msg

    def test_unknown_string_still_acts_like_none(self, efficient_panel):
        # The legacy path performs NO value validation - an unknown string
        # behaves exactly like a plain fit (documented unchanged behavior;
        # the post-fit mixin is what fails closed on unknown types).
        res = _fit_efficient(efficient_panel, aggregate="nonsense")
        plain = _fit_efficient(efficient_panel)
        assert res.event_study_effects is None and res.group_effects is None
        assert res.overall_att == plain.overall_att

    def test_warn_and_still_work(self, efficient_panel, efficient_fitted, efficient_fit_time):
        # The deprecated path still populates the fit-time surfaces AND
        # equals the plain fit's post-fit recompute.
        assert efficient_fit_time.event_study_effects is not None
        assert efficient_fit_time.group_effects is not None
        es = efficient_fitted.aggregate("event_study")
        for e, row in efficient_fit_time.event_study_effects.items():
            i = list(es.event_time).index(e)
            np.testing.assert_allclose(row["effect"], es.att[i], rtol=1e-14)


class TestEfficientAggregate:
    def _assert_es_matches_fit_time(self, res, fit_time, balance_e=None, skip_reference=False):
        es = res.aggregate("event_study", balance_e=balance_e)
        native = fit_time.event_study_effects
        ref_marked = set()
        if skip_reference:
            ref_marked = {int(e) for e, m in zip(es.event_time, es.is_reference) if m}
        assert sorted(int(e) for e in es.event_time) == sorted(native)
        for e, row in native.items():
            i = list(es.event_time).index(e)
            if int(e) in ref_marked:
                # The container's reference normalization NaNs se/t/p/CI/n
                # and forces att 0.0 on the marked row - compare the NATIVE
                # dict instead (its anchor must be an exact mechanical zero).
                assert row["effect"] == 0.0 and row["se"] == 0.0
                assert es.att[i] == 0.0
                assert np.isnan(es.se[i]) and np.isnan(es.n[i])
                continue
            np.testing.assert_allclose(row["effect"], es.att[i], rtol=1e-14)
            np.testing.assert_allclose(row["se"], es.se[i], rtol=1e-14)
            if np.isfinite(row["p_value"]):
                np.testing.assert_allclose(row["p_value"], es.p_value[i], rtol=1e-14)
        return es

    def test_event_study_inertness(self, efficient_fitted, efficient_fit_time):
        self._assert_es_matches_fit_time(efficient_fitted, efficient_fit_time)

    def test_group_inertness(self, efficient_fitted, efficient_fit_time):
        grp = efficient_fitted.aggregate("group")
        for i, g in enumerate(grp.label):
            row = efficient_fit_time.group_effects[g]
            np.testing.assert_allclose(row["effect"], grp.att[i], rtol=1e-14)
            np.testing.assert_allclose(row["se"], grp.se[i], rtol=1e-14)
        assert grp.n_kind == "cells"
        assert grp.weight is None

    def test_simple_relay_bit_exact(self, efficient_fitted):
        simple = efficient_fitted.aggregate("simple")
        assert simple.att[0] == efficient_fitted.overall_att
        assert simple.se[0] == efficient_fitted.overall_se
        assert simple.t_stat[0] == efficient_fitted.overall_t_stat
        assert simple.p_value[0] == efficient_fitted.overall_p_value
        assert simple.conf_int_lower[0] == efficient_fitted.overall_conf_int[0]
        assert simple.conf_int_upper[0] == efficient_fitted.overall_conf_int[1]
        assert simple.target[0] == "att"
        # Disjoint treated+control total (last_cohort trimming reassigns
        # BEFORE the counts, so a true total exists - unlike StackedDiD).
        assert simple.n[0] == float(
            efficient_fitted.n_treated_units + efficient_fitted.n_control_units
        )
        assert simple.n_kind == "units"
        # Plain fit: the kit's post-overall df snapshot is None -> NaN df.
        assert np.isnan(simple.df[0])

    @pytest.mark.parametrize("balance_e", [0, 1, 2])
    def test_balance_e_inertness(self, efficient_panel, efficient_fitted, balance_e):
        fit_time = _fit_efficient(efficient_panel, aggregate="event_study", balance_e=balance_e)
        self._assert_es_matches_fit_time(efficient_fitted, fit_time, balance_e=balance_e)

    def test_cluster_inertness(self):
        d = _efficient_clustered_panel()
        plain = _fit_efficient(d, est_kw={"cluster": "cl"})
        fit_time = _fit_efficient(d, est_kw={"cluster": "cl"}, aggregate="all")
        self._assert_es_matches_fit_time(plain, fit_time)
        grp = plain.aggregate("group")
        for i, g in enumerate(grp.label):
            np.testing.assert_allclose(fit_time.group_effects[g]["se"], grp.se[i], rtol=1e-14)

    def test_survey_tsl_inertness_and_df(self):
        d, _ = _efficient_survey_panel()
        sd = _efficient_survey_design()
        plain = _fit_efficient(d, survey_design=sd)
        fit_time = _fit_efficient(d, survey_design=sd, aggregate="all")
        es = self._assert_es_matches_fit_time(plain, fit_time)
        # TSL survey: the post-overall snapshot is the finite survey df and
        # is provenance-exact on every row.
        assert es.df_survey == plain._aggregation_kit.bookkeeping["df_survey"]
        simple = plain.aggregate("simple")
        assert simple.df[0] == plain._aggregation_kit.bookkeeping["df_survey"]
        grp = plain.aggregate("group")
        # Non-circular oracle: safe_inference(att, se, df=row.df) reproduces
        # each stored group row's inference (a stale df would mismatch p).
        from diff_diff.utils import safe_inference

        for i in range(len(grp.label)):
            t, p, ci = safe_inference(
                float(grp.att[i]),
                float(grp.se[i]),
                alpha=plain.alpha,
                df=float(grp.df[i]),
            )
            np.testing.assert_allclose(t, grp.t_stat[i], rtol=1e-12)
            np.testing.assert_allclose(p, grp.p_value[i], rtol=1e-12)

    def test_replicate_inertness_healthy_and_dropped(self):
        from diff_diff.utils import safe_inference

        for degenerate, expected_df in ((None, 7.0), ("dropped", 5.0)):
            d, rep_cols = _efficient_survey_panel(replicate=True, degenerate=degenerate)
            sd = _efficient_survey_design(rep_cols)
            plain = _fit_efficient(d, survey_design=sd)
            fit_time = _fit_efficient(d, survey_design=sd, aggregate="all")
            # Inertness against the SAME object's stored fit-time surface.
            self._assert_es_matches_fit_time(plain, fit_time)
            grp = plain.aggregate("group")
            # Cross-FIT df contrast: all group rows share n_valid - 1.
            assert set(np.unique(grp.df[np.isfinite(grp.df)])) <= {expected_df}
            for i in range(len(grp.label)):
                if not np.isfinite(grp.df[i]):
                    continue
                t, p, ci = safe_inference(
                    float(grp.att[i]),
                    float(grp.se[i]),
                    alpha=plain.alpha,
                    df=float(grp.df[i]),
                )
                np.testing.assert_allclose(p, grp.p_value[i], rtol=1e-12)

    def test_replicate_undefined_df_degenerate(self):
        # n_valid <= 1: the working df degenerates to None mid-fit. The
        # post-overall snapshot carries that state; the container's scalar
        # df_survey resolves to the 0.0 replicate-undefined sentinel (the
        # shared resolver ladder), NOT the raw survey_metadata value the
        # is-not-None guard leaves finite - this arm is the discriminator
        # for the carrier's metadata copy.
        d, rep_cols = _efficient_survey_panel(replicate=True, degenerate="undefined")
        sd = _efficient_survey_design(rep_cols)
        plain = _fit_efficient(d, survey_design=sd)
        assert plain._aggregation_kit.bookkeeping["df_survey"] is None
        es = plain.aggregate("event_study")
        assert es.df_survey == 0.0
        fit_time = _fit_efficient(d, survey_design=sd, aggregate="all")
        self._assert_es_matches_fit_time(plain, fit_time)

    def test_pt_post_inertness_and_reference(self, efficient_panel):
        plain = _fit_efficient(efficient_panel, est_kw={"pt_assumption": "post"})
        fit_time = _fit_efficient(
            efficient_panel, est_kw={"pt_assumption": "post"}, aggregate="all"
        )
        es = self._assert_es_matches_fit_time(plain, fit_time, skip_reference=True)
        # Exactly one is_reference row, at the materialized mechanical anchor.
        marked = [int(e) for e, m in zip(es.event_time, es.is_reference) if m]
        assert marked == [-1]
        # The NATIVE anchor is an exact mechanical zero BEFORE marking (the
        # explicit-reference branch has no value check; post_init would
        # silently rewrite a nonzero anchor, so the native pin is the guard).
        assert fit_time.event_study_effects[-1]["effect"] == 0.0
        assert fit_time.event_study_effects[-1]["se"] == 0.0
        assert fit_time.event_study_effects[-1]["n_groups"] > 0

    def test_pt_post_absent_anchor_not_synthesized(self):
        # Single cohort baselined at the panel's first period: the anchor
        # cell is never estimated, the membership gate returns None, and the
        # container has NO reference row - nothing synthesized (the
        # no-fabrication rule; results_base synthesis branch NOT taken).
        import pandas as pd

        rng = np.random.default_rng(3)
        rows = []
        for u in range(40):
            coh = 1 if u < 20 else 0
            for t in range(6):
                rows.append(
                    {
                        "unit": u,
                        "period": t,
                        "first_treat": coh,
                        "outcome": rng.normal() + (1.5 if coh == 1 and t >= 1 else 0.0),
                    }
                )
        d = pd.DataFrame(rows)
        plain = _fit_efficient(d, est_kw={"pt_assumption": "post"})
        assert plain.reference_period is None
        es = plain.aggregate("event_study")
        assert not es.is_reference.any()
        assert -1 not in {int(e) for e in es.event_time}

    def test_pt_all_no_reference_row(self, efficient_fitted):
        assert efficient_fitted.reference_period is None
        es = efficient_fitted.aggregate("event_study")
        assert not es.is_reference.any()

    def test_retention_no_dataframe_no_unit_labels(self, efficient_panel):
        import pickle

        import pandas as pd

        d = efficient_panel.copy()
        sentinel = {u: f"SENTINEL-ID-{u}@example.invalid" for u in d["unit"].unique()}
        d["unit"] = d["unit"].map(sentinel)
        res = _fit_efficient(d)
        kit = res._aggregation_kit
        assert kit is not None
        for v in kit.bookkeeping.values():
            assert not isinstance(v, pd.DataFrame)
        assert not isinstance(kit.influence, pd.DataFrame)
        blob = pickle.dumps(res)
        assert b"SENTINEL-ID" not in blob
        # Pickle round-trip: aggregate() still works and matches.
        res2 = pickle.loads(blob)
        np.testing.assert_allclose(
            res.aggregate("group").att, res2.aggregate("group").att, rtol=1e-15
        )

    def test_immutability(self, efficient_fitted):
        MUTATED = ("event_study_effects", "group_effects")
        before = {f: getattr(efficient_fitted, f) for f in MUTATED}
        for level in ("event_study", "group", "simple", "event_study", "group"):
            efficient_fitted.aggregate(level)
        for f in MUTATED:
            assert getattr(efficient_fitted, f) is before[f]
        # Repeated calls agree (no kit mutation between calls).
        a = efficient_fitted.aggregate("group")
        b = efficient_fitted.aggregate("group")
        np.testing.assert_array_equal(a.att, b.att)
        np.testing.assert_array_equal(a.df, b.df)

    def test_survey_metadata_not_mutated(self):
        d, rep_cols = _efficient_survey_panel(replicate=True)
        sd = _efficient_survey_design(rep_cols)
        res = _fit_efficient(d, survey_design=sd)
        before = res.survey_metadata.df_survey
        res.aggregate("event_study")
        res.aggregate("group")
        assert res.survey_metadata.df_survey == before

    def test_retention_bootstrapped_spec_no_leak(self, efficient_panel):
        # The BootstrapReplaySpec adds only the RNG state dict + scalars to
        # the kit; a bootstrapped fit's kit must keep the no-DataFrame /
        # no-unit-label retention contract and pickle round-trip its replay.
        import pickle

        import pandas as pd

        d = efficient_panel.copy()
        sentinel = {u: f"SENTINEL-ID-{u}@example.invalid" for u in d["unit"].unique()}
        d["unit"] = d["unit"].map(sentinel)
        res = _fit_efficient(d, est_kw={"n_bootstrap": 20, "seed": 1})
        kit = res._aggregation_kit
        assert kit.bootstrap is not None and kit.bootstrap.bitgen_state is not None
        for v in kit.bookkeeping.values():
            assert not isinstance(v, pd.DataFrame)
        blob = pickle.dumps(res)
        assert b"SENTINEL-ID" not in blob
        res2 = pickle.loads(blob)
        np.testing.assert_array_equal(res.aggregate("group").se, res2.aggregate("group").se)

    def test_bootstrap_simple_relays_stored_quintet(self, efficient_panel):
        res = _fit_efficient(efficient_panel, est_kw={"n_bootstrap": 20, "seed": 1})
        _assert_bootstrap_simple_relay(
            res, n_expected=res._aggregation_kit.bookkeeping["n_units_total"]
        )

    def test_bootstrap_survey_simple_relay_df_nan(self):
        # EDiD supports full TSL survey designs under bootstrap (only
        # REPLICATE designs are rejected); the relay must NaN the df column
        # beside the finite survey metadata.
        d, _ = _efficient_survey_panel()
        sd = _efficient_survey_design()
        res = _fit_efficient(d, est_kw={"n_bootstrap": 20, "seed": 1}, survey_design=sd)
        assert res.survey_metadata is not None
        _assert_bootstrap_simple_relay(res)

    def test_bootstrap_fit_time_group_rows_clear_df_used(self, efficient_panel):
        res = _fit_efficient(
            efficient_panel,
            est_kw={"n_bootstrap": 20, "seed": 1},
            aggregate="group",
        )
        for row in res.group_effects.values():
            assert row.get("df_used") is None

    def test_weights_rejected(self, efficient_fitted):
        with pytest.raises(ValueError, match="does not accept a weights selector"):
            efficient_fitted.aggregate("event_study", weights="cell")

    @pytest.mark.parametrize("level", ["simple", "group"])
    def test_balance_e_rejected_on_non_event_study(self, efficient_fitted, level):
        with pytest.raises(ValueError, match="balance_e is not used"):
            efficient_fitted.aggregate(level, balance_e=1)

    @pytest.mark.parametrize("bad", ["calendar", "all", "nonsense"])
    def test_unsupported_types_fail_closed(self, efficient_fitted, bad):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            efficient_fitted.aggregate(bad)

    def test_legacy_no_kit_raises(self, efficient_fitted):
        import copy

        legacy = copy.copy(efficient_fitted)
        object.__setattr__(legacy, "_aggregation_kit", None)
        with pytest.raises(ValueError, match="aggregation kit"):
            legacy.aggregate("event_study")

    def test_public_influence_functions_isolated_from_kit(self, efficient_panel):
        # store_eif=True exposes the PUBLIC influence_functions diagnostic;
        # it must be an independent COPY of the kit's canonical EIF payload,
        # else a user mutation of the public field silently corrupts
        # recomputed post-fit SEs/p-values/CIs (review pin, M-023).
        res = _fit_efficient(efficient_panel, store_eif=True)
        kit = res._aggregation_kit
        assert res.influence_functions is not kit.influence
        for gt, arr in res.influence_functions.items():
            assert not np.shares_memory(arr, kit.influence[gt])
        before = res.aggregate("event_study").se.copy()
        first = next(iter(res.influence_functions))
        res.influence_functions[first][:] = 0.0
        after = res.aggregate("event_study").se
        np.testing.assert_array_equal(before, after)

    def test_public_aggregation_inputs_isolated_from_kit(self, efficient_panel):
        # CI review P0: aggregate() must recompute exclusively from the
        # kit's PRIVATE snapshots - mutating the public group_time_effects
        # rows or the groups/time_periods lists after fit must not change
        # post-fit aggregation output (else altered point estimates mix
        # with retained fit-time EIF variance: plausible-but-invalid
        # inference).
        res = _fit_efficient(efficient_panel)
        g_before = res.aggregate("group")
        es_before = res.aggregate("event_study")
        post_gt = next(gt for gt in res.group_time_effects if gt[1] >= gt[0])
        res.group_time_effects[post_gt]["effect"] = 999.0
        res.groups.pop()
        res.time_periods.pop()
        # Provenance mutations (CI review R2): a flipped pt_assumption must
        # not turn the genuine e=-1 estimate into a zeroed reference row,
        # and alpha/anticipation edits must not relabel the containers.
        res.pt_assumption = "post"
        res.anticipation = 3
        res.alpha = 0.5
        g_after = res.aggregate("group")
        es_after = res.aggregate("event_study")
        s_after = res.aggregate("simple")
        np.testing.assert_array_equal(g_before.att, g_after.att)
        np.testing.assert_array_equal(g_before.se, g_after.se)
        assert list(g_before.label) == list(g_after.label)
        np.testing.assert_array_equal(es_before.att, es_after.att)
        np.testing.assert_array_equal(es_before.se, es_after.se)
        assert not es_after.is_reference.any()
        assert es_after.alpha == es_before.alpha
        assert s_after.alpha == 0.05 and float(s_after.n[0]) == float(
            res.n_treated_units + res.n_control_units
        )

    def test_zero_row_balance_e_surface(self, efficient_fitted):
        # An anchor no cohort reaches: warns, returns a LEGAL 0-row surface.
        with pytest.warns(UserWarning, match="anchor horizon"):
            es = efficient_fitted.aggregate("event_study", balance_e=99)
        assert len(es.event_time) == 0


# --------------------------------------------------------------------------- #
# EfficientDiD bootstrap REPLAY (the CS BootstrapReplaySpec mechanism):
# post-fit aggregate('event_study'/'group') on bootstrapped fits replays the
# fit-time multiplier bootstrap from the kit-retained RNG state. The parity
# REFERENCE is always the NATIVE fit-time surface (the kit attaches
# unconditionally, so a fit-time-aggregated result's own aggregate() would
# ALSO replay - replay-vs-replay proves nothing).
# Tolerances: se/ci/t at 1e-13 (the replayed draws are bit-identical; only
# GEMM tile-boundary reassociation differs - ~1 ULP, with headroom for
# quantile interpolation); percentile p-values are COUNT statistics, so a
# draw within a ULP of the point estimate could flip one count - compared
# at atol=2/n_bootstrap.
# --------------------------------------------------------------------------- #

_EDID_NBOOT = 50


def _efficient_boot_fit(data, **fit_kw):
    est_kw = fit_kw.pop("est_kw", {})
    return _fit_efficient(data, est_kw={"n_bootstrap": _EDID_NBOOT, "seed": 42, **est_kw}, **fit_kw)


def _efficient_boot_fit_time(data, *, aggregate="all", **fit_kw):
    est_kw = fit_kw.pop("est_kw", {})
    return _fit_efficient(
        data,
        est_kw={"n_bootstrap": _EDID_NBOOT, "seed": 42, **est_kw},
        aggregate=aggregate,
        **fit_kw,
    )


def _assert_edid_es_replay_parity(es, fit_time, n_boot=_EDID_NBOOT):
    df = es.to_dataframe()
    assert es.vcov is None
    assert np.all(np.isnan(df["df"].to_numpy(dtype=float)))
    assert len(df) == len(fit_time.event_study_effects)
    p_atol = 2.0 / n_boot
    for _, row in df.iterrows():
        ref = fit_time.event_study_effects[int(row["event_time"])]
        assert row["att"] == ref["effect"]
        np.testing.assert_allclose(row["se"], ref["se"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(row["t_stat"], ref["t_stat"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(row["p_value"], ref["p_value"], rtol=1e-13, atol=p_atol)
        np.testing.assert_allclose(
            [row["conf_int_lower"], row["conf_int_upper"]],
            list(ref["conf_int"]),
            rtol=1e-13,
            atol=1e-13,
        )


def _assert_edid_group_replay_parity(grp, fit_time, n_boot=_EDID_NBOOT):
    df = grp.to_dataframe()
    assert np.all(np.isnan(df["df"].to_numpy(dtype=float)))
    assert len(df) == len(fit_time.group_effects)
    p_atol = 2.0 / n_boot
    for _, row in df.iterrows():
        ref = fit_time.group_effects[float(row["label"])]
        assert row["att"] == ref["effect"]
        np.testing.assert_allclose(row["se"], ref["se"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(row["t_stat"], ref["t_stat"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(row["p_value"], ref["p_value"], rtol=1e-13, atol=p_atol)
        np.testing.assert_allclose(
            [row["conf_int_lower"], row["conf_int_upper"]],
            list(ref["conf_int"]),
            rtol=1e-13,
            atol=1e-13,
        )


def _efficient_fractional_panel(cohorts=(3.0, 2.25), seed=0, n_units=90):
    """0.5-spaced panel; cohorts may sit ON the grid or OFF it (e.g. 2.25)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    periods = np.arange(1.0, 5.0, 0.5)
    per_cohort = n_units // (len(cohorts) + 2)
    rows = []
    for u in range(n_units):
        idx = u // per_cohort
        g = cohorts[idx] if idx < len(cohorts) else 0.0
        for t in periods:
            y = rng.normal() + u * 0.01 + t * 0.1 + (0.5 if g > 0 and t >= g else 0.0)
            rows.append((u, t, g, y))
    return pd.DataFrame(rows, columns=["unit", "period", "first_treat", "outcome"])


class TestEfficientBootstrapReplay:
    def test_event_study_parity_with_fit_time(self, efficient_panel):
        res = _efficient_boot_fit(efficient_panel)
        ftime = _efficient_boot_fit_time(efficient_panel)
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)

    def test_group_parity_with_fit_time(self, efficient_panel):
        res = _efficient_boot_fit(efficient_panel)
        ftime = _efficient_boot_fit_time(efficient_panel)
        _assert_edid_group_replay_parity(res.aggregate("group"), ftime)

    def test_balance_e_parity_with_fit_time(self, efficient_panel):
        res = _efficient_boot_fit(efficient_panel)
        ftime = _efficient_boot_fit_time(efficient_panel, aggregate="event_study", balance_e=1)
        _assert_edid_es_replay_parity(res.aggregate("event_study", balance_e=1), ftime)

    def test_balance_e_empty_anchor_replay_warns_zero_rows(self, efficient_panel):
        # The replay re-emits the fit-time anchor warning and returns the
        # LEGAL zero-row surface (the analytical twin of
        # test_zero_row_balance_e_surface, now on the replay route).
        res = _efficient_boot_fit(efficient_panel)
        with pytest.warns(UserWarning, match="anchor horizon"):
            es = res.aggregate("event_study", balance_e=99)
        assert len(es.event_time) == 0

    def test_pt_post_parity_with_fit_time(self, efficient_panel):
        # PT-Post: the bootstrap prep DOES key the finite effect=0.0 anchor
        # at e=-1 and the override NaNs its inference (zero-SE draws); the
        # is_reference marking is label-based and survives. Parity holds
        # row-for-row, the anchor included.
        res = _efficient_boot_fit(efficient_panel, est_kw={"pt_assumption": "post"})
        ftime = _efficient_boot_fit_time(efficient_panel, est_kw={"pt_assumption": "post"})
        es = res.aggregate("event_study")
        df = es.to_dataframe()
        anchor = df[df["is_reference"]]
        assert len(anchor) == 1
        assert float(anchor["att"].iloc[0]) == 0.0
        assert np.isnan(float(anchor["se"].iloc[0]))
        # Non-reference rows hit full parity; the reference row's fit-time
        # DICT entry is also NaN'd by the override (same applier), so the
        # container/dict split is exercised on the fractional fixture below.
        _assert_edid_es_replay_parity(es, ftime)

    def test_covariates_parity_with_fit_time(self, efficient_panel):
        d = efficient_panel.copy()
        rng = np.random.default_rng(3)
        xmap = {u: rng.normal() for u in d["unit"].unique()}
        d["x1"] = d["unit"].map(xmap)
        res = _efficient_boot_fit(d, covariates=["x1"])
        ftime = _efficient_boot_fit_time(d, covariates=["x1"])
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)
        _assert_edid_group_replay_parity(res.aggregate("group"), ftime)

    def test_seedless_fit_replays_and_is_idempotent(self, efficient_panel):
        res = _fit_efficient(efficient_panel, est_kw={"n_bootstrap": _EDID_NBOOT})
        a = res.aggregate("event_study").to_dataframe()
        b = res.aggregate("event_study").to_dataframe()
        np.testing.assert_array_equal(a["se"].to_numpy(), b["se"].to_numpy())

    def test_set_params_and_mutation_immunity(self, efficient_panel):
        from diff_diff import EfficientDiD

        est = EfficientDiD(n_bootstrap=_EDID_NBOOT, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = est.fit(efficient_panel, **EFFICIENT_KW)
        before = res.aggregate("event_study").to_dataframe()
        est.set_params(n_bootstrap=5, seed=1, bootstrap_weights="mammen")
        est.n_bootstrap = 3
        after = res.aggregate("event_study").to_dataframe()
        np.testing.assert_array_equal(before["se"].to_numpy(), after["se"].to_numpy())

    def test_pickle_round_trip_replays(self, efficient_panel):
        import pickle

        res = _efficient_boot_fit(efficient_panel)
        before = res.aggregate("group").to_dataframe()
        res2 = pickle.loads(pickle.dumps(res))
        after = res2.aggregate("group").to_dataframe()
        np.testing.assert_array_equal(before["se"].to_numpy(), after["se"].to_numpy())

    def test_relays_unchanged_and_order_independent(self, efficient_panel):
        res = _efficient_boot_fit(efficient_panel)
        s_before = res.aggregate("simple")
        res.aggregate("event_study")
        s_after = res.aggregate("simple")
        assert float(s_before.att[0]) == float(s_after.att[0]) == res.overall_att
        assert float(s_before.se[0]) == float(s_after.se[0]) == res.overall_se

    def test_legacy_kit_without_spec_fails_closed(self, efficient_panel):
        res = _efficient_boot_fit(efficient_panel)
        object.__setattr__(res._aggregation_kit, "bootstrap", None)
        for level in ("event_study", "group"):
            with pytest.raises(NotImplementedError, match="predates"):
                res.aggregate(level)

    def test_backend_mismatch_fails_closed(self, efficient_panel):
        import dataclasses as dc

        from diff_diff.bootstrap_chunking import effective_weight_backend

        res = _efficient_boot_fit(efficient_panel)
        kit = res._aggregation_kit
        current = effective_weight_backend()
        other = "numpy" if current == "rust" else "rust"
        assert kit.bootstrap.backend == current  # plain fits stamp the live backend
        object.__setattr__(kit, "bootstrap", dc.replace(kit.bootstrap, backend=other))
        for level in ("event_study", "group"):
            with pytest.raises(NotImplementedError, match="weight backend"):
                res.aggregate(level)
        # None (unknown) also fails closed - a permissive default on a
        # safety discriminator would disarm the guard.
        object.__setattr__(kit, "bootstrap", dc.replace(kit.bootstrap, backend=None))
        with pytest.raises(NotImplementedError, match="weight backend"):
            res.aggregate("event_study")

    def test_low_bootstrap_warning_re_emitted_on_replay(self, efficient_panel):
        res = _fit_efficient(efficient_panel, est_kw={"n_bootstrap": 49, "seed": 7})
        with pytest.warns(UserWarning, match="n_bootstrap=49 is low"):
            res.aggregate("event_study")


class TestEfficientBootstrapReplayDesigns:
    def test_cluster_parity(self):
        d = _efficient_clustered_panel()
        res = _efficient_boot_fit(d, est_kw={"cluster": "cl"})
        ftime = _efficient_boot_fit_time(d, est_kw={"cluster": "cl"})
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)
        _assert_edid_group_replay_parity(res.aggregate("group"), ftime)

    def test_weights_only_survey_parity(self):
        # Weights-only design: unit weight PATH + survey EIF scaling.
        d, _ = _efficient_survey_panel()
        sd = _efficient_survey_design()
        res = _efficient_boot_fit(d, survey_design=sd)
        ftime = _efficient_boot_fit_time(d, survey_design=sd)
        assert res._aggregation_kit.bootstrap.backend != "portable"
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)

    @staticmethod
    def _psu_survey_frame(strata=False, fpc=None):
        from diff_diff import SurveyDesign

        d, _ = _efficient_survey_panel()
        d = d.copy()
        d["psu"] = (d["unit"] // 6).astype(int)
        kw = dict(weights="w", psu="psu")
        if strata:
            d["stratum"] = (d["unit"] // 60).astype(int)
            kw["strata"] = "stratum"
            kw["nest"] = True
        if fpc is not None:
            d["fpc_col"] = float(fpc)
            kw["fpc"] = "fpc_col"
        return d, SurveyDesign(**kw)

    def test_stratified_survey_portable_stamp_and_parity(self):
        d, sd = self._psu_survey_frame(strata=True)
        res = _efficient_boot_fit(d, survey_design=sd)
        assert res._aggregation_kit.bootstrap.backend == "portable"
        ftime = _efficient_boot_fit_time(d, survey_design=sd)
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)
        _assert_edid_group_replay_parity(res.aggregate("group"), ftime)

    def test_fpc_parity(self):
        # Non-census FPC (population 10x the PSU count): fpc_scale on top of
        # the backend-dependent generator.
        d, sd = self._psu_survey_frame(fpc=200.0)
        res = _efficient_boot_fit(d, survey_design=sd)
        assert res._aggregation_kit.bootstrap.backend != "portable"
        ftime = _efficient_boot_fit_time(d, survey_design=sd)
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)

    def test_census_fpc_portable(self):
        # Census FPC (fpc == n_psu): every weight block is zeroed, so every
        # bootstrap distribution is EXACTLY CONSTANT at the original effect
        # - the discarded draws' backend is irrelevant (stamped portable),
        # and the constant-distribution guard must NaN ALL inference fields
        # on BOTH routes (a tiny-positive roundoff np.std at a non-zero
        # constant level must never leak a huge finite t - CI review P0).
        d, sd = self._psu_survey_frame(fpc=20.0)  # 120 units // 6 = 20 PSUs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _efficient_boot_fit(d, survey_design=sd)
            ftime = _efficient_boot_fit_time(d, survey_design=sd)
        assert res._aggregation_kit.bootstrap.backend == "portable"
        # Fit-time surfaces: full-NaN inference beside finite effects.
        for surface in (ftime.event_study_effects, ftime.group_effects):
            assert surface
            for row in surface.values():
                assert np.isfinite(row["effect"])
                assert np.isnan(row["se"]) and np.isnan(row["t_stat"])
                assert np.isnan(row["p_value"])
                assert np.isnan(row["conf_int"][0]) and np.isnan(row["conf_int"][1])
        # Replayed surfaces reproduce the same degenerate state.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es_df = res.aggregate("event_study").to_dataframe()
            g_df = res.aggregate("group").to_dataframe()
        for frame in (es_df, g_df):
            att = frame["att"].to_numpy(dtype=float)
            ref_mask = (
                frame["is_reference"].to_numpy(dtype=bool)
                if "is_reference" in frame
                else np.zeros(len(frame), dtype=bool)
            )
            assert np.all(np.isfinite(att[~ref_mask]))
            for col in ("se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper"):
                assert np.all(np.isnan(frame[col].to_numpy(dtype=float)[~ref_mask])), col

    def test_single_psu_nan_surfaces_and_warning(self):
        # n_psu < 2 early-returns the NaN container BEFORE any generation:
        # stamped portable; the replay re-hits the return deterministically
        # and re-emits the PSU warning; inference NaNs on both levels.
        from diff_diff import SurveyDesign

        d, _ = _efficient_survey_panel()
        d = d.copy()
        d["psu"] = 0
        sd = SurveyDesign(weights="w", psu="psu")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _efficient_boot_fit(d, survey_design=sd)
        assert res._aggregation_kit.bootstrap.backend == "portable"
        with pytest.warns(UserWarning, match="n_psu=1"):
            es = res.aggregate("event_study")
        assert np.all(np.isnan(es.se))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grp = res.aggregate("group").to_dataframe()
        assert np.all(np.isnan(grp["se"].to_numpy(dtype=float)))

    def test_anticipation_parity(self, efficient_panel):
        # Pins the replay host's anticipation wiring (shifts the engine's
        # post-treatment mask and the group prep's inclusion rule); a
        # defaulted host attribute would pass every anticipation=0 arm.
        res = _efficient_boot_fit(efficient_panel, est_kw={"anticipation": 1})
        ftime = _efficient_boot_fit_time(efficient_panel, est_kw={"anticipation": 1})
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)
        _assert_edid_group_replay_parity(res.aggregate("group"), ftime)

    def test_alpha_and_mammen_parity(self, efficient_panel):
        # Pins the host/spec wiring of alpha and weight_type - a host that
        # hard-codes the defaults passes every other arm.
        kw = {"alpha": 0.10, "bootstrap_weights": "mammen"}
        res = _efficient_boot_fit(efficient_panel, est_kw=kw)
        ftime = _efficient_boot_fit_time(efficient_panel, est_kw=kw)
        _assert_edid_es_replay_parity(res.aggregate("event_study"), ftime)
        _assert_edid_group_replay_parity(res.aggregate("group"), ftime)


class TestEfficientFractionalPeriods:
    """Decision-10 pins: int(t - g) truncation-bucketing on fractional panels.

    The published fit-time key set is int-bucketed by the ANALYTICAL
    aggregator regardless of the bootstrap prep's keying, so the
    discriminating surfaces are (a) the prep's own key set and (b) the
    off-grid-onset arm, where pre-fix NO raw key intersected the analytical
    buckets and the fit-time rows kept analytical inference.
    """

    def test_prep_keys_match_analytical_buckets_and_warns(self):
        from diff_diff import EfficientDiD

        d = _efficient_fractional_panel()
        # _fit_efficient suppresses warnings, so fit directly to pin the
        # fractional-truncation warning alongside the prep-key assertion.
        with pytest.warns(UserWarning, match="bucketed by int"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                res = EfficientDiD(n_bootstrap=_EDID_NBOOT, seed=42).fit(
                    d, aggregate="all", **EFFICIENT_KW
                )
        assert set(res.bootstrap_results.event_study_ses) >= set(res.event_study_effects)
        assert all(isinstance(e, int) for e in res.bootstrap_results.event_study_ses)

    def test_offgrid_rows_carry_percentile_inference(self):
        # EVERY treated cohort off-grid: raw t-g is never an integer, so
        # pre-fix no override would land and the fit-time rows would keep
        # ANALYTICAL inference - the end-to-end discriminator.
        d = _efficient_fractional_panel(cohorts=(2.25, 3.75))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = _efficient_boot_fit_time(d)
            ana = _fit_efficient(d, aggregate="all")
        for e, row in boot.event_study_effects.items():
            assert row["se"] != ana.event_study_effects[e]["se"]

    def test_replay_parity_on_fractional_panel(self):
        d = _efficient_fractional_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _efficient_boot_fit(d)
            ftime = _efficient_boot_fit_time(d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            es = res.aggregate("event_study")
        _assert_edid_es_replay_parity(es, ftime)

    def test_n_groups_counts_distinct_cohorts(self):
        # Fractional buckets pool multiple cells per cohort; n_groups must
        # count DISTINCT cohorts, not cells.
        d = _efficient_fractional_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _fit_efficient(d, aggregate="event_study")
        for e, row in res.event_study_effects.items():
            assert row["n_groups"] <= 2, (e, row["n_groups"])

    def test_integer_panel_n_groups_regression(self, efficient_fit_time):
        # Identity claim: on integer panels one cell per cohort per bucket,
        # so distinct-cohort counting equals the old cell count. Pin the
        # exact values on the standard 2-cohort fixture (cohorts 4 and 6 on
        # an 8-period panel: both cohorts share buckets -1..1 given cohort
        # 6's horizons span -5..1, cohort 4's -3..3).
        expected = {
            e: len({g for (g, t) in efficient_fit_time.group_time_effects if int(t - g) == e})
            for e in efficient_fit_time.event_study_effects
        }
        for e, row in efficient_fit_time.event_study_effects.items():
            assert row["n_groups"] == expected[e]
        assert max(expected.values()) == 2  # both cohorts pool somewhere

    def test_integer_panel_never_warns(self, efficient_panel):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _fit_efficient(efficient_panel)
            res.aggregate("event_study")
        assert not any("bucketed by int" in str(w.message) for w in caught)

    def test_fractional_balance_e_offgrid_anchor(self):
        # Cohort 2.25 reaches the integer anchor bucket 1 only via raw
        # horizons 1.25/1.75 - the :403 anchor-filter pin: the bootstrap
        # balanced cohort set must match the analytical one.
        d = _efficient_fractional_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = _efficient_boot_fit_time(d, aggregate="event_study", balance_e=1)
            ana = _fit_efficient(d, aggregate="event_study", balance_e=1)
        assert set(boot.bootstrap_results.event_study_ses) >= set(ana.event_study_effects)
        assert ana.event_study_effects[1]["n_groups"] == 2  # both cohorts anchored
        for e, row in boot.event_study_effects.items():
            assert row["se"] != ana.event_study_effects[e]["se"] or np.isnan(row["se"])

    def test_bucket_pooled_att_cell_mass_weighting(self):
        # Hand-computed oracle for the REGISTRY Note's weighting clause: a
        # bucket's ATT is the CELL-MASS weighted mean - one pi_g term per
        # CELL, normalized within the bucket (a cohort with k cells carries
        # k*pi_g mass) - not one term per cohort.
        d = _efficient_fractional_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _fit_efficient(d, aggregate="event_study")
        kit = res._aggregation_kit
        fracs = kit.bookkeeping["cohort_fractions"]
        gt = kit.bookkeeping["group_time_effects"]
        for e, row in res.event_study_effects.items():
            cells = [
                (d_["effect"], fracs.get(g, 0.0))
                for (g, t), d_ in gt.items()
                if int(t - g) == e and np.isfinite(d_["effect"])
            ]
            w = np.array([c[1] for c in cells])
            effs = np.array([c[0] for c in cells])
            w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
            np.testing.assert_allclose(row["effect"], float(np.sum(w * effs)), rtol=1e-12)

    def test_fractional_pt_post_reference_collision(self):
        # Note clause (iii), PER-ROUTE oracles: reference normalization is
        # CONTAINER-only. The bucket at -1-anticipation pools a genuine
        # fractional pre-treatment estimate with the mechanical zero anchor;
        # the container publishes it as the reference (att=0, NaN
        # inference), while the fit-time DICT keeps the pooled effect with
        # percentile inference. Parity asserted on NON-reference rows only.
        d = _efficient_fractional_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _efficient_boot_fit(d, est_kw={"pt_assumption": "post"})
            ftime = _efficient_boot_fit_time(d, est_kw={"pt_assumption": "post"})
            es = res.aggregate("event_study")
        df = es.to_dataframe()
        anchor = df[df["is_reference"]]
        assert len(anchor) == 1 and int(anchor["event_time"].iloc[0]) == -1
        assert float(anchor["att"].iloc[0]) == 0.0
        assert np.isnan(float(anchor["se"].iloc[0]))
        # The fit-time DICT entry for the same bucket is NOT rewritten.
        dict_row = ftime.event_study_effects[-1]
        assert dict_row["effect"] != 0.0 or not np.isnan(dict_row["se"])
        for _, row in df[~df["is_reference"]].iterrows():
            ref = ftime.event_study_effects[int(row["event_time"])]
            assert row["att"] == ref["effect"]
            np.testing.assert_allclose(row["se"], ref["se"], rtol=1e-13, atol=1e-13)

    def test_hausman_pretest_fractional_warns(self):
        d = _efficient_fractional_panel()
        from diff_diff import EfficientDiD

        est = EfficientDiD()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(d, **EFFICIENT_KW)
        with pytest.warns(UserWarning, match="Hausman pre-test horizons are bucketed"):
            est.hausman_pretest(d, **EFFICIENT_KW)


class TestEfficientInternalCallers:
    def test_hausman_pretest_emits_no_future_warning(self, efficient_panel):
        # hausman_pretest refits internally; its fit_kwargs no longer pass
        # aggregate=, so the shim must not fire (regression pin for the
        # internal-caller cleanup - without it a re-added kwarg would warn
        # spuriously from every hausman_pretest / DiagnosticReport._pt_hausman
        # call with no test failing).
        from diff_diff import EfficientDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            EfficientDiD.hausman_pretest(
                efficient_panel,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []


# --------------------------------------------------------------------------- #
# ImputationDiD (rows M-021/M-118): fit(aggregate=/balance_e=) shim + the
# PANEL-BACKED recompute aggregate() (kit refs = the _fit_data objects)
# --------------------------------------------------------------------------- #

IMPUTATION_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _imputation_panel(seed=42, n_units=120, n_periods=8):
    from diff_diff.prep_dgp import generate_staggered_data

    return generate_staggered_data(
        n_units=n_units, n_periods=n_periods, cohort_periods=[4, 6], seed=seed
    )


def _imputation_clustered_panel(seed=11):
    d = _imputation_panel(seed=seed).copy()
    d["cl"] = (d["unit"] // 3).astype(int)
    return d


def _imputation_survey_panel(seed=5, replicate=False, degenerate=None):
    """Unit-constant pweights; optionally a JK replicate design.

    degenerate: None | "dropped" (2 all-zero replicate columns) |
    "undefined" (all but one column all-zero -> n_valid <= 1) |
    "cohort_zero" (replicate rw0 zeroes EVERY row of one cohort -> that
    replicate NaNs the cohort's GROUP target while overall stays finite,
    so the joint [overall, groups] stack drops a replicate the
    [overall]-only stack keeps - the deterministic overall-row
    migration-delta shape).
    """
    d = _imputation_panel(seed=seed).copy()
    rng = np.random.default_rng(seed)
    wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
    d["w"] = d["unit"].map(wmap)
    rep_cols = []
    if replicate:
        n_rep = 8
        cohort4_units = set(d.loc[d["first_treat"] == 4, "unit"].unique())
        for r in range(n_rep):
            col = f"rw{r}"
            rep_cols.append(col)
            if degenerate == "dropped" and r >= n_rep - 2:
                d[col] = 0.0
            elif degenerate == "undefined" and r >= 1:
                d[col] = 0.0
            else:
                jitter = {u: rng.uniform(0.1, 2.0) for u in d["unit"].unique()}
                d[col] = d["unit"].map(jitter) * d["w"]
                if degenerate == "cohort_zero" and r == 0:
                    d.loc[d["unit"].isin(cohort4_units), col] = 0.0
    return d, rep_cols


def _imputation_survey_design(rep_cols=None):
    from diff_diff import SurveyDesign

    if rep_cols:
        return SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
    return SurveyDesign(weights="w")


def _fit_imputation(data, *, est_kw=None, **fit_kw):
    from diff_diff import ImputationDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ImputationDiD(**(est_kw or {})).fit(data, **IMPUTATION_KW, **fit_kw)


def _assert_es_container_matches_fit_time(container, fit_time_results, name=""):
    """Post-fit EventStudyResults must equal the fit-time-built surface
    column-for-column at 1e-14 (incl. reference marking and df provenance)."""
    from diff_diff.results_base import build_event_study_surface

    ref = build_event_study_surface(fit_time_results)
    da, db = container.to_dataframe(), ref.to_dataframe()
    assert list(da.columns) == list(db.columns)
    for c in da.columns:
        x, y = da[c].to_numpy(), db[c].to_numpy()
        if x.dtype.kind in "fc":
            np.testing.assert_allclose(
                x, y, rtol=0, atol=1e-14, equal_nan=True, err_msg=f"{name}/{c}"
            )
        else:
            assert (x == y).all(), (name, c)
    for attr in ("df_survey", "anticipation", "alpha"):
        va, vb = getattr(container, attr), getattr(ref, attr)
        same = va == vb or (va is None and vb is None)
        try:
            same = same or (np.isnan(va) and np.isnan(vb))
        except TypeError:
            pass
        assert same, (name, attr, va, vb)


def _assert_group_matches_fit_time(agg, fit_time_results, name=""):
    for i, g in enumerate(agg.label):
        row = fit_time_results.group_effects[g]
        for field_, key in (
            ("att", "effect"),
            ("se", "se"),
            ("t_stat", "t_stat"),
            ("p_value", "p_value"),
        ):
            np.testing.assert_allclose(
                getattr(agg, field_)[i],
                row[key],
                rtol=0,
                atol=1e-14,
                equal_nan=True,
                err_msg=f"{name}/{g}/{field_}",
            )


@pytest.fixture(scope="module")
def imputation_panel():
    return _imputation_panel()


@pytest.fixture(scope="module")
def imputation_fitted(imputation_panel):
    """Plain fit - the kit (refs to _fit_data) powers aggregate()."""
    return _fit_imputation(imputation_panel)


@pytest.fixture(scope="module")
def imputation_fit_time(imputation_panel):
    """Deprecated fit-time aggregate="all" - the analytical inertness reference."""
    return _fit_imputation(imputation_panel, aggregate="all")


class TestImputationShim:
    def test_plain_fit_does_not_warn(self, imputation_panel):
        from diff_diff import ImputationDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ImputationDiD().fit(imputation_panel, **IMPUTATION_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, imputation_panel):
        from diff_diff import ImputationDiD

        with pytest.warns(FutureWarning, match=r"ImputationDiD\.fit\(aggregate=\)"):
            ImputationDiD().fit(imputation_panel, **IMPUTATION_KW, aggregate=None)

    def test_balance_e_kwarg_warns_alone(self, imputation_panel):
        from diff_diff import ImputationDiD

        with pytest.warns(FutureWarning, match=r"ImputationDiD\.fit\(balance_e=\)"):
            ImputationDiD().fit(imputation_panel, **IMPUTATION_KW, balance_e=None)

    def test_joint_supply_warns_once_naming_both(self, imputation_panel):
        from diff_diff import ImputationDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ImputationDiD().fit(
                imputation_panel, **IMPUTATION_KW, aggregate="event_study", balance_e=0
            )
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1
        msg = str(fw[0].message)
        assert "aggregate=" in msg and "balance_e=" in msg

    def test_unknown_string_still_acts_like_none(self, imputation_panel):
        res = _fit_imputation(imputation_panel, aggregate="nonsense")
        plain = _fit_imputation(imputation_panel)
        assert res.event_study_effects is None and res.group_effects is None
        assert res.overall_att == plain.overall_att

    def test_warn_and_still_work(self, imputation_fitted, imputation_fit_time):
        assert imputation_fit_time.event_study_effects is not None
        assert imputation_fit_time.group_effects is not None
        es = imputation_fitted.aggregate("event_study")
        for e, row in imputation_fit_time.event_study_effects.items():
            i = list(es.event_time).index(e)
            if np.isfinite(row["effect"]) and not es.is_reference[i]:
                np.testing.assert_allclose(row["effect"], es.att[i], rtol=1e-14)

    def test_wrapper_forwarded_aggregate_warns(self, imputation_panel):
        from diff_diff import imputation_did

        # Both warnings fire since 2(d) PR-A (M-070 + the forwarded shim).
        with pytest.warns(FutureWarning) as record:
            imputation_did(
                imputation_panel,
                "outcome",
                "unit",
                "period",
                "first_treat",
                aggregate="event_study",
            )
        msgs = [str(w.message) for w in record]
        assert any("imputation_did() is deprecated" in m for m in msgs), msgs
        assert any(re.search(r"ImputationDiD\.fit\(aggregate=\)", m) for m in msgs), msgs

    def test_wrapper_forwarded_balance_e_warns(self, imputation_panel):
        from diff_diff import imputation_did

        with pytest.warns(FutureWarning) as record:
            imputation_did(
                imputation_panel, "outcome", "unit", "period", "first_treat", balance_e=1
            )
        msgs = [str(w.message) for w in record]
        assert any("imputation_did() is deprecated" in m for m in msgs), msgs
        assert any(re.search(r"ImputationDiD\.fit\(balance_e=\)", m) for m in msgs), msgs

    def test_plain_wrapper_call_fires_only_wrapper_warning(self, imputation_panel):
        # Flipped BY DESIGN in the 2(d) PR-A (M-070): the wrapper's own
        # deprecation warning fires, the forwarded aggregate sentinel
        # still never does - EXACTLY ONE FutureWarning.
        from diff_diff import imputation_did

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            imputation_did(imputation_panel, "outcome", "unit", "period", "first_treat")
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1, [str(w.message) for w in fw]
        assert "imputation_did() is deprecated" in str(fw[0].message)
        assert "ImputationDiD.fit" not in str(fw[0].message)


class TestImputationAggregate:
    def test_event_study_inert(self, imputation_fitted, imputation_fit_time):
        es = imputation_fitted.aggregate("event_study")
        _assert_es_container_matches_fit_time(es, imputation_fit_time, "imp/es")

    @pytest.mark.parametrize("balance_e", [0, 1, 2])
    def test_balance_e_inert(self, imputation_panel, imputation_fitted, balance_e):
        ref = _fit_imputation(imputation_panel, aggregate="event_study", balance_e=balance_e)
        es = imputation_fitted.aggregate("event_study", balance_e=balance_e)
        _assert_es_container_matches_fit_time(es, ref, f"imp/es/be{balance_e}")

    def test_group_inert(self, imputation_fitted, imputation_fit_time):
        _assert_group_matches_fit_time(
            imputation_fitted.aggregate("group"), imputation_fit_time, "imp/gr"
        )

    def test_simple_relay_bit_exact(self, imputation_fitted):
        sm = imputation_fitted.aggregate("simple")
        assert sm.att[0] == imputation_fitted.overall_att
        assert sm.se[0] == imputation_fitted.overall_se
        assert sm.t_stat[0] == imputation_fitted.overall_t_stat
        assert sm.p_value[0] == imputation_fitted.overall_p_value
        assert sm.n_kind == "obs"
        assert sm.n[0] == imputation_fitted.n_treated_obs
        assert np.isnan(sm.to_dataframe()["df"].to_numpy()).all()  # non-survey fit

    def test_cluster_arm_inert(self):
        d = _imputation_clustered_panel()
        ref = _fit_imputation(d, est_kw={"cluster": "cl"}, aggregate="all")
        plain = _fit_imputation(d, est_kw={"cluster": "cl"})
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "imp/cluster")
        _assert_group_matches_fit_time(plain.aggregate("group"), ref, "imp/cluster")

    def test_covariate_arm_inert(self):
        # Time-varying covariate: the kit must carry delta_hat /
        # kept_cov_mask / the covariate columns; an omission escapes every
        # covariate-free arm (local-review P2).
        d = _imputation_panel(seed=23).copy()
        rng = np.random.default_rng(23)
        d["x1"] = rng.normal(size=len(d)) + 0.1 * d["period"]
        ref = _fit_imputation(d, aggregate="all", covariates=["x1"])
        plain = _fit_imputation(d, covariates=["x1"])
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "imp/cov")
        _assert_group_matches_fit_time(plain.aggregate("group"), ref, "imp/cov")

    def test_survey_tsl_arm_inert(self):
        d, _ = _imputation_survey_panel()
        sd = _imputation_survey_design()
        ref = _fit_imputation(d, aggregate="all", survey_design=sd)
        plain = _fit_imputation(d, survey_design=sd)
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "imp/tsl")
        _assert_group_matches_fit_time(plain.aggregate("group"), ref, "imp/tsl")
        sm = plain.aggregate("simple")
        df_col = sm.to_dataframe()["df"].to_numpy()
        assert np.isfinite(df_col).all()  # survey df threads the simple row

    @pytest.mark.parametrize("degenerate", [None, "dropped", "undefined"])
    def test_replicate_arms_level_matched(self, degenerate):
        # LEVEL-MATCHED references: aggregate(L) reproduces fit(aggregate=L),
        # not fit(aggregate='all') (the joint replicate stack couples rows).
        d, rep_cols = _imputation_survey_panel(replicate=True, degenerate=degenerate)
        sd = _imputation_survey_design(rep_cols)
        ref_es = _fit_imputation(d, aggregate="event_study", survey_design=sd)
        ref_gr = _fit_imputation(d, aggregate="group", survey_design=sd)
        plain = _fit_imputation(d, survey_design=sd)
        _assert_es_container_matches_fit_time(
            plain.aggregate("event_study"), ref_es, f"imp/rep/{degenerate}"
        )
        _assert_group_matches_fit_time(plain.aggregate("group"), ref_gr, f"imp/rep/{degenerate}")
        sm = plain.aggregate("simple")
        np.testing.assert_allclose(sm.se[0], plain.overall_se, rtol=0, atol=0, equal_nan=True)

    def test_replicate_undefined_df_sentinel(self):
        # n_valid <= 1: the working df degenerates; the ES container's
        # df_survey resolves to the 0.0 replicate-undefined sentinel on
        # BOTH routes (the metadata-copy discriminator).
        d, rep_cols = _imputation_survey_panel(replicate=True, degenerate="undefined")
        sd = _imputation_survey_design(rep_cols)
        plain = _fit_imputation(d, survey_design=sd)
        es = plain.aggregate("event_study")
        ref = _fit_imputation(d, aggregate="event_study", survey_design=sd)
        from diff_diff.results_base import build_event_study_surface

        assert es.df_survey == build_event_study_surface(ref).df_survey == 0.0

    def test_replicate_overall_row_migration_delta(self):
        # The documented migration consequence: the deprecated fit(aggregate=)
        # coupled the OVERALL row to the joint replicate stack. On the
        # cohort-zero design (one replicate NaNs one cohort's group target
        # while overall stays finite) the joint [overall, groups] stack drops
        # a replicate the [overall]-only stack keeps -> plain-fit overall_se
        # differs from fit(aggregate='group') overall_se; each surface is
        # self-consistent, and post-fit aggregate('group') level-matches the
        # deprecated fit(aggregate='group') rows exactly.
        d, rep_cols = _imputation_survey_panel(replicate=True, degenerate="cohort_zero")
        sd = _imputation_survey_design(rep_cols)
        plain = _fit_imputation(d, survey_design=sd)
        ref_gr = _fit_imputation(d, aggregate="group", survey_design=sd)
        assert plain.overall_se != ref_gr.overall_se
        _assert_group_matches_fit_time(plain.aggregate("group"), ref_gr, "imp/delta")
        # healthy design: no coupling -> equality
        dh, rep_h = _imputation_survey_panel(replicate=True)
        sdh = _imputation_survey_design(rep_h)
        # Healthy design: no replicate-drop coupling, so the two stacks
        # agree - but NOT bit-identically: the [overall] vs
        # [overall, groups] layouts route the replicate-variance matmul
        # through different BLAS kernel shapes (~1 ULP reassociation;
        # bit-identical on Accelerate, not on OpenBLAS-ARM/Windows).
        # A REAL coupling delta is O(se) itself, far above this band.
        np.testing.assert_allclose(
            _fit_imputation(dh, survey_design=sdh).overall_se,
            _fit_imputation(dh, aggregate="group", survey_design=sdh).overall_se,
            rtol=1e-12,
        )

    def test_pretrends_arm_inert(self, imputation_panel):
        ref = _fit_imputation(imputation_panel, est_kw={"pretrends": True}, aggregate="event_study")
        plain = _fit_imputation(imputation_panel, est_kw={"pretrends": True})
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "imp/pretrends")

    def test_isolation_under_public_field_mutation(self, imputation_panel):
        res = _fit_imputation(imputation_panel)
        base_es = res.aggregate("event_study").to_dataframe()
        base_gr = res.aggregate("group").to_dataframe()
        base_sm = res.aggregate("simple").to_dataframe()
        res.groups.pop()
        res.time_periods.pop()
        object.__setattr__(res, "alpha", 0.5)
        object.__setattr__(res, "anticipation", 3)
        # Imputation-only public config fields the kit snapshots:
        object.__setattr__(res, "leave_one_out", True)
        object.__setattr__(res, "df_convention", "cluster")
        pd.testing.assert_frame_equal(res.aggregate("event_study").to_dataframe(), base_es)
        pd.testing.assert_frame_equal(res.aggregate("group").to_dataframe(), base_gr)
        pd.testing.assert_frame_equal(res.aggregate("simple").to_dataframe(), base_sm)

    def test_metadata_never_mutated_and_isolated(self):
        d, rep_cols = _imputation_survey_panel(replicate=True)
        sd = _imputation_survey_design(rep_cols)
        res = _fit_imputation(d, survey_design=sd)
        base = res.aggregate("event_study").to_dataframe()
        base_df_survey = res.aggregate("event_study").df_survey
        res.survey_metadata.df_survey = 999.0
        res.survey_metadata.replicate_method = None  # the 0.0-sentinel discriminator
        after = res.aggregate("event_study")
        pd.testing.assert_frame_equal(after.to_dataframe(), base)
        assert after.df_survey == base_df_survey
        assert res.survey_metadata.df_survey == 999.0  # aggregate() never writes back

    def test_repeated_and_order_independent(self, imputation_fitted):
        a = imputation_fitted.aggregate("group").to_dataframe()
        imputation_fitted.aggregate("event_study")
        imputation_fitted.aggregate("simple")
        b = imputation_fitted.aggregate("group").to_dataframe()
        pd.testing.assert_frame_equal(a, b)

    def test_bootstrap_recompute_levels_fail_closed(self, imputation_panel):
        res = _fit_imputation(imputation_panel, est_kw={"n_bootstrap": 19, "seed": 1})
        for level in ("event_study", "group"):
            with pytest.raises(NotImplementedError, match="bootstrap") as exc:
                res.aggregate(level)
            assert "aggregate('simple') and, where supported, aggregate('total') relay" in str(
                exc.value
            )

    def test_bootstrap_simple_relays_stored_quintet(self, imputation_panel):
        res = _fit_imputation(imputation_panel, est_kw={"n_bootstrap": 19, "seed": 1})
        _assert_bootstrap_simple_relay(
            res, n_expected=res._aggregation_kit.bookkeeping["n_treated_obs"]
        )

    def test_bootstrap_survey_simple_relay_df_nan(self):
        # TSL survey + bootstrap is a supported combination (only replicate
        # designs are rejected under bootstrap); the relay NaNs the df column
        # beside the finite survey metadata.
        d, _ = _imputation_survey_panel()
        sd = _imputation_survey_design()
        res = _fit_imputation(d, est_kw={"n_bootstrap": 19, "seed": 1}, survey_design=sd)
        assert res.survey_metadata is not None
        _assert_bootstrap_simple_relay(res)

    def test_pretrends_replicate_es_fails_closed(self):
        d, rep_cols = _imputation_survey_panel(replicate=True)
        sd = _imputation_survey_design(rep_cols)
        res = _fit_imputation(d, est_kw={"pretrends": True}, survey_design=sd)
        with pytest.raises(NotImplementedError, match="per-replicate"):
            res.aggregate("event_study")
        res.aggregate("group")
        res.aggregate("simple")

    def test_fail_closed_vocabulary(self, imputation_fitted):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            imputation_fitted.aggregate("calendar")
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            imputation_fitted.aggregate("all")
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            imputation_fitted.aggregate("nonsense")
        with pytest.raises(ValueError, match="balance_e"):
            imputation_fitted.aggregate("group", balance_e=1)
        with pytest.raises(ValueError, match="balance_e"):
            imputation_fitted.aggregate("simple", balance_e=1)
        with pytest.raises(ValueError):
            imputation_fitted.aggregate("group", weights="cell")

    def test_legacy_pickle_without_kit(self, imputation_fitted):
        import copy

        legacy = copy.copy(imputation_fitted)
        object.__setattr__(legacy, "_aggregation_kit", None)
        with pytest.raises(ValueError, match="aggregation kit"):
            legacy.aggregate("group")

    def test_pickle_roundtrip(self, imputation_panel):
        import pickle

        res = _fit_imputation(imputation_panel)
        clone = pickle.loads(pickle.dumps(res))
        pd.testing.assert_frame_equal(
            clone.aggregate("group").to_dataframe(), res.aggregate("group").to_dataframe()
        )

    def test_group_df_used_relay(self):
        # Survey fit: every group row's df equals the survey df its
        # safe_inference received (per-row df_used capture).
        d, _ = _imputation_survey_panel()
        sd = _imputation_survey_design()
        plain = _fit_imputation(d, survey_design=sd)
        gr = plain.aggregate("group")
        df_col = gr.to_dataframe()["df"].to_numpy()
        finite_p = np.isfinite(gr.p_value)
        assert np.isfinite(df_col[finite_p]).all()
        # Plain fit: normal theory -> all-NaN df column.
        plain2 = _fit_imputation(_imputation_panel())
        assert np.isnan(plain2.aggregate("group").to_dataframe()["df"].to_numpy()).all()

    def test_bootstrap_group_rows_clear_df_used(self, imputation_panel):
        # Fit-time bootstrap override must never publish an analytical df
        # beside percentile inference (public group_effects row dicts).
        res = _fit_imputation(
            imputation_panel, est_kw={"n_bootstrap": 19, "seed": 1}, aggregate="group"
        )
        for row in res.group_effects.values():
            if np.isfinite(row["effect"]):
                assert row.get("df_used") is None

    def test_empty_balance_window(self, imputation_fitted):
        with pytest.warns(UserWarning, match="no horizons"):
            es = imputation_fitted.aggregate("event_study", balance_e=100)
        df_ = es.to_dataframe()
        assert (df_["is_reference"] | ~np.isfinite(df_["att"])).all()


# --------------------------------------------------------------------------- #
# TwoStageDiD (rows M-022/M-119): fit(aggregate=/balance_e=) shim + the
# PANEL-BACKED recompute aggregate() (column-subset working-frame kit)
# --------------------------------------------------------------------------- #

TWOSTAGE_KW = dict(outcome="outcome", unit="unit", time="period", first_treat="first_treat")


def _twostage_panel(seed=42, n_units=120, n_periods=8):
    from diff_diff.prep_dgp import generate_staggered_data

    return generate_staggered_data(
        n_units=n_units, n_periods=n_periods, cohort_periods=[4, 6], seed=seed
    )


def _twostage_prop5_panel(seed=13):
    """No never-treated units, multiple cohorts -> Proposition-5 NaN rows."""
    d = _twostage_panel(seed=seed)
    return d[d["first_treat"] > 0].copy()


def _twostage_survey_panel(seed=5, replicate=False, degenerate=None):
    d = _twostage_panel(seed=seed).copy()
    rng = np.random.default_rng(seed)
    wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
    d["w"] = d["unit"].map(wmap)
    rep_cols = []
    if replicate:
        n_rep = 8
        cohort4_units = set(d.loc[d["first_treat"] == 4, "unit"].unique())
        for r in range(n_rep):
            col = f"rw{r}"
            rep_cols.append(col)
            if degenerate == "dropped" and r >= n_rep - 2:
                d[col] = 0.0
            elif degenerate == "undefined" and r >= 1:
                d[col] = 0.0
            else:
                jitter = {u: rng.uniform(0.1, 2.0) for u in d["unit"].unique()}
                d[col] = d["unit"].map(jitter) * d["w"]
                if degenerate == "cohort_zero" and r == 0:
                    d.loc[d["unit"].isin(cohort4_units), col] = 0.0
    return d, rep_cols


def _twostage_always_treated_panel(seed=7):
    """Adds always-treated units so the survey Wave-E.3 pad activates
    (the only shape where the kit's score_pad_mask/cluster_ids_full
    snapshots are non-None)."""
    d, _ = _twostage_survey_panel(seed=seed)
    rng = np.random.default_rng(seed)
    extra = []
    base_unit = int(d["unit"].max()) + 1
    for k in range(8):
        u = base_unit + k
        for t in sorted(d["period"].unique()):
            extra.append((u, t, rng.normal() + 2.0, 1, 1.0))
    extra_df = pd.DataFrame(extra, columns=["unit", "period", "outcome", "first_treat", "w"])
    return pd.concat([d, extra_df], ignore_index=True)


def _twostage_survey_design(rep_cols=None, psu=False):
    from diff_diff import SurveyDesign

    if rep_cols:
        return SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
    if psu:
        return SurveyDesign(weights="w", psu="unit")
    return SurveyDesign(weights="w")


def _fit_twostage(data, *, est_kw=None, **fit_kw):
    from diff_diff import TwoStageDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TwoStageDiD(**(est_kw or {})).fit(data, **TWOSTAGE_KW, **fit_kw)


@pytest.fixture(scope="module")
def twostage_panel():
    return _twostage_panel()


@pytest.fixture(scope="module")
def twostage_fitted(twostage_panel):
    return _fit_twostage(twostage_panel)


@pytest.fixture(scope="module")
def twostage_fit_time(twostage_panel):
    return _fit_twostage(twostage_panel, aggregate="all")


class TestTwoStageShim:
    def test_plain_fit_does_not_warn(self, twostage_panel):
        from diff_diff import TwoStageDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TwoStageDiD().fit(twostage_panel, **TWOSTAGE_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, twostage_panel):
        from diff_diff import TwoStageDiD

        with pytest.warns(FutureWarning, match=r"TwoStageDiD\.fit\(aggregate=\)"):
            TwoStageDiD().fit(twostage_panel, **TWOSTAGE_KW, aggregate=None)

    def test_balance_e_kwarg_warns_alone(self, twostage_panel):
        from diff_diff import TwoStageDiD

        with pytest.warns(FutureWarning, match=r"TwoStageDiD\.fit\(balance_e=\)"):
            TwoStageDiD().fit(twostage_panel, **TWOSTAGE_KW, balance_e=None)

    def test_joint_supply_warns_once_naming_both(self, twostage_panel):
        from diff_diff import TwoStageDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TwoStageDiD().fit(twostage_panel, **TWOSTAGE_KW, aggregate="event_study", balance_e=0)
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1
        msg = str(fw[0].message)
        assert "aggregate=" in msg and "balance_e=" in msg

    def test_unknown_string_still_acts_like_none(self, twostage_panel):
        res = _fit_twostage(twostage_panel, aggregate="nonsense")
        plain = _fit_twostage(twostage_panel)
        assert res.event_study_effects is None and res.group_effects is None
        assert res.overall_att == plain.overall_att

    def test_warn_and_still_work(self, twostage_fitted, twostage_fit_time):
        assert twostage_fit_time.event_study_effects is not None
        assert twostage_fit_time.group_effects is not None
        es = twostage_fitted.aggregate("event_study")
        for e, row in twostage_fit_time.event_study_effects.items():
            i = list(es.event_time).index(e)
            if np.isfinite(row["effect"]) and not es.is_reference[i]:
                np.testing.assert_allclose(row["effect"], es.att[i], rtol=1e-14)

    def test_wrapper_forwarded_aggregate_warns(self, twostage_panel):
        from diff_diff import two_stage_did

        # Both warnings fire since 2(d) PR-A (M-071 + the forwarded shim).
        with pytest.warns(FutureWarning) as record:
            two_stage_did(
                twostage_panel,
                "outcome",
                "unit",
                "period",
                "first_treat",
                aggregate="event_study",
            )
        msgs = [str(w.message) for w in record]
        assert any("two_stage_did() is deprecated" in m for m in msgs), msgs
        assert any(re.search(r"TwoStageDiD\.fit\(aggregate=\)", m) for m in msgs), msgs

    def test_wrapper_forwarded_balance_e_warns(self, twostage_panel):
        from diff_diff import two_stage_did

        with pytest.warns(FutureWarning) as record:
            two_stage_did(twostage_panel, "outcome", "unit", "period", "first_treat", balance_e=1)
        msgs = [str(w.message) for w in record]
        assert any("two_stage_did() is deprecated" in m for m in msgs), msgs
        assert any(re.search(r"TwoStageDiD\.fit\(balance_e=\)", m) for m in msgs), msgs

    def test_plain_wrapper_call_fires_only_wrapper_warning(self, twostage_panel):
        # Flipped BY DESIGN in the 2(d) PR-A (M-071): the wrapper's own
        # deprecation warning fires, the forwarded aggregate sentinel
        # still never does - EXACTLY ONE FutureWarning.
        from diff_diff import two_stage_did

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            two_stage_did(twostage_panel, "outcome", "unit", "period", "first_treat")
        fw = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(fw) == 1, [str(w.message) for w in fw]
        assert "two_stage_did() is deprecated" in str(fw[0].message)
        assert "TwoStageDiD.fit" not in str(fw[0].message)


class TestTwoStageAggregate:
    def test_event_study_inert(self, twostage_fitted, twostage_fit_time):
        es = twostage_fitted.aggregate("event_study")
        _assert_es_container_matches_fit_time(es, twostage_fit_time, "ts/es")

    @pytest.mark.parametrize("balance_e", [0, 1, 2])
    def test_balance_e_inert(self, twostage_panel, twostage_fitted, balance_e):
        ref = _fit_twostage(twostage_panel, aggregate="event_study", balance_e=balance_e)
        es = twostage_fitted.aggregate("event_study", balance_e=balance_e)
        _assert_es_container_matches_fit_time(es, ref, f"ts/es/be{balance_e}")

    def test_group_inert(self, twostage_fitted, twostage_fit_time):
        _assert_group_matches_fit_time(
            twostage_fitted.aggregate("group"), twostage_fit_time, "ts/gr"
        )

    def test_simple_relay_bit_exact(self, twostage_fitted):
        sm = twostage_fitted.aggregate("simple")
        assert sm.att[0] == twostage_fitted.overall_att
        assert sm.se[0] == twostage_fitted.overall_se
        assert sm.n_kind == "obs"
        assert sm.n[0] == twostage_fitted.n_treated_obs
        assert np.isnan(sm.to_dataframe()["df"].to_numpy()).all()

    def test_m092_vcov_parity_analytical(self, twostage_fitted, twostage_panel):
        # The post-fit container threads the recomputed joint GMM vcov +
        # index + df exactly as the level-matched fit-time container does.
        from diff_diff.results_base import build_event_study_surface

        ref = build_event_study_surface(_fit_twostage(twostage_panel, aggregate="event_study"))
        es = twostage_fitted.aggregate("event_study")
        assert es.vcov is not None and ref.vcov is not None
        np.testing.assert_allclose(es.vcov, ref.vcov, rtol=0, atol=1e-14, equal_nan=True)
        assert list(es.vcov_index) == list(ref.vcov_index)

    def test_m092_vcov_cleared_on_replicate(self):
        d, rep_cols = _twostage_survey_panel(replicate=True)
        sd = _twostage_survey_design(rep_cols)
        plain = _fit_twostage(d, survey_design=sd)
        es = plain.aggregate("event_study")
        assert es.vcov is None and es.vcov_index is None
        # df provenance still threads (level-matched replayed value)
        ref = _fit_twostage(d, aggregate="event_study", survey_design=sd)
        from diff_diff.results_base import build_event_study_surface

        ref_surface = build_event_study_surface(ref)
        assert (es.df_survey == ref_surface.df_survey) or (
            es.df_survey is None and ref_surface.df_survey is None
        )

    def test_cluster_arm_inert(self):
        d = _twostage_panel(seed=11).copy()
        d["cl"] = (d["unit"] // 3).astype(int)
        ref = _fit_twostage(d, est_kw={"cluster": "cl"}, aggregate="all")
        plain = _fit_twostage(d, est_kw={"cluster": "cl"})
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "ts/cl")
        _assert_group_matches_fit_time(plain.aggregate("group"), ref, "ts/cl")

    def test_cluster_naming_unit_column_inert(self, twostage_panel):
        # cluster= legally names the unit column - the kit's column-subset
        # dedup must keep the frame single-labeled (a duplicated column
        # would break df[unit].map in the moved Stage-1 helpers).
        ref = _fit_twostage(twostage_panel, est_kw={"cluster": "unit"}, aggregate="all")
        plain = _fit_twostage(twostage_panel, est_kw={"cluster": "unit"})
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "ts/cl-unit")
        _assert_group_matches_fit_time(plain.aggregate("group"), ref, "ts/cl-unit")

    def test_covariate_arm_inert(self):
        d = _twostage_panel(seed=23).copy()
        rng = np.random.default_rng(23)
        d["x1"] = rng.normal(size=len(d)) + 0.1 * d["period"]
        ref = _fit_twostage(d, aggregate="all", covariates=["x1"])
        plain = _fit_twostage(d, covariates=["x1"])
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "ts/cov")
        _assert_group_matches_fit_time(plain.aggregate("group"), ref, "ts/cov")

    def test_prop5_arm_inert(self):
        d = _twostage_prop5_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = _fit_twostage(d, aggregate="event_study")
            plain = _fit_twostage(d)
            es = plain.aggregate("event_study")
        _assert_es_container_matches_fit_time(es, ref, "ts/prop5")

    def test_survey_tsl_arm_inert(self):
        d, _ = _twostage_survey_panel()
        sd = _twostage_survey_design(psu=True)
        ref = _fit_twostage(d, aggregate="all", survey_design=sd)
        plain = _fit_twostage(d, survey_design=sd)
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "ts/tsl")
        _assert_group_matches_fit_time(plain.aggregate("group"), ref, "ts/tsl")
        gr = plain.aggregate("group")
        df_col = gr.to_dataframe()["df"].to_numpy()
        finite_p = np.isfinite(gr.p_value)
        assert np.isfinite(df_col[finite_p]).all()  # scalar survey-df broadcast

    def test_always_treated_pad_arm_inert(self):
        # The only arm where score_pad_mask/cluster_ids_full are non-None:
        # an implementation storing None unconditionally passes every other
        # arm but breaks inertness here.
        d = _twostage_always_treated_panel()
        sd = _twostage_survey_design(psu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = _fit_twostage(d, aggregate="all", survey_design=sd)
            plain = _fit_twostage(d, survey_design=sd)
            es = plain.aggregate("event_study")
            gr = plain.aggregate("group")
        _assert_es_container_matches_fit_time(es, ref, "ts/pad")
        _assert_group_matches_fit_time(gr, ref, "ts/pad")

    @pytest.mark.parametrize("degenerate", [None, "dropped", "undefined"])
    def test_replicate_arms_level_matched(self, degenerate):
        d, rep_cols = _twostage_survey_panel(replicate=True, degenerate=degenerate)
        sd = _twostage_survey_design(rep_cols)
        ref_es = _fit_twostage(d, aggregate="event_study", survey_design=sd)
        ref_gr = _fit_twostage(d, aggregate="group", survey_design=sd)
        plain = _fit_twostage(d, survey_design=sd)
        _assert_es_container_matches_fit_time(
            plain.aggregate("event_study"), ref_es, f"ts/rep/{degenerate}"
        )
        _assert_group_matches_fit_time(plain.aggregate("group"), ref_gr, f"ts/rep/{degenerate}")

    def test_all_prop5_horizons_survive_early_return(self):
        # Local-review P1 (pre-existing corner in the verbatim-moved code,
        # fixed in this PR): when EVERY non-reference horizon is
        # Proposition-5-unidentified, est_horizons empties and the early
        # return used to DROP the built prop5 rows - real treated horizons
        # reported as absent instead of unidentified. Both the fit-time
        # and post-fit surfaces must keep them as all-NaN rows with
        # n_obs > 0, plus the consolidated Prop-5 warning.
        rng = np.random.default_rng(5)
        rows = []
        for u in range(12):  # cohort 2: pre at t=1, post only at h >= h_bar
            for t in (1, 8, 9, 10):
                rows.append((u, t, rng.normal() + (2.0 if t >= 2 else 0.0), 2))
        for u in range(20, 32):  # cohort 8: pre-periods only
            for t in range(1, 8):
                rows.append((u, t, rng.normal(), 8))
        d = pd.DataFrame(rows, columns=["unit", "period", "outcome", "first_treat"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ref = _fit_twostage(d.copy(), aggregate="event_study")
            plain = _fit_twostage(d.copy())
            es = plain.aggregate("event_study")
        assert any("Proposition 5" in str(w.message) for w in caught)
        for surface_dict in (ref.event_study_effects,):
            prop5 = {h: r for h, r in surface_dict.items() if h != -1}
            assert prop5, "Prop-5 rows must survive the early return"
            for h, r in prop5.items():
                assert h >= 6 and r["n_obs"] > 0
                assert np.isnan(r["effect"]) and np.isnan(r["se"])
        df_ = es.to_dataframe()
        prop5_rows = df_[(df_["event_time"] >= 6)]
        assert len(prop5_rows) > 0
        assert np.isnan(prop5_rows["att"]).all()
        assert (prop5_rows["n"] > 0).all()
        _assert_es_container_matches_fit_time(es, ref, "ts/all-prop5")

    def test_replicate_overall_row_migration_delta(self):
        d, rep_cols = _twostage_survey_panel(replicate=True, degenerate="cohort_zero")
        sd = _twostage_survey_design(rep_cols)
        plain = _fit_twostage(d, survey_design=sd)
        ref_gr = _fit_twostage(d, aggregate="group", survey_design=sd)
        assert plain.overall_se != ref_gr.overall_se
        _assert_group_matches_fit_time(plain.aggregate("group"), ref_gr, "ts/delta")
        dh, rep_h = _twostage_survey_panel(replicate=True)
        sdh = _twostage_survey_design(rep_h)
        # Same BLAS-shape caveat as the Imputation twin above.
        np.testing.assert_allclose(
            _fit_twostage(dh, survey_design=sdh).overall_se,
            _fit_twostage(dh, aggregate="group", survey_design=sdh).overall_se,
            rtol=1e-12,
        )

    def test_pretrends_arm_inert(self, twostage_panel):
        ref = _fit_twostage(twostage_panel, est_kw={"pretrends": True}, aggregate="event_study")
        plain = _fit_twostage(twostage_panel, est_kw={"pretrends": True})
        _assert_es_container_matches_fit_time(plain.aggregate("event_study"), ref, "ts/pretrends")

    def test_isolation_under_public_field_mutation(self, twostage_panel):
        res = _fit_twostage(twostage_panel)
        base_es = res.aggregate("event_study").to_dataframe()
        base_gr = res.aggregate("group").to_dataframe()
        base_sm = res.aggregate("simple").to_dataframe()
        res.groups.pop()
        res.time_periods.pop()
        object.__setattr__(res, "alpha", 0.5)
        object.__setattr__(res, "anticipation", 3)
        pd.testing.assert_frame_equal(res.aggregate("event_study").to_dataframe(), base_es)
        pd.testing.assert_frame_equal(res.aggregate("group").to_dataframe(), base_gr)
        pd.testing.assert_frame_equal(res.aggregate("simple").to_dataframe(), base_sm)

    def test_metadata_never_mutated_and_isolated(self):
        d, rep_cols = _twostage_survey_panel(replicate=True)
        sd = _twostage_survey_design(rep_cols)
        res = _fit_twostage(d, survey_design=sd)
        base = res.aggregate("event_study").to_dataframe()
        base_df_survey = res.aggregate("event_study").df_survey
        res.survey_metadata.df_survey = 999.0
        res.survey_metadata.replicate_method = None
        after = res.aggregate("event_study")
        pd.testing.assert_frame_equal(after.to_dataframe(), base)
        assert after.df_survey == base_df_survey
        assert res.survey_metadata.df_survey == 999.0

    def test_repeated_and_order_independent(self, twostage_fitted):
        a = twostage_fitted.aggregate("group").to_dataframe()
        twostage_fitted.aggregate("event_study")
        twostage_fitted.aggregate("simple")
        b = twostage_fitted.aggregate("group").to_dataframe()
        pd.testing.assert_frame_equal(a, b)

    def test_bootstrap_recompute_levels_fail_closed(self, twostage_panel):
        res = _fit_twostage(twostage_panel, est_kw={"n_bootstrap": 19, "seed": 1})
        for level in ("event_study", "group"):
            with pytest.raises(NotImplementedError, match="bootstrap") as exc:
                res.aggregate(level)
            assert "aggregate('simple') and, where supported, aggregate('total') relay" in str(
                exc.value
            )

    def test_bootstrap_simple_relays_stored_quintet(self, twostage_panel):
        res = _fit_twostage(twostage_panel, est_kw={"n_bootstrap": 19, "seed": 1})
        _assert_bootstrap_simple_relay(
            res, n_expected=res._aggregation_kit.bookkeeping["n_treated_obs"]
        )

    def test_bootstrap_survey_simple_relay_df_nan(self):
        # TSL survey + bootstrap is a supported combination (only replicate
        # designs are rejected under bootstrap); the relay NaNs the df column
        # beside the finite survey metadata.
        d, _ = _twostage_survey_panel()
        sd = _twostage_survey_design()
        res = _fit_twostage(d, est_kw={"n_bootstrap": 19, "seed": 1}, survey_design=sd)
        assert res.survey_metadata is not None
        _assert_bootstrap_simple_relay(res)

    def test_fail_closed_vocabulary(self, twostage_fitted):
        for bad in ("calendar", "all", "nonsense"):
            with pytest.raises(ValueError, match="Unsupported aggregation type"):
                twostage_fitted.aggregate(bad)
        with pytest.raises(ValueError, match="balance_e"):
            twostage_fitted.aggregate("group", balance_e=1)
        with pytest.raises(ValueError):
            twostage_fitted.aggregate("group", weights="cell")

    def test_legacy_pickle_without_kit(self, twostage_fitted):
        import copy

        legacy = copy.copy(twostage_fitted)
        object.__setattr__(legacy, "_aggregation_kit", None)
        with pytest.raises(ValueError, match="aggregation kit"):
            legacy.aggregate("group")

    def test_pickle_roundtrip(self, twostage_panel):
        import pickle

        res = _fit_twostage(twostage_panel)
        clone = pickle.loads(pickle.dumps(res))
        pd.testing.assert_frame_equal(
            clone.aggregate("group").to_dataframe(), res.aggregate("group").to_dataframe()
        )

    def test_empty_balance_window(self, twostage_fitted):
        with pytest.warns(UserWarning, match="balance_e"):
            es = twostage_fitted.aggregate("event_study", balance_e=100)
        df_ = es.to_dataframe()
        assert (df_["is_reference"] | ~np.isfinite(df_["att"])).all()


# --------------------------------------------------------------------------- #
# ContinuousDiD (row M-025): fit(aggregate=) shim + the MIXED view/recompute
# aggregate() - 'simple'/'dose' are pure VIEWS over stored fields (permitted
# on bootstrap fits), 'event_study' is a pruned-IF-payload kit recompute
# (bootstrap fails closed; replicate designs supported, no refit replay)
# --------------------------------------------------------------------------- #

CONT_KW = dict(
    outcome="outcome", unit="unit", time="period", first_treat="first_treat", dose="dose"
)


def _cont_panel(seed=19, n_units=110, n_periods=6, cohort_periods=None):
    from diff_diff import generate_continuous_did_data

    return generate_continuous_did_data(
        n_units=n_units,
        n_periods=n_periods,
        cohort_periods=cohort_periods or [3, 4],
        seed=seed,
    )


def _cont_covariate_panel(seed=23):
    d = _cont_panel(seed=seed)
    rng = np.random.default_rng(seed)
    xmap = {u: rng.normal() for u in d["unit"].unique()}
    d["x1"] = d["unit"].map(xmap)
    return d


def _cont_survey_panel(seed=29, replicate=False, degenerate=None, zero_dose_treated=False):
    d = _cont_panel(seed=seed)
    rng = np.random.default_rng(seed)
    wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
    d["w"] = d["unit"].map(wmap)
    d["strata"] = d["unit"] % 4
    d["psu"] = d["unit"]
    rep_cols = []
    if replicate:
        n_rep = 8
        for r in range(n_rep):
            col = f"rw{r}"
            rep_cols.append(col)
            if degenerate == "dropped" and r >= n_rep - 2:
                d[col] = 0.0
            elif degenerate == "undefined" and r >= 1:
                d[col] = 0.0
            else:
                jitter = {u: rng.uniform(0.1, 2.0) for u in d["unit"].unique()}
                d[col] = d["unit"].map(jitter) * d["w"]
    if zero_dose_treated:
        # Treated units with dose == 0: fires the fit-time drop (with
        # UserWarning) AND the post-drop survey re-resolution - the exact
        # resolved_survey object the kit snapshots.
        treated_units = sorted(d.loc[d["first_treat"] > 0, "unit"].unique())[:4]
        d.loc[d["unit"].isin(treated_units), "dose"] = 0.0
    return d, rep_cols


def _cont_survey_design(rep_cols=None, tsl=True):
    from diff_diff import SurveyDesign

    if rep_cols:
        return SurveyDesign(weights="w", replicate_weights=rep_cols, replicate_method="JK1")
    if tsl:
        return SurveyDesign(weights="w", strata="strata", psu="psu")
    return SurveyDesign(weights="w")


def _cont_discrete_panel(seed=31):
    d = _cont_panel(seed=seed, cohort_periods=[3])
    rng = np.random.default_rng(seed)
    treated = d["first_treat"] > 0
    level_map = {u: float(rng.choice([1.0, 2.0, 3.0])) for u in d.loc[treated, "unit"].unique()}
    d.loc[treated, "dose"] = d.loc[treated, "unit"].map(level_map)
    return d


def _cont_lowest_dose_panel(seed=37, n_units=60):
    """Single cohort, no never-treated units, mass point at d_L."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        d_u = 0.5 if u < 10 else float(rng.uniform(1.0, 3.0))
        g = 3
        for t in range(1, 6):
            treated = t >= g
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "first_treat": g,
                    "dose": d_u,
                    "outcome": u / 25
                    + 0.3 * t
                    + (1.0 + 0.8 * d_u if treated else 0.0)
                    + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _cont_empty_post_panel(seed=41, n_units=50):
    """All treatment starts AFTER the last observed period: pre-period
    (g,t) cells exist but post_gt is empty (the fit-time warning path;
    overall/dose fields are all-NaN and ES rows keep NaN inference)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = 0 if u % 3 == 0 else 5
        d_u = 0.0 if g == 0 else float(rng.uniform(0.5, 2.0))
        for t in range(1, 5):
            rows.append(
                {
                    "unit": u,
                    "period": t,
                    "first_treat": g,
                    "dose": d_u,
                    "outcome": u / 20 + 0.4 * t + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _fit_cont(data, *, est_kw=None, **fit_kw):
    from diff_diff import ContinuousDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ContinuousDiD(**(est_kw or {})).fit(data, **CONT_KW, **fit_kw)


@pytest.fixture(scope="module")
def cont_panel():
    return _cont_panel()


@pytest.fixture(scope="module")
def cont_fitted(cont_panel):
    """Plain fit - the pruned-payload kit powers aggregate('event_study')."""
    return _fit_cont(cont_panel)


@pytest.fixture(scope="module")
def cont_fit_time(cont_panel):
    """Deprecated fit-time aggregate='eventstudy' - the inertness reference."""
    return _fit_cont(cont_panel, aggregate="eventstudy")


@pytest.fixture(scope="module")
def cont_bootstrap(cont_panel):
    return _fit_cont(cont_panel, est_kw=dict(n_bootstrap=30, seed=3))


class TestContinuousShim:
    def test_plain_fit_does_not_warn(self, cont_panel):
        from diff_diff import ContinuousDiD

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ContinuousDiD().fit(cont_panel, **CONT_KW)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_aggregate_kwarg_warns_even_at_none(self, cont_panel):
        from diff_diff import ContinuousDiD

        with pytest.warns(FutureWarning, match=r"ContinuousDiD\.fit\(aggregate=\)"):
            ContinuousDiD().fit(cont_panel, **CONT_KW, aggregate=None)

    def test_dose_value_warns_and_is_inert(self, cont_panel, cont_fitted):
        """aggregate='dose' was always a fit-time no-op - the deprecated
        call warns (the message says so) and produces the identical fit."""
        from diff_diff import ContinuousDiD

        with pytest.warns(FutureWarning, match="already[\\s\\n ]*redundant"):
            res = ContinuousDiD().fit(cont_panel, **CONT_KW, aggregate="dose")
        assert res.overall_att == cont_fitted.overall_att
        assert res.overall_acrt_se == cont_fitted.overall_acrt_se
        np.testing.assert_array_equal(
            res.dose_response_att.effects, cont_fitted.dose_response_att.effects
        )
        assert res.event_study_effects is None

    def test_warn_and_still_work(self, cont_fit_time):
        """The deprecated 'eventstudy' value still computes the fit-time
        surface (legacy routing unchanged until 4.0)."""
        assert cont_fit_time.event_study_effects is not None
        assert any(np.isfinite(v["se"]) for v in cont_fit_time.event_study_effects.values())

    def test_invalid_value_warns_then_raises(self, cont_panel):
        """The PRE-EXISTING fit-time value validation survives the shim
        (a delta vs the EfficientDiD/Imputation no-validation shims):
        unknown strings still raise ValueError - after the warning."""
        from diff_diff import ContinuousDiD

        with pytest.warns(FutureWarning):
            with pytest.raises(ValueError, match="Invalid aggregate"):
                ContinuousDiD().fit(cont_panel, **CONT_KW, aggregate="event_study")


class TestContinuousAggregate:
    # ---------------- event_study: kit-recompute inertness ----------------

    def test_event_study_inert(self, cont_fitted, cont_fit_time):
        es = cont_fitted.aggregate("event_study")
        _assert_es_container_matches_fit_time(es, cont_fit_time, "cont/es")
        # n_kind is a container ATTRIBUTE (not an EVENT_STUDY_SCHEMA
        # column): ContinuousDiD's ES dict carries no count key.
        assert es.n_kind is None

    @pytest.mark.parametrize(
        "arm",
        [
            "multi_cohort",
            "anticipation",
            "covariates",
            "survey_tsl",
            "survey_zero_dose_drop",
            "replicate_healthy",
            "replicate_dropped",
            "replicate_undefined",
            "discrete",
            "lowest_dose",
            "not_yet_treated",
            "nondefault_config",
        ],
    )
    def test_event_study_inert_across_designs(self, arm):
        est_kw, fit_kw = {}, {}
        if arm == "multi_cohort":
            data = _cont_panel(seed=43, cohort_periods=[3, 4, 5])
        elif arm == "anticipation":
            data = _cont_panel(seed=47)
            est_kw = dict(anticipation=1)
        elif arm == "covariates":
            data = _cont_covariate_panel()
            fit_kw = dict(covariates=["x1"])
        elif arm == "survey_tsl":
            data, _ = _cont_survey_panel()
            fit_kw = dict(survey_design=_cont_survey_design())
        elif arm == "survey_zero_dose_drop":
            data, _ = _cont_survey_panel(zero_dose_treated=True)
            fit_kw = dict(survey_design=_cont_survey_design())
        elif arm == "replicate_healthy":
            data, rep = _cont_survey_panel(replicate=True)
            fit_kw = dict(survey_design=_cont_survey_design(rep_cols=rep))
        elif arm == "replicate_dropped":
            data, rep = _cont_survey_panel(replicate=True, degenerate="dropped")
            fit_kw = dict(survey_design=_cont_survey_design(rep_cols=rep))
        elif arm == "replicate_undefined":
            data, rep = _cont_survey_panel(replicate=True, degenerate="undefined")
            fit_kw = dict(survey_design=_cont_survey_design(rep_cols=rep))
        elif arm == "discrete":
            data = _cont_discrete_panel()
            est_kw = dict(treatment_type="discrete")
        elif arm == "lowest_dose":
            data = _cont_lowest_dose_panel()
            est_kw = dict(control_group="lowest_dose")
        elif arm == "not_yet_treated":
            data = _cont_panel(seed=53, cohort_periods=[3, 5])
            est_kw = dict(control_group="not_yet_treated")
        else:  # nondefault_config
            data = _cont_panel(seed=59)
            est_kw = dict(alpha=0.10, base_period="universal")
        plain = _fit_cont(data, est_kw=est_kw, **fit_kw)
        ref = _fit_cont(data, est_kw=est_kw, aggregate="eventstudy", **fit_kw)
        es = plain.aggregate("event_study")
        _assert_es_container_matches_fit_time(es, ref, f"cont/{arm}")

    def test_nondefault_provenance_discriminates_defaults(self):
        """alpha=0.10 / base_period='universal' must SURFACE - a kit that
        hard-codes the defaults would pass mutation-isolation alone."""
        data = _cont_panel(seed=59)
        res = _fit_cont(data, est_kw=dict(alpha=0.10, base_period="universal"))
        es = res.aggregate("event_study")
        assert es.alpha == 0.10
        assert es.base_period == "universal"
        s = res.aggregate("simple")
        assert s.alpha == 0.10

    def test_empty_post_gt_all_levels(self):
        """Fit-faithful empty-post_gt quirk: ES rows keep NaN inference on
        BOTH routes; the views relay the stored all-NaN fields."""
        data = _cont_empty_post_panel()
        plain = _fit_cont(data)
        ref = _fit_cont(data, aggregate="eventstudy")
        es = plain.aggregate("event_study")
        _assert_es_container_matches_fit_time(es, ref, "cont/empty_post")
        frame = es.to_dataframe()
        assert not np.isfinite(frame["se"]).any()
        s = plain.aggregate("simple")
        assert np.isnan(s.att).all() and np.isnan(s.se).all()
        d = plain.aggregate("dose")
        assert np.isnan(d.att).all() and np.isnan(d.se).all()

    # ---------------- dose view: per-curve to_dataframe parity ----------------

    def test_dose_view_per_curve_parity(self, cont_fitted):
        agg = cont_fitted.aggregate("dose")
        n = len(cont_fitted.dose_grid)
        assert list(agg.target) == ["att"] * n + ["acrt"] * n
        for block, curve in (
            (slice(0, n), cont_fitted.dose_response_att),
            (slice(n, 2 * n), cont_fitted.dose_response_acrt),
        ):
            frame = curve.to_dataframe()
            np.testing.assert_array_equal(agg.att[block], frame["effect"].to_numpy())
            np.testing.assert_array_equal(agg.se[block], frame["se"].to_numpy())
            np.testing.assert_array_equal(
                agg.conf_int_lower[block], frame["conf_int_lower"].to_numpy()
            )
            np.testing.assert_allclose(
                agg.t_stat[block], frame["t_stat"].to_numpy(), rtol=0, atol=0, equal_nan=True
            )
            np.testing.assert_allclose(
                agg.p_value[block], frame["p_value"].to_numpy(), rtol=0, atol=0, equal_nan=True
            )
        assert agg.n_kind is None and agg.weight is None
        assert np.isnan(agg.n).all()

    def test_dose_view_survey_df(self):
        """Survey fits: finite df threads into the derived t/p AND the df
        column; the per-curve to_dataframe oracle still holds."""
        data, _ = _cont_survey_panel()
        res = _fit_cont(data, survey_design=_cont_survey_design())
        assert res.dose_response_att.df_survey is not None
        agg = res.aggregate("dose")
        n = len(res.dose_grid)
        frame = res.dose_response_att.to_dataframe()
        np.testing.assert_allclose(
            agg.p_value[:n], frame["p_value"].to_numpy(), rtol=0, atol=0, equal_nan=True
        )
        finite = np.isfinite(agg.p_value)
        assert np.isfinite(agg.df[finite]).all()

    def test_dose_view_replicate_undefined_sentinel(self):
        """The replicate-undefined 0-sentinel: NaN t/p via safe_inference
        (the raw stored value feeds the derivation) and NaN df column."""
        data, rep = _cont_survey_panel(replicate=True, degenerate="undefined")
        res = _fit_cont(data, survey_design=_cont_survey_design(rep_cols=rep))
        assert res.dose_response_att.df_survey == 0
        agg = res.aggregate("dose")
        assert np.isnan(agg.t_stat).all() and np.isnan(agg.p_value).all()
        assert np.isnan(agg.df).all()

    def test_dose_view_bootstrap_relays_stored_p(self, cont_bootstrap):
        agg = cont_bootstrap.aggregate("dose")
        n = len(cont_bootstrap.dose_grid)
        assert np.isnan(agg.t_stat).all()
        np.testing.assert_array_equal(
            agg.p_value[:n], np.asarray(cont_bootstrap.dose_response_att.p_value, dtype=float)
        )
        assert np.isnan(agg.df).all()

    # ---------------- simple view ----------------

    def test_simple_view_bit_exact(self, cont_fitted):
        s = cont_fitted.aggregate("simple")
        assert list(s.label) == ["overall", "overall"]
        assert list(s.target) == ["att", "acrt"]
        assert s.att[0] == cont_fitted.overall_att
        assert s.att[1] == cont_fitted.overall_acrt
        assert s.se[0] == cont_fitted.overall_att_se
        assert s.se[1] == cont_fitted.overall_acrt_se
        assert s.t_stat[0] == cont_fitted.overall_att_t_stat
        assert s.p_value[1] == cont_fitted.overall_acrt_p_value
        assert s.conf_int_lower[0] == cont_fitted.overall_att_conf_int[0]
        assert s.conf_int_upper[1] == cont_fitted.overall_acrt_conf_int[1]
        # Disjoint treated/control unit sets -> the CS total convention.
        assert s.n_kind == "units"
        expected_n = float(cont_fitted.n_treated_units + cont_fitted.n_control_units)
        assert (s.n == expected_n).all()
        np.testing.assert_array_equal(s.weight, [1.0, 1.0])

    def test_simple_view_bootstrap_finite_t_relays(self, cont_bootstrap):
        """fit() stores a FINITE safe_inference t beside the percentile
        p/CI on bootstrap fits - the relay carries it through verbatim
        (bit-exact relay, NOT NaN); only the df column is NaN."""
        s = cont_bootstrap.aggregate("simple")
        assert s.t_stat[0] == cont_bootstrap.overall_att_t_stat
        assert np.isfinite(s.t_stat[0])
        assert s.p_value[0] == cont_bootstrap.overall_att_p_value
        assert np.isnan(s.df).all()

    def test_simple_view_survey_df(self):
        data, _ = _cont_survey_panel()
        res = _fit_cont(data, survey_design=_cont_survey_design())
        s = res.aggregate("simple")
        assert np.isfinite(s.df[0])
        assert s.df[0] == float(res.dose_response_att.df_survey)

    # ---------------- heterogeneous-target rendering ----------------

    def test_summary_renders_target_column(self, cont_fitted):
        s_text = cont_fitted.aggregate("simple").summary()
        assert "target" in s_text and "estimate" in s_text
        assert "acrt" in s_text
        # The uniform-target 'ATT' heading must NOT appear as a column head.
        header_line = [ln for ln in s_text.splitlines() if "estimate" in ln][0]
        assert "ATT" not in header_line

    def test_uniform_target_summary_byte_stable(self, fitted):
        """A uniform-target producer's summary() renders EXACTLY as before
        the heterogeneous-target amendment (no target column, ATT head)."""
        s_text = fitted.aggregate("simple").summary()
        assert "target" not in s_text
        assert "ATT" in s_text

    def test_dose_ordering_att_block_first_with_unsorted_dvals(self):
        """FIRST-APPEARANCE target blocks (att then acrt - NOT lexicographic,
        which would invert) with labels ascending WITHIN each block; the
        custom dvals grid is unsorted so within-block sorting is actually
        exercised (the default grid is ascending by construction)."""
        data = _cont_panel(seed=61)
        dvals = np.array([2.0, 1.0, 1.5])
        res = _fit_cont(data, est_kw=dict(dvals=dvals))
        frame = res.aggregate("dose").to_dataframe()
        n = len(dvals)
        assert list(frame["target"]) == ["att"] * n + ["acrt"] * n
        att_labels = frame["label"][:n].astype(float).to_numpy()
        acrt_labels = frame["label"][n:].astype(float).to_numpy()
        np.testing.assert_array_equal(att_labels, np.sort(dvals))
        np.testing.assert_array_equal(acrt_labels, np.sort(dvals))

    def test_mixed_type_labels_preserve_producer_order(self):
        """The _sortable fallback survives the heterogeneous-target branch:
        mixed-type labels keep producer order per block, never raise."""
        agg = AggregationResult(
            level="dose",
            label=np.array(["b", 2, "a", 1], dtype=object),
            target=np.array(["att", "att", "acrt", "acrt"], dtype=object),
            att=np.zeros(4),
            se=np.ones(4),
            t_stat=np.zeros(4),
            p_value=np.ones(4),
            conf_int_lower=np.zeros(4),
            conf_int_upper=np.zeros(4),
            n=np.full(4, np.nan),
            df=np.full(4, np.nan),
        )
        frame = agg.to_dataframe()
        assert list(frame["label"]) == ["b", 2, "a", 1]
        assert list(frame["target"]) == ["att", "att", "acrt", "acrt"]
        assert "target" in agg.summary()

    # ---------------- bootstrap gating + kit shape ----------------

    def test_bootstrap_event_study_fails_closed(self, cont_bootstrap):
        with pytest.raises(NotImplementedError, match="bootstrap"):
            cont_bootstrap.aggregate("event_study")

    def test_bootstrap_views_still_work(self, cont_bootstrap):
        assert cont_bootstrap.aggregate("simple") is not None
        assert cont_bootstrap.aggregate("dose") is not None

    def test_bootstrap_kit_is_scalars_only(self, cont_bootstrap):
        """Dead-retention guarantee: the ES payload can never be consumed
        on a bootstrap fit, so it is not retained."""
        bk = cont_bootstrap._aggregation_kit.bookkeeping
        assert bk["gt_es_payload"] == {}
        assert bk["gt_summary"] == {}
        assert bk["n_units"] is None
        assert bk["unit_cohorts"] is None
        assert bk["resolved_survey"] is None
        assert bk["n_bootstrap"] == 30

    # ---------------- isolation (ES route only - views are views) ----------------

    def test_es_isolation_from_public_field_mutation(self, cont_panel):
        res = _fit_cont(cont_panel)
        baseline = res.aggregate("event_study").to_dataframe()
        res.event_study_effects = {99: {"effect": 1.0}}
        res.group_time_effects = {}
        res.groups = []
        res.alpha = 0.5
        res.anticipation = 7
        res.base_period = "mutated"
        if res.survey_metadata is not None:
            res.survey_metadata.df_survey = -1
        again = res.aggregate("event_study").to_dataframe()
        pd.testing.assert_frame_equal(baseline, again)

    def test_bootstrap_gate_reads_kit_not_field(self, cont_panel):
        """The sharpest isolation arm: mutating res.n_bootstrap = 0 must
        NOT bypass the fail-closed gate (it reads the kit)."""
        res = _fit_cont(cont_panel, est_kw=dict(n_bootstrap=30, seed=3))
        res.n_bootstrap = 0
        with pytest.raises(NotImplementedError, match="bootstrap"):
            res.aggregate("event_study")

    def test_repeated_calls_idempotent(self, cont_fitted):
        a = cont_fitted.aggregate("event_study").to_dataframe()
        b = cont_fitted.aggregate("event_study").to_dataframe()
        pd.testing.assert_frame_equal(a, b)

    def test_aggregate_does_not_mutate_survey_metadata(self):
        data, rep = _cont_survey_panel(replicate=True)
        res = _fit_cont(data, survey_design=_cont_survey_design(rep_cols=rep))
        before = copy.deepcopy(res.survey_metadata.__dict__)
        res.aggregate("event_study")
        res.aggregate("simple")
        res.aggregate("dose")
        assert res.survey_metadata.__dict__ == before

    # ---------------- no-kit legacy + vocabulary ----------------

    def test_no_kit_es_raises_views_work(self, cont_panel):
        res = _fit_cont(cont_panel)
        object.__setattr__(res, "_aggregation_kit", None)
        with pytest.raises(ValueError, match="aggregation kit"):
            res.aggregate("event_study")
        # Views need no kit - deliberately still work on legacy pickles.
        assert res.aggregate("simple") is not None
        assert res.aggregate("dose") is not None

    @pytest.mark.parametrize("bad", ["group", "calendar", "all", "nonsense"])
    def test_unsupported_types_fail_closed(self, cont_fitted, bad):
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            cont_fitted.aggregate(bad)

    @pytest.mark.parametrize("level", ["simple", "event_study", "dose"])
    def test_balance_e_rejected_empty_vocabulary(self, cont_fitted, level):
        with pytest.raises(ValueError, match="no aggregation type on this estimator"):
            cont_fitted.aggregate(level, balance_e=1)

    def test_weights_rejected(self, cont_fitted):
        with pytest.raises(ValueError, match="does not accept a weights selector"):
            cont_fitted.aggregate("simple", weights="cell")

    # ---------------- pickle + retention ----------------

    def test_pickle_round_trip(self, cont_fitted):
        clone = pickle.loads(pickle.dumps(cont_fitted))
        a = cont_fitted.aggregate("event_study").to_dataframe()
        b = clone.aggregate("event_study").to_dataframe()
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(
            cont_fitted.aggregate("dose").to_dataframe(),
            clone.aggregate("dose").to_dataframe(),
        )

    def test_no_raw_unit_identifiers_are_retained(self, cont_fitted):
        """The kit stores positional indices and first_treat cohort values
        only - never the raw unit identifier column."""
        bk = cont_fitted._aggregation_kit.bookkeeping

        def _walk(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from _walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    yield from _walk(v)
            else:
                yield obj

        for leaf in _walk({k: v for k, v in bk.items() if k != "survey_metadata"}):
            assert not isinstance(leaf, pd.DataFrame), "kit retains a DataFrame"
            assert not isinstance(leaf, pd.Series), "kit retains a Series"


# --------------------------------------------------------------------------- #
# HeterogeneousAdoptionDiD (rows M-027/M-139): the fit + workflow MODE-SELECTOR
# shims (panel-shape inference) and the two PURE-VIEW results classes.
# --------------------------------------------------------------------------- #

HAD_KW = dict(outcome="outcome", dose="dose", time="period", unit="unit")


def _had_panel_2p(seed=11, n_units=200, mass_point=False, constant_outcome=False):
    """Two-period HAD panel (the 'overall' mode shape)."""
    rng = np.random.default_rng(seed)
    if mass_point:
        d = np.where(rng.uniform(size=n_units) < 0.5, 0.5, rng.uniform(1.0, 2.0, n_units))
    else:
        d = rng.uniform(0.1, 2.0, n_units)
    rows = []
    for i in range(n_units):
        y0 = 0.0 if constant_outcome else rng.normal()
        y1 = 0.0 if constant_outcome else 1.5 * d[i] + rng.normal()
        rows.append((i, 0, 0.0, y0, i % 5))
        rows.append((i, 1, d[i], y1, i % 5))
    return pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome", "grp"])


def _had_panel_multi(seed=13, n_units=200, n_periods=5, F=2):
    """Multi-period common-adoption HAD panel (the 'event_study' mode shape)."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.1, 2.0, n_units)
    rows = []
    for i in range(n_units):
        for t in range(n_periods):
            dose_it = d[i] if t >= F else 0.0
            rows.append((i, t, dose_it, 1.5 * dose_it + rng.normal(), F))
    return pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome", "ft"])


def _had_panel_post_filter_too_small(seed=17, n_units=120):
    """Staggered T>2 panel whose last-cohort auto-filter leaves < 3 periods.

    Earlier-cohort units (first_treat=2) span periods 1-4; last-cohort
    (first_treat=4) and never-treated units are observed ONLY at periods
    3-4. Raw distinct periods = 4 -> the sentinel infers event_study; the
    Appendix-B.2 filter keeps only the last cohort + never-treated, whose
    observed periods are {3, 4} -> the post-filter shape error fires on a
    PLAIN fit (no FutureWarning).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        kind = i % 3  # 0: early cohort, 1: last cohort, 2: never-treated
        if kind == 0:
            periods, ft, dose = range(1, 5), 2, rng.uniform(0.5, 2.0)
        elif kind == 1:
            periods, ft, dose = (3, 4), 4, rng.uniform(0.5, 2.0)
        else:
            periods, ft, dose = (3, 4), 0, 0.0
        for t in periods:
            treated = ft > 0 and t >= ft
            d_it = dose if treated else 0.0
            rows.append((i, t, d_it, 0.5 * d_it + rng.normal(), ft))
    return pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome", "ft"])


def _fit_had(data, *, est_kw=None, **fit_kw):
    from diff_diff import HeterogeneousAdoptionDiD

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HeterogeneousAdoptionDiD(**(est_kw or {})).fit(data, **HAD_KW, **fit_kw)


@pytest.fixture(scope="module")
def had_panel_2p():
    return _had_panel_2p()


@pytest.fixture(scope="module")
def had_panel_multi():
    return _had_panel_multi()


@pytest.fixture(scope="module")
def had_fitted_2p(had_panel_2p):
    return _fit_had(had_panel_2p)


@pytest.fixture(scope="module")
def had_fitted_multi(had_panel_multi):
    return _fit_had(had_panel_multi, first_treat="ft")


class TestHadShim:
    def test_plain_fit_2p_infers_overall_without_warning(self, had_panel_2p):
        from diff_diff import HeterogeneousAdoptionDiD, HeterogeneousAdoptionDiDResults

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = HeterogeneousAdoptionDiD().fit(had_panel_2p, **HAD_KW)
        assert isinstance(res, HeterogeneousAdoptionDiDResults)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_plain_fit_multi_infers_event_study_without_warning(self, had_panel_multi):
        # The headline M-027 behavior delta: a plain multi-period fit()
        # previously raised the two-period shape error; the sentinel now
        # infers the event-study mode (error -> works).
        from diff_diff import (
            HeterogeneousAdoptionDiD,
            HeterogeneousAdoptionDiDEventStudyResults,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = HeterogeneousAdoptionDiD().fit(had_panel_multi, **HAD_KW, first_treat="ft")
        assert isinstance(res, HeterogeneousAdoptionDiDEventStudyResults)
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    @pytest.mark.parametrize("mode", ["overall", "event_study"])
    def test_supplied_mode_warns_and_still_works(self, had_panel_2p, had_panel_multi, mode):
        from diff_diff import HeterogeneousAdoptionDiD

        data = had_panel_2p if mode == "overall" else had_panel_multi
        kw = {} if mode == "overall" else {"first_treat": "ft"}
        with pytest.warns(FutureWarning, match=r"fit\(aggregate=\) is deprecated"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                HeterogeneousAdoptionDiD().fit(data, **HAD_KW, aggregate=mode, **kw)

    def test_aggregate_none_warns_then_raises(self, had_panel_2p):
        # Unlike ContinuousDiD, None is NOT in HAD's _VALID_AGGREGATES: an
        # explicit None is a supplied INVALID value - a third distinct
        # behavior beside omitted (infers) and supplied-valid (legacy).
        from diff_diff import HeterogeneousAdoptionDiD

        with pytest.warns(FutureWarning, match="aggregate"):
            with pytest.raises(ValueError, match="Invalid aggregate=None"):
                HeterogeneousAdoptionDiD().fit(had_panel_2p, **HAD_KW, aggregate=None)

    def test_invalid_value_warns_then_raises(self, had_panel_2p):
        from diff_diff import HeterogeneousAdoptionDiD

        with pytest.warns(FutureWarning, match="aggregate"):
            with pytest.raises(ValueError, match="Invalid aggregate='bogus'"):
                HeterogeneousAdoptionDiD().fit(had_panel_2p, **HAD_KW, aggregate="bogus")

    def test_supplied_overall_on_multi_warns_then_shape_error(self, had_panel_multi):
        from diff_diff import HeterogeneousAdoptionDiD

        with pytest.warns(FutureWarning, match="aggregate"):
            with pytest.raises(ValueError, match="exactly two time periods"):
                HeterogeneousAdoptionDiD().fit(
                    had_panel_multi, **HAD_KW, aggregate="overall", first_treat="ft"
                )

    def test_sentinel_post_filter_shape_error_no_warning(self):
        # Sentinel-reachable shape error: T>2 raw panel infers event_study,
        # then the last-cohort auto-filter drops it below three periods.
        # The message must read correctly on a plain fit (no kwarg
        # teaching) and no FutureWarning fires.
        from diff_diff import HeterogeneousAdoptionDiD

        data = _had_panel_post_filter_too_small()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="staggered auto-filter"):
                HeterogeneousAdoptionDiD().fit(data, **HAD_KW, first_treat="ft")
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_missing_time_column_still_value_error(self, had_panel_2p):
        # The inference helper must not regress the tested ValueError to a
        # raw KeyError (it dereferences the time column first).
        from diff_diff import HeterogeneousAdoptionDiD

        with pytest.raises(ValueError, match="column"):
            HeterogeneousAdoptionDiD().fit(
                had_panel_2p, outcome="outcome", dose="dose", time="missing", unit="unit"
            )

    @pytest.mark.parametrize("shape", ["2p", "multi"])
    def test_inference_equivalence_bit_identical(self, had_panel_2p, had_panel_multi, shape):
        """Plain fit ≡ fit(aggregate=<mode>) field-for-field per shape."""
        if shape == "2p":
            plain = _fit_had(had_panel_2p)
            legacy = _fit_had(had_panel_2p, aggregate="overall")
        else:
            plain = _fit_had(had_panel_multi, first_treat="ft")
            legacy = _fit_had(had_panel_multi, first_treat="ft", aggregate="event_study")
        import dataclasses

        assert type(plain) is type(legacy)
        for f in dataclasses.fields(plain):
            a, b = getattr(plain, f.name), getattr(legacy, f.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            elif isinstance(a, float):
                assert (a == b) or (np.isnan(a) and np.isnan(b)), f.name
            elif isinstance(a, (list, tuple)) and a and isinstance(a[0], float):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            elif f.name in (
                "bandwidth_diagnostics",
                "bias_corrected_fit",
                "survey_metadata",
                "filter_info",
            ):
                assert (a is None) == (b is None), f.name
            else:
                assert a == b, f.name

    def test_workflow_plain_call_infers_without_warning(self, had_panel_2p):
        from diff_diff import did_had_pretest_workflow

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rep = did_had_pretest_workflow(had_panel_2p, **HAD_KW, n_bootstrap=99, seed=3)
        assert rep.aggregate == "overall"
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []

    def test_workflow_supplied_mode_warns(self, had_panel_2p):
        from diff_diff import did_had_pretest_workflow

        with pytest.warns(FutureWarning, match=r"workflow\(aggregate=\) is deprecated"):
            did_had_pretest_workflow(
                had_panel_2p, **HAD_KW, n_bootstrap=99, seed=3, aggregate="overall"
            )

    def test_workflow_invalid_value_warns_then_raises(self, had_panel_2p):
        from diff_diff import did_had_pretest_workflow

        with pytest.warns(FutureWarning, match="aggregate"):
            with pytest.raises(ValueError, match="aggregate must be one of"):
                did_had_pretest_workflow(had_panel_2p, **HAD_KW, aggregate="junk")


class TestHadAggregate:
    def _assert_simple_relay(self, res):
        agg = res.aggregate("simple")
        assert isinstance(agg, AggregationResult)
        assert agg.level == "simple"
        assert list(agg.label) == ["overall"]
        assert list(agg.target) == [res.target_parameter]
        for got, want in (
            (agg.att[0], res.att),
            (agg.se[0], res.se),
            (agg.t_stat[0], res.t_stat),
            (agg.p_value[0], res.p_value),
            (agg.conf_int_lower[0], res.conf_int[0]),
            (agg.conf_int_upper[0], res.conf_int[1]),
        ):
            assert (float(got) == want) or (np.isnan(got) and np.isnan(want))
        assert float(agg.n[0]) == float(res.n_obs)
        assert agg.n_kind == "units"
        assert agg.estimator == "HeterogeneousAdoptionDiD"
        return agg

    def test_simple_view_bit_exact_continuous(self, had_fitted_2p):
        agg = self._assert_simple_relay(had_fitted_2p)
        assert had_fitted_2p.target_parameter in ("WAS", "WAS_d_lower")
        # Continuous designs: disjoint treated/control split.
        assert had_fitted_2p.n_treated + had_fitted_2p.n_control == had_fitted_2p.n_obs
        # Plain fit passed df=None into safe_inference: NaN df column.
        assert np.isnan(agg.df[0])

    def test_simple_view_bit_exact_mass_point(self):
        res = _fit_had(_had_panel_2p(mass_point=True))
        assert res.design == "mass_point"
        # NO n identity assert here: the mass-point control mask uses a
        # tolerance while treated is strict, so the sets can overlap in a
        # ~1-ULP band; n_obs is the single authoritative count.
        self._assert_simple_relay(res)

    def test_simple_view_clustered(self, had_panel_2p):
        res = _fit_had(had_panel_2p, est_kw={"cluster": "grp"})
        agg = self._assert_simple_relay(res)
        assert np.isnan(agg.df[0])

    def test_simple_view_survey_tsl_df_provenance(self, had_panel_2p):
        from diff_diff import SurveyDesign

        d = had_panel_2p.copy()
        rng = np.random.default_rng(23)
        wmap = {u: rng.uniform(0.5, 2.0) for u in d["unit"].unique()}
        d["w"] = d["unit"].map(wmap)
        d["psu"] = d["unit"] % 9
        res = _fit_had(d, survey_design=SurveyDesign(weights="w", psu="psu"))
        agg = self._assert_simple_relay(res)
        # The survey path passed resolved.df_survey into safe_inference and
        # mirrors it on survey_metadata: the relay is provenance-exact.
        assert res.survey_metadata is not None
        assert float(agg.df[0]) == float(res.survey_metadata.df_survey)

    def test_simple_view_constant_outcome_all_nan_relay(self):
        # Degenerate fit contract (had.py class docstring): constant outcome
        # on the continuous paths returns (att=nan, se=nan) and the
        # safe_inference gate NaNs the downstream triple. The relay carries
        # the NaN quintet honestly and __post_init__ NaNs the df column
        # wherever p is non-finite.
        res = _fit_had(_had_panel_2p(constant_outcome=True))
        agg = self._assert_simple_relay(res)
        assert np.isnan(agg.att[0]) and np.isnan(agg.se[0])
        assert np.isnan(agg.t_stat[0]) and np.isnan(agg.p_value[0])
        assert np.isnan(agg.df[0])

    def test_simple_view_single_cluster_finite_att_nan_inference(self):
        # Mass-point single-cluster: att stays finite (the Wald-IV ratio is
        # well defined) while the CR1 SE is NaN, so the downstream triple is
        # NaN via the safe_inference gate - the relay is bit-exact on that
        # mixed state too.
        d = _had_panel_2p(mass_point=True).copy()
        d["one"] = 0
        res = _fit_had(d, est_kw={"cluster": "one"})
        assert np.isfinite(res.att) and np.isnan(res.se)
        agg = self._assert_simple_relay(res)
        assert np.isfinite(agg.att[0]) and np.isnan(agg.se[0])
        assert np.isnan(agg.t_stat[0]) and np.isnan(agg.p_value[0])
        assert np.isnan(agg.df[0])

    def test_simple_summary_estimand_heading(self, had_fitted_2p):
        # The M-027 heading widening: a single non-'att' target renders the
        # target column + neutral 'estimate' heading (never the hard-coded
        # ATT heading, which would mislabel a WAS).
        s = had_fitted_2p.aggregate("simple").summary()
        assert "estimate" in s
        assert had_fitted_2p.target_parameter in s
        assert not any("ATT" in line and "label" in line for line in s.splitlines())

    def test_dcdh_simple_summary_estimand_heading(self, dcdh_fitted):
        # The widening also fixes dCDH's shipped mislabel: its estimand-
        # labelled single-row relay previously rendered under 'ATT'.
        s = dcdh_fitted.aggregate("simple").summary()
        assert "estimate" in s
        assert "DID_M" in s
        assert not any("ATT" in line and "label" in line for line in s.splitlines())

    def test_event_study_view_matches_builder(self, had_fitted_multi):
        from diff_diff.results_base import build_event_study_surface

        es = had_fitted_multi.aggregate("event_study")
        assert isinstance(es, EventStudyResults)
        assert es.n_kind == "units"  # the corrected _from_had kind
        built = build_event_study_surface(had_fitted_multi)
        a, b = es.to_dataframe(), built.to_dataframe()
        assert list(a.columns) == list(b.columns)
        assert a.shape == b.shape
        for col in a.columns:
            av, bv = a[col].to_numpy(), b[col].to_numpy()
            if av.dtype.kind in "fc":
                np.testing.assert_allclose(
                    av.astype(float), bv.astype(float), rtol=0, atol=0, equal_nan=True
                )
            else:
                assert list(av) == list(bv)

    def test_event_study_view_estimand_labels(self, had_fitted_multi, had_panel_multi):
        # CI review R1 P1: the container must carry the WAS estimand label -
        # summary() previously rendered HAD's numbers under a hardcoded ATT
        # heading with no estimand metadata anywhere on the surface.
        es = had_fitted_multi.aggregate("event_study")
        assert es.estimand == had_fitted_multi.target_parameter
        assert es.estimand in ("WAS", "WAS_d_lower")
        s = es.summary()
        heading = next(line for line in s.splitlines() if "Event time" in line)
        assert es.estimand in heading and "ATT" not in heading
        assert f"estimand: {es.estimand}" in s
        assert es.to_dict()["estimand"] == es.estimand
        # The detached frame carries the discriminator too (CI review R2):
        # a bare att column would be indistinguishable from an ATT.
        frame = es.to_dataframe()
        assert list(frame["estimand"].unique()) == [es.estimand]

    def test_event_study_view_estimand_was_at_zero(self):
        # The continuous_at_zero design labels the estimand "WAS" (vs the
        # near-d_lower fixture's "WAS_d_lower") - both must relay.
        rng = np.random.default_rng(31)
        n = 150
        d = rng.uniform(0.0, 2.0, n)
        d[0] = 0.0
        rows = []
        for i in range(n):
            ft = 0 if d[i] == 0.0 else 2
            for t in range(5):
                di = d[i] if t >= 2 else 0.0
                rows.append((i, t, di, 1.5 * di + rng.normal(), ft))
        pm = pd.DataFrame(rows, columns=["unit", "period", "dose", "outcome", "ft"])
        res = _fit_had(pm, first_treat="ft")
        assert res.target_parameter == "WAS"
        es = res.aggregate("event_study")
        assert es.estimand == "WAS"
        heading = next(line for line in es.summary().splitlines() if "Event time" in line)
        assert " WAS " in heading and "ATT" not in heading

    def test_event_study_view_non_had_estimand_none_att_stable(self, fitted):
        # Byte-stability: every other producer's estimand stays None and the
        # ATT heading is unchanged (no estimand metadata line).
        es = fitted.aggregate("event_study")
        assert es.estimand is None
        s = es.summary()
        heading = next(line for line in s.splitlines() if "Event time" in line)
        assert " ATT " in heading
        assert "estimand:" not in s
        assert list(es.to_dataframe()["estimand"].unique()) == ["att"]

    def test_event_study_view_cband_relays(self, had_panel_multi):
        # cluster= fires the clustered sup-t band even on an unweighted fit;
        # the view must carry the cband fields through _from_had.
        d = had_panel_multi.copy()
        d["grp"] = d["unit"] % 6
        res = _fit_had(
            d, first_treat="ft", est_kw={"cluster": "grp", "n_bootstrap": 199, "seed": 5}
        )
        assert res.cband_low is not None
        es = res.aggregate("event_study")
        np.testing.assert_array_equal(es.cband_lower, np.asarray(res.cband_low))
        np.testing.assert_array_equal(es.cband_upper, np.asarray(res.cband_high))
        assert es.cband_crit_value == res.cband_crit_value

    def test_cross_level_fail_closed_overall_class(self, had_fitted_2p):
        for bad in ("event_study", "group", "calendar", "dose", "all"):
            with pytest.raises(ValueError, match="Unsupported aggregation type"):
                had_fitted_2p.aggregate(bad)

    def test_cross_level_fail_closed_es_class(self, had_fitted_multi):
        for bad in ("simple", "group", "calendar", "all"):
            with pytest.raises(ValueError, match="Unsupported aggregation type"):
                had_fitted_multi.aggregate(bad)

    def test_balance_e_and_weights_rejected(self, had_fitted_2p, had_fitted_multi):
        with pytest.raises(ValueError, match="balance_e"):
            had_fitted_2p.aggregate("simple", balance_e=1)
        with pytest.raises(ValueError, match="balance_e"):
            had_fitted_multi.aggregate("event_study", balance_e=1)
        with pytest.raises(ValueError, match="weights"):
            had_fitted_2p.aggregate("simple", weights="cell")

    def test_idempotent_and_isolated(self, had_fitted_2p, had_fitted_multi):
        a1 = had_fitted_2p.aggregate("simple")
        a2 = had_fitted_2p.aggregate("simple")
        np.testing.assert_array_equal(a1.att, a2.att)
        e1 = had_fitted_multi.aggregate("event_study").to_dataframe()
        e2 = had_fitted_multi.aggregate("event_study").to_dataframe()
        assert e1.equals(e2)

    def test_pickle_round_trip_no_kit(self, had_fitted_2p, had_fitted_multi):
        # PURE VIEWS: no kit is attached, so results unpickled from ANY
        # release aggregate identically - there is no no-kit error path.
        import pickle

        assert not hasattr(had_fitted_2p, "_aggregation_kit")
        rt2 = pickle.loads(pickle.dumps(had_fitted_2p))
        np.testing.assert_array_equal(
            rt2.aggregate("simple").att, had_fitted_2p.aggregate("simple").att
        )
        rtm = pickle.loads(pickle.dumps(had_fitted_multi))
        assert (
            rtm.aggregate("event_study")
            .to_dataframe()
            .equals(had_fitted_multi.aggregate("event_study").to_dataframe())
        )

    def test_no_future_warning_from_post_fit_routes(self, had_fitted_2p, had_fitted_multi):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            had_fitted_2p.aggregate("simple")
            had_fitted_multi.aggregate("event_study")
        assert [w for w in caught if issubclass(w.category, FutureWarning)] == []


# =============================================================================
# aggregate('total') - the estimator-owned total incremental outcome
# (exact relay C x overall, CONDITIONAL on the realized aggregation mass)
# =============================================================================


def _raw_treated_obs(frame, time_col="time"):
    """Frame-derived treated unit-period count (the container-independent oracle)."""
    return int(((frame["first_treat"] > 0) & (frame[time_col] >= frame["first_treat"])).sum())


class TestTotalCallawaySantAnna:
    """CS total: relay identity, mass oracles, and the fail-closed routings."""

    def test_identity_and_independent_mass(self, panel, fitted):
        tot = fitted.aggregate("total")
        simple = fitted.aggregate("simple")
        assert tot.level == "total"
        assert tot.label[0] == "total" and tot.target[0] == "total"
        assert tot.n_kind == "obs" and tot.weight[0] == 1.0
        # Independent mass oracles: kept finite post cells AND the raw frame
        # (clean balanced DGP, so the two coincide).
        n_cells = sum(
            d["n_treated"]
            for (g, t), d in fitted.group_time_effects.items()
            if t >= g and np.isfinite(d["effect"])
        )
        assert tot.n[0] == float(n_cells) == float(_raw_treated_obs(panel))
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.allclose(tot.se[0], tot.n[0] * simple.se[0], rtol=1e-12)
        assert tot.t_stat[0] == simple.t_stat[0]
        assert tot.p_value[0] == simple.p_value[0]
        assert np.allclose(tot.conf_int_lower[0], tot.n[0] * simple.conf_int_lower[0], rtol=1e-12)
        assert np.allclose(tot.conf_int_upper[0], tot.n[0] * simple.conf_int_upper[0], rtol=1e-12)

    def test_df_carrier_plain_panel_is_nan(self, fitted):
        # A plain (non-cluster, non-survey) analytical CS fit has no survey df
        # and no df_inference, so the total row's df provenance column is NaN -
        # same carrier as the simple relay.
        tot = fitted.aggregate("total")
        simple = fitted.aggregate("simple")
        assert np.isnan(tot.df[0]) == np.isnan(simple.df[0])

    def test_bare_cluster_admitted_with_finite_df(self, panel):
        """cluster= (no survey_design) is a mainstream ADMITTED routing: the
        synthesized all-ones design is not a survey. On a clean panel the
        cohort-mass branch coincides with the complete-case count, and the
        fit's cluster df flows into the total row's df carrier (clean
        fixtures alone would hide a wrong carrier)."""
        from diff_diff.aggregation import resolve_inference_df

        res = CallawaySantAnna(cluster="unit").fit(panel, **FIT_KW)
        tot = res.aggregate("total")
        assert tot.n[0] == float(_raw_treated_obs(panel))
        expected_df = resolve_inference_df(res)
        if expected_df is not None and np.isfinite(expected_df):
            assert tot.df[0] == expected_df
        assert np.allclose(tot.att[0], tot.n[0] * res.aggregate("simple").att[0], rtol=1e-12)

    def test_cluster_divergence_fails_closed(self, panel):
        """Bare-cluster fit whose kept cells have INCOMPLETE treated support:
        the cohort-mass weighting (full cohort per period) disagrees with the
        complete-case count, so the total is ambiguous and fails closed."""
        holed = panel.copy()
        treated_rows = holed.index[
            (holed["first_treat"] > 0) & (holed["time"] >= holed["first_treat"])
        ]
        holed.loc[treated_rows[:3], "y"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = CallawaySantAnna().fit(holed.dropna(), **FIT_KW)
            clustered = CallawaySantAnna(cluster="unit").fit(holed.dropna(), **FIT_KW)
        # The plain fit is admitted at the complete-case count...
        tot_plain = plain.aggregate("total")
        assert tot_plain.n[0] < _raw_treated_obs(panel)
        # ...while the cluster fit's masses branch diverges and refuses.
        with pytest.raises(NotImplementedError, match="incomplete treated support"):
            clustered.aggregate("total")

    def test_rcs_fails_closed(self):
        data = _rcs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = CallawaySantAnna(panel=False).fit(data, **FIT_KW)
        with pytest.raises(NotImplementedError, match="repeated-cross-section"):
            res.aggregate("total")

    def test_genuinely_unbalanced_rc_routing_fails_closed(self, panel):
        thinned = panel.drop(panel.index[::13]).reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(allow_unbalanced_panel=True).fit(thinned, **FIT_KW)
        with pytest.raises(NotImplementedError, match="repeated-cross-section"):
            res.aggregate("total")

    def test_balanced_panel_with_unbalanced_flag_is_admitted(self, panel):
        """allow_unbalanced_panel=True on a BALANCED panel stays panel-routed
        (routing keys on the data, not the constructor flag) and is admitted."""
        res = CallawaySantAnna(allow_unbalanced_panel=True).fit(panel, **FIT_KW)
        tot = res.aggregate("total")
        assert tot.n[0] == float(_raw_treated_obs(panel))

    def test_survey_fails_closed_and_gate_is_immutable(self, survey_fit):
        with pytest.raises(NotImplementedError, match="declaring a survey_design"):
            survey_fit.aggregate("total")
        # Mutating the PUBLIC field must not bypass the gate: it reads the
        # kit's fit-time is_survey_fit snapshot, never self.survey_metadata.
        survey_fit.survey_metadata = None
        with pytest.raises(NotImplementedError, match="declaring a survey_design"):
            survey_fit.aggregate("total")

    def test_survey_plus_rc_gets_the_rc_message(self):
        """A fit that is BOTH survey-declared and RC-routed deterministically
        gets the (more informative) RC message - the gate order is pinned."""
        from diff_diff import SurveyDesign

        data = _rcs()
        data = data.assign(psu=np.arange(len(data)) % 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CallawaySantAnna(panel=False).fit(
                data, survey_design=SurveyDesign(psu="psu"), **FIT_KW
            )
        with pytest.raises(NotImplementedError, match="repeated-cross-section"):
            res.aggregate("total")

    def test_legacy_kit_fails_closed(self, panel):
        """Pre-upgrade kits (no fit-time snapshots) refuse with the refit
        message - never a mutable-field or public-replay fallback."""
        for missing in ("agg_gt_cells", "is_survey_fit", "both"):
            res = CallawaySantAnna().fit(panel, **FIT_KW)
            bk = res._aggregation_kit.bookkeeping
            if missing in ("agg_gt_cells", "both"):
                del bk["agg_gt_cells"]
            if missing in ("is_survey_fit", "both"):
                del bk["is_survey_fit"]
            with pytest.raises(NotImplementedError, match="refit"):
                res.aggregate("total")

    def test_post_fit_mutation_immunity(self, panel):
        """The mass comes from the immutable kit snapshot; the scalars come
        from the same stored fields the simple relay reads - so a public-field
        edit cannot desynchronize total == n x simple."""
        res = CallawaySantAnna().fit(panel, **FIT_KW)
        n_before = res.aggregate("total").n[0]
        for k in list(res.group_time_effects):
            res.group_time_effects[k]["n_treated"] = 9999
            res.group_time_effects[k]["effect"] = np.nan
        tot = res.aggregate("total")
        assert tot.n[0] == n_before
        assert np.allclose(tot.att[0], tot.n[0] * res.aggregate("simple").att[0], rtol=1e-12)

    def test_anticipation_window_cells_count(self, panel):
        """anticipation=1 keeps the g-1 window cells in the post set (cells
        with g - anticipation <= t < g COUNT as post), so the mass grows by
        exactly the window cells' treated observations."""
        res0 = CallawaySantAnna().fit(panel, **FIT_KW)
        res1 = CallawaySantAnna(anticipation=1).fit(panel, **FIT_KW)
        n0 = res0.aggregate("total").n[0]
        n1 = res1.aggregate("total").n[0]
        window = sum(
            d["n_treated"]
            for (g, t), d in res1.group_time_effects.items()
            if g - 1 <= t < g and np.isfinite(d["effect"])
        )
        assert n1 == n0 + window and window > 0

    def test_universal_base_reference_cells_excluded(self, panel):
        """Universal-base fits carry reference cells; the anticipation filter
        alone excludes them from the mass (their period is < g by
        construction), so the mass matches the varying-base fit's."""
        res_u = CallawaySantAnna(base_period="universal").fit(panel, **FIT_KW)
        res_v = CallawaySantAnna().fit(panel, **FIT_KW)
        assert res_u.aggregate("total").n[0] == res_v.aggregate("total").n[0]

    def test_zero_c_arms_emit_all_nan_row_without_warning(self, panel):
        """Empty post set / all-NaN effects -> all-NaN row, NO UserWarning
        (the RELAY-level no-re-warn convention; the recompute levels'
        bootstrap replay re-emits by design), on the plain-panel branch AND
        the cohort-mass (bare-cluster) branch - the isfinite(C) guard must
        run BEFORE the coincidence comparison (NaN != 0.0 is True)."""
        for ctor in ({}, {"cluster": "unit"}):
            res = CallawaySantAnna(**ctor).fit(panel, **FIT_KW)
            cells = res._aggregation_kit.bookkeeping["agg_gt_cells"]
            res._aggregation_kit.bookkeeping["agg_gt_cells"] = tuple(
                (g, t, float("nan"), n) for (g, t, _, n) in cells
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tot = res.aggregate("total")
            assert [w for w in caught if issubclass(w.category, UserWarning)] == []
            for col in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper", "n"):
                assert np.isnan(getattr(tot, col)[0]), (ctor, col)
            assert np.isnan(tot.df[0])

    def test_true_overflow_blanks_whole_row(self, panel):
        """A FINITE overall value that becomes non-finite BY the x C scaling
        blanks the row entirely - the one path allowed to blank a finite att,
        pinned on both the att and se halves of the guard."""
        for field_name in ("overall_att", "overall_se"):
            res = CallawaySantAnna().fit(panel, **FIT_KW)
            setattr(res, field_name, 1e308)
            tot = res.aggregate("total")
            for col in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper", "n"):
                assert np.isnan(getattr(tot, col)[0]), (field_name, col)

    def test_bootstrap_total_relays(self, panel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boot = CallawaySantAnna(n_bootstrap=49, seed=42).fit(panel, **FIT_KW)
        tot = boot.aggregate("total")
        simple = boot.aggregate("simple")
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.allclose(tot.se[0], tot.n[0] * simple.se[0], rtol=1e-12)
        assert np.isnan(tot.df[0])

    def test_pickle_round_trip(self, fitted):
        import pickle

        clone = pickle.loads(pickle.dumps(fitted))
        a, b = fitted.aggregate("total"), clone.aggregate("total")
        assert a.att[0] == b.att[0] and a.n[0] == b.n[0]

    def test_inert_across_other_levels(self, fitted):
        before = fitted.aggregate("total").att[0]
        fitted.aggregate("group")
        fitted.aggregate("event_study")
        assert fitted.aggregate("total").att[0] == before

    def test_constant_tau_recovers_total(self):
        """Leg 2 (runs in the default suite; POINT-ESTIMATE estimand check -
        no confidence-interval assertion intended): on a constant-effect DGP
        the total approximates tau x the raw treated unit-period count, with
        BOTH oracle factors container-independent."""
        rng = np.random.default_rng(7)
        tau = 2.0
        rows = []
        for u in range(60):
            g = [0, 4, 5][u % 3]
            for t in range(1, 8):
                y = 1.0 + 0.3 * t + 0.2 * (u % 7) + (tau if g and t >= g else 0.0)
                rows.append({"unit": u, "time": t, "first_treat": g, "y": y + rng.normal(0, 0.05)})
        frame = pd.DataFrame(rows)
        res = CallawaySantAnna().fit(frame, **FIT_KW)
        tot = res.aggregate("total")
        expected = tau * _raw_treated_obs(frame)
        assert abs(tot.att[0] - expected) / expected < 0.05


class TestTotalNonAdopters:
    """'total' is vocabulary-wide; non-adopters fail closed with the suffix."""

    def test_stacked_rejects_total_with_vocabulary_suffix(self, stacked_fitted):
        # StackedDiD is the in-file AggregationMixin non-adopter exemplar:
        # 'total' is vocabulary-wide, so the mixin's fail-closed error names
        # it as known-but-unimplemented rather than unknown.
        with pytest.raises(ValueError, match="Unsupported aggregation type") as exc:
            stacked_fitted.aggregate("total")
        assert "vocabulary" in str(exc.value)

    def test_wooldridge_custom_aggregate_rejects_total_its_own_way(self):
        """WooldridgeDiDResults is NOT an AggregationMixin subclass - its
        custom aggregate(type=...) raises its own fail-closed ValueError
        naming what it supports (no vocabulary suffix expected)."""
        from diff_diff import WooldridgeDiD

        rng = np.random.default_rng(9)
        rows = []
        for u in range(40):
            g = [0, 3, 4][u % 3]
            for t in range(1, 7):
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "first_treat": g,
                        "y": 1.0 + 0.1 * t + (1.0 if g and t >= g else 0.0) + rng.normal(0, 0.2),
                    }
                )
        df = pd.DataFrame(rows)
        res = WooldridgeDiD().fit(
            df, outcome="y", unit="unit", time="time", first_treat="first_treat"
        )
        with pytest.raises(ValueError, match="type must be one of"):
            res.aggregate("total")


class TestTotalImputation:
    """ImputationDiD total: finite-support mass (sum-tau identity)."""

    def _fit(self, frame, **kw):
        from diff_diff import ImputationDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ImputationDiD().fit(frame, **IMPUTATION_KW, **kw)

    def test_identity_sum_tau_and_raw_frame_oracle(self):
        frame = _imputation_panel()
        res = self._fit(frame)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        # fully identified DGP
        assert tot.n[0] == float(_raw_treated_obs(frame, time_col="period"))
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.allclose(tot.se[0], tot.n[0] * simple.se[0], rtol=1e-12)
        # C x overall == sum(tau) exactly (overall is the finite-support mean)
        assert np.allclose(tot.att[0], res.treatment_effects["tau_hat"].sum(), rtol=1e-10)
        assert tot.df[0] == simple.df[0] or (np.isnan(tot.df[0]) and np.isnan(simple.df[0]))

    def test_survey_fails_closed_analytic_and_replicate(self):
        for replicate in (False, True):
            frame, rep_cols = _imputation_survey_panel(replicate=replicate)
            design = _imputation_survey_design(rep_cols if replicate else None)
            from diff_diff import ImputationDiD

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ImputationDiD().fit(frame, survey_design=design, **IMPUTATION_KW)
            with pytest.raises(NotImplementedError, match="declaring a survey_design"):
                res.aggregate("total")
            res.survey_metadata = None  # gate reads the kit snapshot
            with pytest.raises(NotImplementedError, match="declaring a survey_design"):
                res.aggregate("total")

    def test_live_frame_mutation_immunity(self):
        """The kit's 'df' is a live _fit_data reference; the mass must come
        from the fit-time total_support stash, not a frame recompute."""
        res = self._fit(_imputation_panel())
        n_before = res.aggregate("total").n[0]
        res._aggregation_kit.bookkeeping["df"]["_tau_hat"] = np.nan
        assert res.aggregate("total").n[0] == n_before

    def test_legacy_kit_fails_closed(self):
        res = self._fit(_imputation_panel())
        del res._aggregation_kit.bookkeeping["total_support"]
        with pytest.raises(NotImplementedError, match="refit"):
            res.aggregate("total")

    def test_anticipation_window_obs_enter_the_mass(self):
        """anticipation=1 widens the treated support to t >= g - 1, so the
        total mass grows by exactly the frame-derived window observations
        (ImputationDiD's mass comes from anticipation-adjusted masks)."""
        from diff_diff import ImputationDiD

        frame = _imputation_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n0 = ImputationDiD().fit(frame, **IMPUTATION_KW).aggregate("total").n[0]
            n1 = ImputationDiD(anticipation=1).fit(frame, **IMPUTATION_KW).aggregate("total").n[0]
        window = int(
            (
                (frame["first_treat"] > 0)
                & (frame["period"] >= frame["first_treat"] - 1)
                & (frame["period"] < frame["first_treat"])
            ).sum()
        )
        assert window > 0
        assert n1 == n0 + window

    def test_zero_support_emits_all_nan_row_without_warning(self):
        """C == 0 (the all-unidentified fit's support) maps to a NaN mass and
        an all-NaN row - never a finite n=0 beside NaN inference."""
        res = self._fit(_imputation_panel())
        res._aggregation_kit.bookkeeping["total_support"] = 0.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tot = res.aggregate("total")
        assert [w for w in caught if issubclass(w.category, UserWarning)] == []
        for col in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper", "n"):
            assert np.isnan(getattr(tot, col)[0]), col
        assert np.isnan(tot.df[0])

    def test_degenerate_inference_passes_through(self):
        """Constant outcome: overall att=0, se=0, CI=(nan, nan), t/p NaN with
        C > 0. The total row MIRRORS the simple relay - att/n stay finite,
        inherited NaNs pass through, nothing is blanket-blanked (the repo's
        non-estimable-row convention)."""
        frame = _imputation_panel()
        frame = frame.assign(outcome=5.0)
        res = self._fit(frame)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        # The constant-outcome fit really is degenerate: se == 0, CI == NaN.
        assert simple.se[0] == 0.0 and np.isnan(simple.conf_int_lower[0])
        assert np.isfinite(tot.n[0]) and tot.n[0] > 0
        assert tot.att[0] == tot.n[0] * simple.att[0]
        assert tot.se[0] == 0.0
        assert np.isnan(tot.conf_int_lower[0])

    def test_bootstrap_total_relays(self):
        from diff_diff import ImputationDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ImputationDiD(n_bootstrap=49, seed=3).fit(_imputation_panel(), **IMPUTATION_KW)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.allclose(tot.se[0], tot.n[0] * simple.se[0], rtol=1e-12)
        assert tot.t_stat[0] == simple.t_stat[0]
        assert tot.p_value[0] == simple.p_value[0]
        assert np.isnan(tot.df[0])

    def test_leg4_divergence_finite_support_not_raw(self):
        """On a partially unidentified fit the total uses the FINITE support -
        the fix for the documented scale='auto' overcount ('simple''s n stays
        raw |Omega_1| by contract)."""
        frame = _imputation_panel()
        # Drop the never-treated observations at the last period so some
        # treated cells lose identification (the tutorial divergence recipe).
        keep = ~((frame["first_treat"] == 0) & (frame["period"] == frame["period"].max()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._fit(frame[keep])
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        raw_n = simple.n[0]
        # The recipe MUST produce reduced finite support (166 < 250 on this
        # fixture) - an unconditional pin, so a fixture drift that stops
        # exercising the divergence fails loudly instead of passing vacuously.
        assert np.isfinite(tot.n[0]) and tot.n[0] < raw_n, (tot.n[0], raw_n)
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert not np.allclose(tot.att[0], raw_n * simple.att[0], rtol=1e-12)


class TestTotalTwoStage:
    """TwoStageDiD total: post-filter D-support."""

    def _fit(self, frame, **kw):
        from diff_diff import TwoStageDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TwoStageDiD().fit(frame, **TWOSTAGE_KW, **kw)

    def test_identity_and_raw_frame_oracle(self):
        frame = _twostage_panel()
        res = self._fit(frame)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        assert tot.n[0] == float(_raw_treated_obs(frame, time_col="period"))
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.allclose(tot.se[0], tot.n[0] * simple.se[0], rtol=1e-12)

    def test_survey_fails_closed_and_gate_is_immutable(self):
        from diff_diff import SurveyDesign, TwoStageDiD

        frame = _twostage_panel()
        frame = frame.assign(psu=frame["unit"] % 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD().fit(frame, survey_design=SurveyDesign(psu="psu"), **TWOSTAGE_KW)
        with pytest.raises(NotImplementedError, match="declaring a survey_design"):
            res.aggregate("total")
        res.survey_metadata = None
        with pytest.raises(NotImplementedError, match="declaring a survey_design"):
            res.aggregate("total")

    def test_anticipation_window_obs_enter_the_mass(self):
        """anticipation=1 widens the treated support to t >= g - 1, so the
        total mass grows by exactly the frame-derived window observations
        (TwoStageDiD's mass comes from anticipation-adjusted masks)."""
        from diff_diff import TwoStageDiD

        frame = _twostage_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n0 = TwoStageDiD().fit(frame, **TWOSTAGE_KW).aggregate("total").n[0]
            n1 = TwoStageDiD(anticipation=1).fit(frame, **TWOSTAGE_KW).aggregate("total").n[0]
        window = int(
            (
                (frame["first_treat"] > 0)
                & (frame["period"] >= frame["first_treat"] - 1)
                & (frame["period"] < frame["first_treat"])
            ).sum()
        )
        assert window > 0
        assert n1 == n0 + window

    def test_reduced_post_filter_support(self):
        """total.n is the POST-FILTER treatment-indicator support: non-finite
        treated `_y_tilde` rows leave D, so n drops below raw |Omega_1| (the
        'simple' row's documented pre-filter count)."""
        res = self._fit(_twostage_panel())
        bk = res._aggregation_kit.bookkeeping
        d_mask = np.asarray(bk["omega_1_mask"], dtype=bool)
        raw_n = float(d_mask.sum())
        # Degrade three treated rows' first-stage residuals in the kit's
        # PRIVATE frame copy (the state the fit-time masker would have seen).
        idx = bk["df"].index[d_mask][:3]
        bk["df"].loc[idx, "_y_tilde"] = np.nan
        tot = res.aggregate("total")
        assert tot.n[0] == raw_n - 3
        simple = res.aggregate("simple")
        assert simple.n[0] == raw_n  # 'simple' keeps the raw count by contract
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)

    def test_zero_support_emits_all_nan_row_without_warning(self):
        res = self._fit(_twostage_panel())
        bk = res._aggregation_kit.bookkeeping
        d_mask = np.asarray(bk["omega_1_mask"], dtype=bool)
        bk["df"].loc[bk["df"].index[d_mask], "_y_tilde"] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tot = res.aggregate("total")
        assert [w for w in caught if issubclass(w.category, UserWarning)] == []
        for col in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper", "n"):
            assert np.isnan(getattr(tot, col)[0]), col

    def test_bootstrap_total_relays(self):
        from diff_diff import TwoStageDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStageDiD(n_bootstrap=49, seed=3).fit(_twostage_panel(), **TWOSTAGE_KW)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.isnan(tot.df[0])


class TestTotalEfficientDiD:
    """EfficientDiD total: integer sum of kept cells' n_treated."""

    def _fit(self, frame, **kw):
        from diff_diff import EfficientDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return EfficientDiD().fit(frame, **EFFICIENT_KW, **kw)

    def test_identity_and_integer_mass(self):
        frame = _efficient_panel()
        res = self._fit(frame)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        # Integer by construction: sum of per-cell integer n_treated, never
        # the float n_units x sum(cohort_fractions) product.
        assert tot.n[0] == int(tot.n[0])
        n_cells = sum(
            c["n_treated"]
            for (g, t), c in res._aggregation_kit.bookkeeping["group_time_effects"].items()
            if t >= g and np.isfinite(c["effect"])
        )
        assert tot.n[0] == float(n_cells)
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.allclose(tot.se[0], tot.n[0] * simple.se[0], rtol=1e-12)

    def test_survey_fails_closed_all_design_shapes(self):
        """The gate is weight-type-agnostic: analytic pweight, replicate,
        UNWEIGHTED psu-only, and analytic FWEIGHT declared designs all fail
        closed (fweight resolved weights stay RAW - the gate must not depend
        on any weight-scale property)."""
        from diff_diff import EfficientDiD, SurveyDesign

        frame, _ = _efficient_survey_panel()
        frame_rep, rep_cols = _efficient_survey_panel(replicate=True)
        plain = _efficient_panel()
        arms = [
            (frame, _efficient_survey_design()),
            (frame_rep, _efficient_survey_design(rep_cols)),
            (plain.assign(psu=plain["unit"] % 10), SurveyDesign(psu="psu")),
            (
                plain.assign(w=10.0 + (plain["unit"] % 5)),
                SurveyDesign(weights="w", weight_type="fweight"),
            ),
        ]
        for arm_frame, arm_design in arms:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = EfficientDiD().fit(arm_frame, survey_design=arm_design, **EFFICIENT_KW)
            with pytest.raises(NotImplementedError, match="declaring a survey_design"):
                res.aggregate("total")
            res.survey_metadata = None
            with pytest.raises(NotImplementedError, match="declaring a survey_design"):
                res.aggregate("total")

    def test_bootstrap_total_relays(self):
        from diff_diff import EfficientDiD

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = EfficientDiD(n_bootstrap=49, seed=3).fit(_efficient_panel(), **EFFICIENT_KW)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        assert np.allclose(tot.att[0], tot.n[0] * simple.att[0], rtol=1e-12)
        assert np.allclose(tot.se[0], tot.n[0] * simple.se[0], rtol=1e-12)
        assert tot.t_stat[0] == simple.t_stat[0]
        assert tot.p_value[0] == simple.p_value[0]
        assert np.isnan(tot.df[0])

    def test_post_fit_mutation_immunity(self):
        """The mass reads the kit's deep-copied cells; a NaN edit of the
        public field (which WOULD perturb finite-keeper selection) is inert."""
        res = self._fit(_efficient_panel())
        n_before = res.aggregate("total").n[0]
        for k in list(res.group_time_effects):
            res.group_time_effects[k]["effect"] = np.nan
            res.group_time_effects[k]["n_treated"] = 777
        assert res.aggregate("total").n[0] == n_before

    def test_anticipation_window_cells_enter_the_mass(self):
        """anticipation=1 keeps the g-1 window cells as post, so C grows by
        exactly those cells' n_treated (the CS twin of this pin)."""
        from diff_diff import EfficientDiD

        frame = _efficient_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res0 = EfficientDiD().fit(frame, **EFFICIENT_KW)
            res1 = EfficientDiD(anticipation=1).fit(frame, **EFFICIENT_KW)
        n0 = res0.aggregate("total").n[0]
        n1 = res1.aggregate("total").n[0]
        window = sum(
            c["n_treated"]
            for (g, t), c in res1._aggregation_kit.bookkeeping["group_time_effects"].items()
            if g - 1 <= t < g and np.isfinite(float(c["effect"]))
        )
        assert n1 == n0 + window and window > 0

    def test_zero_keepers_emit_all_nan_row_without_warning(self):
        res = self._fit(_efficient_panel())
        gte = res._aggregation_kit.bookkeeping["group_time_effects"]
        for cell in gte.values():
            cell["effect"] = float("nan")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tot = res.aggregate("total")
        assert [w for w in caught if issubclass(w.category, UserWarning)] == []
        for col in ("att", "se", "t_stat", "p_value", "conf_int_lower", "conf_int_upper", "n"):
            assert np.isnan(getattr(tot, col)[0]), col

    def test_degenerate_inference_passes_through(self):
        frame = _efficient_panel().assign(outcome=5.0)
        res = self._fit(frame)
        tot = res.aggregate("total")
        simple = res.aggregate("simple")
        # The constant-outcome fit really is degenerate: se == 0.
        assert simple.se[0] == 0.0
        assert np.isfinite(tot.n[0]) and tot.n[0] > 0
        assert tot.att[0] == tot.n[0] * simple.att[0]
        assert tot.se[0] == 0.0

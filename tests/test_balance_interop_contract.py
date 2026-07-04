"""Contract tests for the surface consumed by ``balance.interop.diff_diff``.

Meta's `balance` package (>=0.21) ships a one-way adapter,
``balance.interop.diff_diff`` (facebookresearch/balance PR #465), whose
``pip install balance[did]`` extra pins ``diff-diff>=3.3.0,<4``. The adapter
depends on specific diff-diff behaviors beyond the flat result aliases
already pinned in ``tests/test_result_aliases.py``:

- ``diff_diff.aggregate_survey`` (top-level export): forwarded params and
  the ``(panel_df, second_stage_design)`` return pair
  (``to_panel_for_did`` wraps it verbatim);
- ``SurveyDesign`` field names (``balance.interop.conventions``
  hard-codes ``DEFAULT_DESIGN_COLUMNS`` and the adapter validates
  ``design_columns=`` overrides against the dataclass fields);
- estimator resolution by class name and short alias via
  ``getattr(diff_diff, name)`` (``fit_did``'s dispatch), with ``fit()``
  accepting ``survey_design=``;
- results accepting attribute attachment (``fit_did`` sets a
  ``_balance_adjustment`` provenance side-channel via ``setattr``);
- the pweight-only guard on staggered estimators (the adapter defaults
  ``weight_type="pweight"`` because of it);
- ``SurveyMetadata`` attribute names read by ``as_balance_diagnostic``
  (``design_effect`` / ``effective_n`` / ``sum_weights``) via defensive
  ``getattr(..., None)`` - a rename would silently NULL balance's
  diagnostics rather than raise.

These tests intentionally import NO balance code: they pin OUR public
surface so a diff-diff refactor cannot silently break ``balance[did]``.
"""

import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

import diff_diff
from diff_diff import CallawaySantAnna, SurveyDesign, aggregate_survey

# ---------------------------------------------------------------------------
# Shared tiny survey micro-frame + one fitted survey CS result
# ---------------------------------------------------------------------------

N_UNITS = 12
YEARS = [2019, 2020, 2021, 2022]


def _make_micro(seed=0):
    """Respondent-level microdata: 12 units x 4 years x 40 respondents."""
    rng = np.random.default_rng(seed)
    n_per = 40
    unit = np.repeat(np.arange(N_UNITS), len(YEARS) * n_per)
    year = np.tile(np.repeat(YEARS, n_per), N_UNITS)
    g = np.where(unit < 4, 2021, 0)[np.arange(len(unit))]
    y = (
        1.0
        + 0.1 * (year - YEARS[0])
        + 0.5 * unit / N_UNITS
        - 1.0 * ((g > 0) & (year >= g))
        + rng.normal(0, 0.5, len(unit))
    )
    return pd.DataFrame(
        {
            "unit": unit,
            "year": year,
            "g": g,
            "y": y,
            "w": rng.uniform(0.5, 2.0, len(unit)),
            "stratum": unit % 3,
            "psu": unit * 10 + rng.integers(0, 4, len(unit)),
        }
    )


@pytest.fixture(scope="module")
def micro():
    return _make_micro()


@pytest.fixture(scope="module")
def panel_and_design(micro):
    design = SurveyDesign(weights="w", strata="stratum", psu="psu")
    return aggregate_survey(micro, by=["unit", "year"], outcomes="y", survey_design=design)


@pytest.fixture(scope="module")
def fitted_cs(panel_and_design, micro):
    panel, second_stage = panel_and_design
    panel = panel.merge(micro[["unit", "g"]].drop_duplicates(), on="unit", how="left")
    cs = CallawaySantAnna(estimation_method="reg", base_period="universal")
    return cs.fit(
        panel,
        outcome="y_mean",
        unit="unit",
        time="year",
        first_treat="g",
        survey_design=second_stage,
    )


# ---------------------------------------------------------------------------
# 1-2. aggregate_survey: signature superset + return contract
# ---------------------------------------------------------------------------

# Params to_panel_for_did forwards verbatim (balance/interop/diff_diff.py,
# aggregate_survey call). The adapter routes lonely_psu into the
# first-stage SurveyDesign, NOT into aggregate_survey.
ADAPTER_FORWARDED_PARAMS = {
    "data",
    "by",
    "outcomes",
    "survey_design",
    "covariates",
    "min_n",
    "second_stage_weights",
}


def test_aggregate_survey_signature_superset():
    # Superset (not exact-set) pinning: an ADDITIVE optional param cannot
    # break balance's keyword-arg adapter, so it must not break this test.
    params = set(inspect.signature(aggregate_survey).parameters)
    missing = ADAPTER_FORWARDED_PARAMS - params
    assert not missing, (
        f"aggregate_survey lost parameter(s) {sorted(missing)} that "
        "balance.interop.diff_diff.to_panel_for_did forwards verbatim."
    )


def test_aggregate_survey_return_pair_and_panel_schema(panel_and_design):
    panel, second_stage = panel_and_design
    assert isinstance(panel, pd.DataFrame)
    assert isinstance(second_stage, SurveyDesign)
    # to_panel_for_did documents the {outcome}_mean/_se/_n/_precision cells
    # and wires the second-stage design's weights column into the panel.
    for col in ["y_mean", "y_se", "y_n", "y_precision"]:
        assert col in panel.columns, f"panel lost column {col}"
    assert second_stage.weights in panel.columns
    # The first `by` element becomes the second-stage clustering variable.
    assert second_stage.psu == "unit"


# ---------------------------------------------------------------------------
# 3. SurveyDesign field-name contract
# ---------------------------------------------------------------------------

# Normative list: balance/interop/conventions.py DEFAULT_DESIGN_COLUMNS plus
# the adapter's _ALLOWED_DESIGN_FIELDS validate against these exact names.
SURVEY_DESIGN_FIELDS = {
    "weights",
    "strata",
    "psu",
    "fpc",
    "weight_type",
    "nest",
    "lonely_psu",
    "replicate_weights",
    "replicate_method",
    "replicate_strata",
    "fay_rho",
    "combined_weights",
    "replicate_scale",
    "replicate_rscales",
    "mse",
}


def test_survey_design_field_names_exact():
    fields = {f.name for f in dataclasses.fields(SurveyDesign)}
    assert fields == SURVEY_DESIGN_FIELDS, (
        "SurveyDesign dataclass fields changed "
        f"(removed: {sorted(SURVEY_DESIGN_FIELDS - fields)}, "
        f"added: {sorted(fields - SURVEY_DESIGN_FIELDS)}). The exact pin is "
        "intentional friction: balance's adapter enumerates these names in "
        "conventions.py/_ALLOWED_DESIGN_FIELDS, so ANY change (adds included) "
        "should be consciously synced with the balance maintainers before "
        "updating this list. Removed/renamed fields break balance[did] "
        "outright."
    )


def test_survey_design_tsl_construction():
    # The TSL combo to_survey_design builds by default (auto-wired
    # stratum/psu/fpc convention columns + the adapter's lonely_psu default).
    design = SurveyDesign(
        weights="w",
        strata="stratum",
        psu="psu",
        fpc="fpc",
        nest=True,
        lonely_psu="adjust",
    )
    assert design.weight_type == "pweight"  # adapter's documented default


def test_survey_design_replicate_construction():
    # Replicate combo (mutually exclusive with strata/psu/fpc).
    design = SurveyDesign(
        weights="w",
        replicate_weights=["rep_1", "rep_2"],
        replicate_method="JK1",
    )
    assert design.replicate_method == "JK1"


# ---------------------------------------------------------------------------
# 4. Estimator resolution by name and alias
# ---------------------------------------------------------------------------

# The 17 estimator names promised in balance/interop/diff_diff.py's module
# docstring ("weight_type='pweight' is ... compatible with ...").
ADAPTER_DOCSTRING_ESTIMATORS = [
    "CallawaySantAnna",
    "StackedDiD",
    "ImputationDiD",
    "HeterogeneousAdoptionDiD",
    "TwoStageDiD",
    "WooldridgeDiD",
    "TROP",
    "StaggeredTripleDifference",
    "ChaisemartinDHaultfoeuille",
    "TripleDifference",
    "SyntheticDiD",
    "EfficientDiD",
    "DifferenceInDifferences",
    "TwoWayFixedEffects",
    "MultiPeriodDiD",
    "SunAbraham",
    "ContinuousDiD",
]

# Short aliases promised in fit_did's docstring; removing one breaks
# balance's documented examples (verified exports in diff_diff/__init__.py).
ADAPTER_DOCSTRING_ALIASES = {
    "CS": "CallawaySantAnna",
    "DiD": "DifferenceInDifferences",
    "BJS": "ImputationDiD",
    "HAD": "HeterogeneousAdoptionDiD",
}


@pytest.mark.parametrize("name", ADAPTER_DOCSTRING_ESTIMATORS)
def test_estimator_resolves_with_survey_design_fit(name):
    cls = getattr(diff_diff, name, None)
    assert cls is not None and inspect.isclass(cls), (
        f"diff_diff.{name} is no longer an exported class; "
        "balance.interop.diff_diff.fit_did resolves estimators via "
        "getattr(diff_diff, name)."
    )
    fit_params = set(inspect.signature(cls.fit).parameters)
    assert "survey_design" in fit_params, (
        f"{name}.fit() no longer accepts survey_design=; fit_did would "
        "warn and silently run the fit without the balance-built design."
    )


@pytest.mark.parametrize("alias,target", sorted(ADAPTER_DOCSTRING_ALIASES.items()))
def test_estimator_alias_resolves(alias, target):
    cls = getattr(diff_diff, alias, None)
    assert cls is not None, f"short alias diff_diff.{alias} was removed"
    assert cls.__name__ == target


# ---------------------------------------------------------------------------
# 5. Provenance side-channel: setattr on results must keep working
# ---------------------------------------------------------------------------


def test_result_accepts_provenance_attribute(fitted_cs):
    # fit_did(preserve_adjustment=True) attaches the balance Sample via
    # setattr(results, "_balance_adjustment", sample). A future __slots__
    # on result dataclasses would break this silently for balance users.
    sentinel = object()
    fitted_cs._balance_adjustment = sentinel
    assert fitted_cs._balance_adjustment is sentinel
    del fitted_cs._balance_adjustment
    # ...while the flat aliases stay read-only properties (full alias
    # coverage lives in tests/test_result_aliases.py - not duplicated here).
    with pytest.raises(AttributeError):
        fitted_cs.att = 0.0


# ---------------------------------------------------------------------------
# 6. pweight-only guard (why the adapter defaults weight_type="pweight")
# ---------------------------------------------------------------------------


def test_cs_rejects_fweight_design(micro):
    data = micro.copy()
    data["w_int"] = 2  # fweights must be non-negative integers
    design = SurveyDesign(weights="w_int", weight_type="fweight")
    cs = CallawaySantAnna(estimation_method="reg")
    with pytest.raises(ValueError, match="pweight"):
        cs.fit(
            data,
            outcome="y",
            unit="unit",
            time="year",
            first_treat="g",
            survey_design=design,
        )


# ---------------------------------------------------------------------------
# 7. SurveyMetadata attribute names read by as_balance_diagnostic
# ---------------------------------------------------------------------------


def test_survey_metadata_attribute_contract(fitted_cs):
    sm = fitted_cs.survey_metadata
    assert sm is not None, (
        "survey-fitted CallawaySantAnnaResults.survey_metadata is None; "
        "as_balance_diagnostic would silently report None diagnostics."
    )
    for attr in ("design_effect", "effective_n", "sum_weights"):
        value = getattr(sm, attr, None)
        assert value is not None and np.isfinite(value), (
            f"SurveyMetadata.{attr} missing or non-finite; balance reads it "
            "via getattr(sm, ..., None) and would silently emit None."
        )

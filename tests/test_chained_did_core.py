"""Restricted contribution boundaries and analytical plotting semantics."""

import importlib

import numpy as np
import pytest

from diff_diff import ChainedDiD
from diff_diff.chained_did import _pair_ipw
from tests.test_chained_did import rotating_panel


@pytest.mark.parametrize(
    "kwargs", [{"n_bootstrap": 0}, {"n_bootstrap": 19}, {"seed": None}, {"seed": 1}]
)
def test_core_constructor_rejects_optional_extension_parameters(kwargs):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ChainedDiD(**kwargs)


@pytest.mark.parametrize("unavailable", [False, True])
def test_structural_plot_reference_preserves_analytical_intervals(monkeypatch, unavailable):
    data = rotating_panel()
    result = ChainedDiD(cluster="id" if unavailable else None).fit(data, "y", "id", "t", "g")
    event = result.aggregate("event_study")
    if unavailable:
        event.conf_int_lower[:] = np.nan
        event.conf_int_upper[:] = np.nan
    module = importlib.import_module("diff_diff.visualization._event_study")
    monkeypatch.setattr(module, "_render_event_study_mpl", lambda df, **kw: df)
    inferred = module.plot_event_study(event, show=False)
    explicit = module.plot_event_study(event, show=False, reference_period=-1)
    np.testing.assert_array_equal(inferred.ci_lower, explicit.ci_lower)
    np.testing.assert_array_equal(inferred.ci_upper, explicit.ci_upper)
    if unavailable:
        assert explicit.ci_lower.isna().all() and explicit.ci_upper.isna().all()
    with pytest.raises(ValueError, match="fitted reference"):
        module.plot_event_study(event, show=False, reference_period=0)


@pytest.mark.parametrize("treated_x1", [0, 1])
def test_native_pair_distinguishes_separation_from_an_overlapping_counterpart(treated_x1):
    x = np.r_[np.zeros(83), np.ones(41 + treated_x1)][:, None]
    treated = np.r_[np.zeros(45), np.ones(38), np.zeros(41), np.ones(treated_x1)].astype(bool)
    changes = np.linspace(-1, 2, len(treated))
    if not treated_x1:
        with pytest.raises(ValueError, match="IPW propensity fit failed:.*did not converge"):
            _pair_ipw(changes, treated, x)
    else:
        effect, contributions, info = _pair_ipw(changes, treated, x)
        assert np.isfinite(effect) and np.isfinite(contributions).all()
        assert contributions.sum() == pytest.approx(0, abs=1e-10)
        assert 0 < info["propensity_min"] <= info["propensity_max"] < 1

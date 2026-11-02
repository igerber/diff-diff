"""Independent algebraic checks of chained DiD estimating equations."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root
from scipy.special import expit

from diff_diff import ChainedDiD
from diff_diff.bootstrap_chunking import iter_weight_blocks


def binary_covariate_panel():
    rows, changes, treated, xs = [], [], [], []
    for g, x, count in ((1, 0, 12), (1, 1, 24), (0, 0, 30), (0, 1, 18)):
        for noise in np.linspace(-1, 1, count):
            uid = len(changes)
            change = (5 + 2 * x if g else 1 + 3 * x) + noise
            rows.extend([(uid, 0, g, uid / 10, x), (uid, 1, g, uid / 10 + change, x)])
            changes.append(change)
            treated.append(bool(g))
            xs.append(x)
    return (
        pd.DataFrame(rows, columns=["id", "t", "g", "y", "x"]),
        np.array(changes),
        np.array(treated),
        np.array(xs),
    )


def test_binary_saturated_ipw_matches_stratum_formula_and_complete_influence():
    data, delta, treated, x = binary_covariate_panel()
    result = ChainedDiD().fit(data, "y", "id", "t", "g", covariates=["x"])
    nt = treated.sum()
    control_means = np.array([delta[(~treated) & (x == k)].mean() for k in (0, 1)])
    expected = delta[treated].mean() - control_means[x[treated]].mean()
    q = np.empty(len(delta))
    q[treated] = (delta[treated] - control_means[x[treated]] - expected) / nt
    for k in (0, 1):
        controls = (~treated) & (x == k)
        q[controls] = (
            -(treated & (x == k)).sum()
            / (nt * controls.sum())
            * (delta[controls] - control_means[k])
        )
    assert result.att == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(result.influence_functions[:, 1], q, atol=1e-10)
    assert result.se**2 == pytest.approx(q @ q, abs=1e-11)


def test_continuous_covariates_match_independent_numerical_score_jacobian():
    rng = np.random.default_rng(86)
    x = rng.normal(size=(180, 2))
    z = np.column_stack([np.ones(len(x)), x])
    treated = rng.random(len(x)) < expit(z @ np.array([0.2, 0.4, -0.3]))
    delta = 1 + x[:, 0] + x[:, 1] ** 2 + treated + rng.normal(size=len(x))
    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(len(x)), 2),
            "t": np.tile([0, 1], len(x)),
            "g": np.repeat(treated.astype(int), 2),
            "y": np.column_stack([np.zeros(len(x)), delta]).ravel(),
            "x1": np.repeat(x[:, 0], 2),
            "x2": np.repeat(x[:, 1], 2),
        }
    )
    fitted = root(lambda beta: z.T @ (treated - expit(z @ beta)), np.zeros(3))
    assert fitted.success
    p = expit(z @ fitted.x)
    w = (~treated) * p / (1 - p)
    mt, mc = delta[treated].mean(), w @ delta / w.sum()
    parameters = np.r_[fitted.x, mt, mc]

    def scores(a):
        ps = expit(z @ a[:3])
        return np.column_stack(
            [
                z * (treated - ps)[:, None],
                treated * (delta - a[3]),
                (~treated) * ps / (1 - ps) * (delta - a[4]),
            ]
        )

    step = 1e-5
    perturbations = np.eye(5) * step
    jacobian = np.column_stack(
        [
            (scores(parameters + d).sum(axis=0) - scores(parameters - d).sum(axis=0)) / (2 * step)
            for d in perturbations
        ]
    )
    independent = -np.linalg.solve(jacobian, scores(parameters).T).T
    q = independent[:, 3] - independent[:, 4]
    result = ChainedDiD().fit(data, "y", "id", "t", "g", covariates=["x1", "x2"])
    assert result.att == pytest.approx(mt - mc, abs=1e-8)
    np.testing.assert_allclose(result.influence_functions[:, 1], q, atol=1e-8)


def test_indirectly_supported_earlier_nuisance_is_not_a_direct_pre_diagnostic():
    from tests.test_chained_did import rotating_panel

    data = rotating_panel()
    data["g"] = data.g.map({0: 0, 2: 3})
    result = ChainedDiD().fit(data, "y", "id", "t", "g")
    assert result.group_time_effects[3, 3].effect == pytest.approx(4)
    assert result.group_time_effects[3, 0].skip_reason == "unsupported_direct_pre_comparison"
    assert np.isnan(result.group_time_effects[3, 0].effect)


def test_covariate_affine_scaling_does_not_change_fitted_comparison_or_inference():
    data, *_ = binary_covariate_panel()
    a = ChainedDiD().fit(data, "y", "id", "t", "g", covariates=["x"])
    data["x"] = 1e8 + 1e4 * data.x
    b = ChainedDiD().fit(data, "y", "id", "t", "g", covariates=["x"])
    assert a.att == pytest.approx(b.att, abs=1e-9)
    np.testing.assert_allclose(a.vcov, b.vcov, atol=1e-10)


@pytest.mark.parametrize("fault", ["separation", "constant", "missing", "changing"])
def test_failed_or_invalid_covariate_models_are_not_replaced_with_unconditional(fault):
    data, *_ = binary_covariate_panel()
    if fault == "separation":
        data["x"] = data.g
    elif fault == "constant":
        data["x"] = 1
    elif fault == "missing":
        data.loc[0, "x"] = np.nan
    else:
        data.loc[0, "x"] = 4
    with pytest.raises(ValueError):
        ChainedDiD().fit(data, "y", "id", "t", "g", covariates=["x"])


def test_covariance_retains_shared_controls_across_cohorts():
    rows = []
    for g in (0, 1, 2):
        for j, noise in enumerate((-1.5, -0.5, 0.5, 1.5)):
            for t in (0, 1, 2):
                rows.append((f"{g}-{j}", t, g, noise * t + (2 * g if g and t >= g else 0)))
    result = ChainedDiD().fit(
        pd.DataFrame(rows, columns=["id", "t", "g", "y"]), "y", "id", "t", "g"
    )
    j, k = result.vcov_index.index((1, 1)), result.vcov_index.index((2, 2))
    assert result.vcov[j, k] == pytest.approx(5 / 16)


def test_registry_only_units_change_aggregation_masses_and_share_influence():
    rows = []
    for g in (0, 1, 2):
        for j, noise in enumerate((-1.5, -0.5, 0.5, 1.5)):
            for t in (0, 1, 2):
                effect = (2 * t if g == 1 else 10) if g and t >= g else 0
                rows.append((f"{g}-{j}", t, g, noise * t + effect))
    for j in range(4):
        rows.append((f"registry-{j}", 0, 2, np.nan if j < 2 else 0))
    data = pd.DataFrame(rows, columns=["id", "t", "g", "y"])
    result = ChainedDiD().fit(data, "y", "id", "t", "g")
    assert result.n_units == 16
    assert result.att == pytest.approx(6.5)
    event = result.aggregate("event_study")
    assert event.att[event.event_time == 0][0] == pytest.approx(22 / 3)
    assert np.all(result.influence_functions[-4:] == 0)
    _, contribution, _ = result._combine(result._post_keys())
    assert np.all(np.abs(contribution[-4:]) > 0)


def test_joint_cluster_multiplier_reproduces_the_declared_covariance():
    data, *_ = binary_covariate_panel()
    data["cluster"] = data.id // 4
    result = ChainedDiD(cluster="cluster").fit(data, "y", "id", "t", "g", covariates=["x"])
    q = result.influence_functions
    labels = data.drop_duplicates("id").cluster.to_numpy()
    clusters = np.unique(labels)
    sums = np.array([q[labels == c].sum(axis=0) for c in clusters])
    expected = len(clusters) / (len(clusters) - 1) * sums.T @ sums
    np.testing.assert_allclose(result.vcov, expected)
    draws = np.concatenate(
        [
            block
            for _, block in iter_weight_blocks(
                30000, len(clusters), "rademacher", np.random.default_rng(4)
            )
        ]
    )
    perturbations = np.sqrt(len(clusters) / (len(clusters) - 1)) * draws @ sums
    np.testing.assert_allclose(np.cov(perturbations, rowvar=False), expected, rtol=0.03, atol=1e-12)


def test_two_date_unconditional_matches_hand_calculation_and_existing_estimator():
    from diff_diff import CallawaySantAnna

    data, delta, treated, _ = binary_covariate_panel()
    result = ChainedDiD().fit(data, "y", "id", "t", "g")
    expected = delta[treated].mean() - delta[~treated].mean()
    variance = delta[treated].var() / treated.sum() + delta[~treated].var() / (~treated).sum()
    existing = CallawaySantAnna(n_bootstrap=0, estimation_method="ipw").fit(
        data, "y", "id", "t", "g"
    )
    assert result.att == pytest.approx(expected)
    assert result.se**2 == pytest.approx(variance)
    assert result.att == pytest.approx(existing.att)
    assert result.se == pytest.approx(existing.se)


def test_overidentified_catalogue_matches_independent_graph_laplacian_solution():
    rows = []
    pair_means = {(0, 1): 2.0, (0, 2): 6.0, (0, 3): 8.0, (1, 2): 3.0, (1, 3): 7.0, (2, 3): 5.0}
    for (s, t), mean in pair_means.items():
        for g in (0, 2):
            for j, noise in enumerate((-1.0, 0.0, 1.0)):
                uid = f"{s}-{t}-{g}-{j}"
                rows.extend([(uid, s, g, 0.0), (uid, t, g, noise + (mean if g else 0))])
    data = pd.DataFrame(rows, columns=["id", "t", "g", "y"])
    # Solve the full-rank reference-anchored graph Laplacian independently of SVD.
    incidence = np.array([[-1, 0, 0], [-1, 1, 0], [-1, 0, 1], [0, 1, 0], [0, 0, 1], [0, -1, 1]])
    loading = np.linalg.solve(incidence.T @ incidence, incidence.T)
    expected = loading @ np.array(list(pair_means.values()))
    result = ChainedDiD().fit(data, "y", "id", "t", "g")
    assert result.group_time_effects[2, 2].effect == pytest.approx(expected[1])
    assert result.group_time_effects[2, 3].effect == pytest.approx(expected[2])
    assert len(result.diagnostics["catalogue"]) == 6
    q = np.zeros((data.id.nunique(), 6))
    units = data.id.drop_duplicates().tolist()
    for k, (s, t) in enumerate(pair_means):
        for g in (0, 2):
            for j, noise in enumerate((-1.0, 0.0, 1.0)):
                q[units.index(f"{s}-{t}-{g}-{j}"), k] = (1 if g else -1) * noise / 3
    expected_q = q @ loading.T
    columns = [result.vcov_index.index((2, t)) for t in (2, 3)]
    np.testing.assert_allclose(
        result.vcov[np.ix_(columns, columns)], (expected_q.T @ expected_q)[1:, 1:], atol=1e-12
    )

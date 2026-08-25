"""Duck-typed nuisance-learner protocol + native learners (private DML infra).

Learner contract
----------------
Learners ALWAYS receive a raw covariate matrix ``X`` with NO intercept column;
every learner manages the intercept internally (sklearn convention). Learners
are INSTANCES with sklearn fit-reset semantics: ``fit`` fully re-initializes
the fitted state and returns ``self``. A stateful/warm-start user learner that
violates fit-reset cannot be detected without taking a clone dependency —
documented accepted limitation.

Native learners (``"linear"``, ``"ridge"``, ``"logit"``, ``"sieve"``) wrap
the ``diff_diff.linalg`` solvers and expose sklearn-style fitted state
(``coef_``, ``intercept_``). Rank deficiency: the UNPENALIZED fixed-design
learners (``LinearLearner``, ``LogitLearner``) FAIL CLOSED — an unpenalized
rank-deficient fit has no well-defined out-of-sample prediction; only
``RidgeLearner`` proceeds, predicting from the identified
(non-NaN-coefficient) columns after zero-variance drops (the penalty keeps
the fit well-posed and the intercept is preserved structurally).
``SieveLearner`` selects its polynomial degree ADAPTIVELY (information
criterion over admissible degrees, skipping ill-conditioned ones) and falls
back to an intercept-only fit — with a loud ``UserWarning`` — only when
every candidate degree is inadmissible on a non-empty support; genuinely
degenerate inputs (empty data, fewer than 2 positive-weight rows) fail
closed with a targeted ``ValueError``.

Duck-typed user learners (e.g. sklearn estimators) plug in via the structural
Protocols below; ``validate_learner`` checks the required methods up front
(``TypeError`` per Python convention for wrong-typed objects), and
``_validate_predictions`` checks a learner's OUTPUT (``ValueError``, the
``conley._validate_callable_metric_result`` precedent).
"""

import warnings
from typing import Any, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
from scipy.special import expit

from diff_diff.linalg import solve_logit, solve_ols, solve_ridge

__all__ = [
    "LearnerName",
    "RegressorLearner",
    "ClassifierLearner",
    "validate_learner",
    "make_learner",
    "LinearLearner",
    "RidgeLearner",
    "LogitLearner",
    "SieveLearner",
]

LearnerName = Literal["linear", "ridge", "logit", "sieve"]

_REGRESSOR_NAMES = ("linear", "ridge", "sieve")
_CLASSIFIER_NAMES = ("logit",)


class RegressorLearner(Protocol):
    """Structural contract for outcome-regression learners."""

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "RegressorLearner": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class ClassifierLearner(Protocol):
    """Structural contract for propensity (binary classification) learners."""

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "ClassifierLearner": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def validate_learner(obj: Any, *, kind: str, param_name: str) -> None:
    """Validate a duck-typed learner object up front.

    One targeted ``TypeError`` per missing/uncallable required method.
    """
    if kind not in ("regressor", "classifier"):
        raise ValueError(f"kind must be 'regressor' or 'classifier', got {kind!r}")
    required = ("fit", "predict") if kind == "regressor" else ("fit", "predict_proba")
    for method in required:
        attr = getattr(obj, method, None)
        if attr is None:
            raise TypeError(
                f"{param_name}: learner object {type(obj).__name__!r} is missing "
                f"the required method {method!r} (a {kind} learner needs "
                f"{' and '.join(required)})"
            )
        if not callable(attr):
            raise TypeError(
                f"{param_name}: learner attribute {method!r} on "
                f"{type(obj).__name__!r} is not callable"
            )


def _validate_predictions(
    arr: object,
    n: int,
    *,
    kind: str,
    context: str,
    classes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Validate a learner's prediction output; returns a float64 P(y=1)/(n,) array.

    Regressors must return shape (n,) exactly (a multi-output (n, 2) result is
    rejected — silently taking one column would feed the wrong target as a
    nuisance). Classifiers may return (n,) — interpreted as P(y=1) — or the
    sklearn ``predict_proba`` (n, 2) layout. Column selection for (n, 2) is
    CLASS-AWARE when the fitted learner exposes ``classes_`` (pass it via
    ``classes``): the classes must be exactly {0, 1} and the column aligned
    with class 1 is taken, so a learner with ``classes_ == [1, 0]`` cannot
    silently feed P(y=0) as the propensity. Without ``classes_`` the sklearn
    convention (column 1 = P(y=1); classes sorted ascending) is assumed and
    documented. Finite; classifiers must yield probabilities in [0, 1]. No
    clipping.
    """
    out = np.asarray(arr, dtype=np.float64)
    if kind == "classifier" and out.ndim == 2 and out.shape[1] == 2:
        col = 1
        if classes is not None:
            cls = np.asarray(classes, dtype=np.float64)
            if cls.shape != (2,) or set(cls.tolist()) != {0.0, 1.0}:
                raise ValueError(
                    f"{context}: classifier classes_ must be exactly {{0, 1}}, " f"got {classes!r}"
                )
            col = int(np.flatnonzero(cls == 1.0)[0])
        out = out[:, col]
    if out.ndim != 1:
        allowed = f"({n},) or ({n}, 2)" if kind == "classifier" else f"({n},)"
        raise ValueError(
            f"{context}: {kind} learner output must have shape {allowed}, "
            f"got shape {np.asarray(arr).shape}"
        )
    if out.shape[0] != n:
        raise ValueError(f"{context}: learner output has length {out.shape[0]}, expected {n}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{context}: learner output contains non-finite values")
    if kind == "classifier" and (np.any(out < 0.0) or np.any(out > 1.0)):
        raise ValueError(f"{context}: classifier output must be probabilities in [0, 1]")
    return out


def make_learner(spec: Union[str, object], *, kind: str, **options: Any) -> Any:
    """Resolve a learner spec: string enum -> native class; object -> validate.

    Valid names per kind: regressor -> {"linear", "ridge", "sieve"};
    classifier -> {"logit"}. A wrong pairing or unknown string raises
    ``ValueError`` naming the valid names FOR THAT KIND.
    """
    if kind not in ("regressor", "classifier"):
        raise ValueError(f"kind must be 'regressor' or 'classifier', got {kind!r}")
    if isinstance(spec, str):
        valid = _REGRESSOR_NAMES if kind == "regressor" else _CLASSIFIER_NAMES
        if spec not in valid:
            raise ValueError(
                f"Unknown {kind} learner {spec!r}; valid names for kind="
                f"{kind!r} are {sorted(valid)}"
            )
        if spec == "linear":
            return LinearLearner(**options)
        if spec == "ridge":
            return RidgeLearner(**options)
        if spec == "sieve":
            # Explicit branch is mandatory: once "sieve" passes the name
            # check, the final fall-through below would silently hand back a
            # LogitLearner (a classifier) for a regressor request.
            return SieveLearner(**options)
        return LogitLearner(**options)
    validate_learner(spec, kind=kind, param_name="learner")
    return spec


def _predict_identified(X_aug: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Linear prediction using only identified (non-NaN) coefficients."""
    identified = np.isfinite(coefs)
    return X_aug[:, identified] @ coefs[identified]


class LinearLearner:
    """OLS regressor (``solve_ols``); intercept prepended internally."""

    # Config attrs audited by DMLDiD's native-trust predicate (exact
    # primitive types required for the verbatim repr/error path).
    _CONFIG_ATTRS: Tuple[str, ...] = ()

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._coefs: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        # Configuration-aware repr: DMLDiDResults stores the learner spec AS
        # its repr, so the default address-based object.__repr__ would make
        # results serialization non-deterministic.
        return "LinearLearner()"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LinearLearner":
        X = np.asarray(X, dtype=np.float64)
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        coefs, _, _ = solve_ols(
            X_aug,
            np.asarray(y, dtype=np.float64),
            return_vcov=False,
            weights=sample_weight,
        )
        if not np.all(np.isfinite(coefs)):
            # Fail closed: an unpenalized rank-deficient fit has no
            # well-defined out-of-sample prediction — the retained collinear
            # column can stand in for the intercept in-sample and produce
            # arbitrary variation on held-out rows whose values differ.
            # cross_fit_predict converts this to DegenerateFoldError; use
            # RidgeLearner when rank-deficient designs are expected.
            raise ValueError(
                "LinearLearner: the training design is rank-deficient "
                f"({int(np.sum(~np.isfinite(coefs)))} dropped direction(s)); "
                "an unpenalized fit has no well-defined out-of-sample "
                "prediction. Use RidgeLearner (penalized, well-posed) or drop "
                "the collinear columns."
            )
        self._coefs = coefs
        self.intercept_ = float(coefs[0])
        self.coef_ = coefs[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coefs is None:
            raise ValueError("LinearLearner is not fitted; call fit first")
        X = np.asarray(X, dtype=np.float64)
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        return _predict_identified(X_aug, self._coefs)


class RidgeLearner:
    """Ridge regressor (``solve_ridge``); intercept handled by the solver.

    ``alpha`` is stored verbatim (sklearn-clone identity contract); the
    default ``"loocv"`` selects the penalty by closed-form leave-one-out CV.
    """

    _CONFIG_ATTRS: Tuple[str, ...] = ("alpha",)

    def __init__(self, alpha: Union[float, str] = "loocv") -> None:
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._coefs: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        # Configuration-aware repr (estimate-moving config surfaces in
        # results provenance; no address-based object.__repr__).
        return f"RidgeLearner(alpha={self.alpha!r})"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "RidgeLearner":
        coefs = solve_ridge(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            alpha=self.alpha,
            weights=sample_weight,
        )
        self._coefs = coefs
        self.intercept_ = float(coefs[0]) if np.isfinite(coefs[0]) else float("nan")
        self.coef_ = coefs[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coefs is None:
            raise ValueError("RidgeLearner is not fitted; call fit first")
        X = np.asarray(X, dtype=np.float64)
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        return _predict_identified(X_aug, self._coefs)


class LogitLearner:
    """IRLS logistic classifier (``solve_logit``); intercept added by the solver."""

    _CONFIG_ATTRS: Tuple[str, ...] = ("max_iter", "tol")

    def __init__(self, max_iter: int = 25, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._coefs: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        # Configuration-aware repr (estimate-moving config surfaces in
        # results provenance; no address-based object.__repr__).
        return f"LogitLearner(max_iter={self.max_iter!r}, tol={self.tol!r})"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LogitLearner":
        beta, _ = solve_logit(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            max_iter=self.max_iter,
            tol=self.tol,
            weights=sample_weight,
            # Learner-appropriate EPV remedy: the shared solver default
            # recommends estimation_method='reg', an API the learner's
            # consumers (DMLDiD cross-fitting) do not expose.
            epv_remedy=(
                "Consider fewer covariates, a penalized custom classifier "
                "learner, or CallawaySantAnna(estimation_method='reg')."
            ),
        )
        if not np.all(np.isfinite(beta)):
            # Fail closed, same rationale as LinearLearner: an unpenalized
            # rank-deficient fit has no well-defined out-of-sample prediction.
            raise ValueError(
                "LogitLearner: the training design is rank-deficient "
                f"({int(np.sum(~np.isfinite(beta)))} dropped direction(s)); "
                "an unpenalized fit has no well-defined out-of-sample "
                "prediction. Drop the collinear columns (or use a penalized "
                "custom classifier)."
            )
        self._coefs = beta  # intercept first (solve_logit prepends it)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._coefs is None:
            raise ValueError("LogitLearner is not fitted; call fit first")
        X = np.asarray(X, dtype=np.float64)
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
        eta = _predict_identified(X_aug, self._coefs)
        p1 = expit(eta)  # overflow-stable sigmoid
        return np.column_stack([1.0 - p1, p1])


class SieveLearner:
    """Polynomial-sieve regressor with information-criterion degree selection.

    For each candidate total degree ``K = 1..k_max`` (auto ``k_max =
    floor(n_pos**0.2)``, floored at 1, where ``n_pos`` is the positive-weight
    support count; degrees whose basis dimension reaches ``n_pos`` are
    inadmissible), builds the monomial basis via the EfficientDiD sieve
    helper, fits by (W)LS, and scores ``IC(K) = n_pos * log(RSS_w / n_pos) +
    c_n * p_K`` with ``c_n = 2`` (AIC) or ``log(n_pos)`` (BIC). Degrees with
    an ill-conditioned (weighted) design Gram are skipped (one consolidated
    ``UserWarning``). If EVERY degree is inadmissible on a non-empty support,
    falls back to an intercept-only (weighted-mean) fit with a
    ``UserWarning``; genuinely degenerate inputs (empty data, fewer than 2
    positive-weight rows) raise a targeted ``ValueError`` instead (the
    fail-closed sibling contract).

    The standardization statistics ``center_``/``scale_`` are UNWEIGHTED even
    under ``sample_weight`` — the helper standardizes for numerical
    conditioning only, and fitted values are standardization-invariant.
    Fitted state is plain floats/ndarrays (deep-copyable, picklable);
    ``fit`` fully re-initializes it (sklearn fit-reset semantics).
    """

    _CONFIG_ATTRS: Tuple[str, ...] = ("k_max", "criterion")

    def __init__(self, k_max: Optional[int] = None, criterion: str = "bic") -> None:
        if criterion not in ("aic", "bic"):
            raise ValueError(f"SieveLearner: criterion must be 'aic' or 'bic', got {criterion!r}")
        if k_max is not None and (
            isinstance(k_max, bool) or not isinstance(k_max, (int, np.integer)) or k_max < 1
        ):
            raise ValueError(f"SieveLearner: k_max must be None or an integer >= 1, got {k_max!r}")
        self.k_max = int(k_max) if k_max is not None else None
        self.criterion = criterion
        self.degree_: Optional[int] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._coefs: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        # Configuration-aware repr: DMLDiDResults stores the learner spec AS
        # its repr, so the default address-based object.__repr__ would make
        # results serialization non-deterministic and hide the config.
        return f"SieveLearner(k_max={self.k_max!r}, criterion={self.criterion!r})"

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "SieveLearner":
        # Function-level import: keeps _learners.py free of a module-level
        # infra -> estimator-module edge.
        from math import comb

        from diff_diff.efficient_did_covariates import _polynomial_sieve_basis

        # Full fit-reset FIRST (sklearn fit-reset semantics): a refit that
        # fails validation below must not leave the PREVIOUS fitted model
        # in place — predict() after a failed fit raises "not fitted".
        self.degree_ = None
        self.coef_ = None
        self.intercept_ = None
        self.center_ = None
        self.scale_ = None
        self._coefs = None

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"SieveLearner: X must be 2-dimensional, got shape {X.shape}")
        n, d = X.shape
        if y.shape != (n,):
            raise ValueError(
                f"SieveLearner: y must have shape ({n},) matching X, got {np.asarray(y).shape}"
            )
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError(
                    f"SieveLearner: sample_weight must have shape ({n},), got {w.shape}"
                )
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("SieveLearner: sample_weight must be finite and non-negative")
        else:
            w = None
        # Fail closed on degenerate supports: the intercept-only fallback is
        # for the every-degree-inadmissible case on a NON-degenerate support;
        # an empty or singleton support has no usable fit at all.
        support = w > 0 if w is not None else np.ones(n, dtype=bool)
        n_pos = int(np.sum(support))
        if n == 0 or n_pos < 2:
            raise ValueError(
                f"SieveLearner: needs at least 2 positive-weight rows to fit, "
                f"got {n_pos} (n={n})"
            )
        # Zero-weight rows are FULLY inert: the fit operates exclusively on
        # the positive-weight support (basis, Gram, solve, RSS, IC — an
        # extreme zero-weight covariate would otherwise overflow the basis
        # powers before its zero weight could annihilate it), and the
        # finiteness contract applies to the support only.
        if w is not None and not support.all():
            X_fit = X[support]
            y_fit = y[support]
            w_fit: Optional[np.ndarray] = w[support]
        else:
            X_fit, y_fit, w_fit = X, y, w
        if not np.all(np.isfinite(X_fit)) or not np.all(np.isfinite(y_fit)):
            raise ValueError("SieveLearner: X and y must be finite on the positive-weight support")

        y_pos = y_fit
        if w_fit is not None:
            fallback_mean = float(np.average(y_fit, weights=w_fit))
        else:
            fallback_mean = float(np.mean(y_fit))

        k_max = self.k_max if self.k_max is not None else max(int(n_pos**0.2), 1)
        c_n = 2.0 if self.criterion == "aic" else float(np.log(max(n_pos, 2)))
        cond_threshold = 1.0 / np.sqrt(np.finfo(float).eps)
        # RSS floor so a near-perfect fit cannot drive log -> -inf and
        # spuriously select a high degree. The floor is built from the
        # WEIGHTED centered total sum of squares so it scales with the
        # weighted objective: under w -> c*w both RSS and the floor scale
        # by c and the selected degree is invariant to the weight scale
        # (an unweighted-variance floor binds under tiny weight scales and
        # silently changes model selection).
        if w_fit is not None:
            y_mean_w = float(np.average(y_fit, weights=w_fit))
            tss_w = float(np.sum(w_fit * (y_fit - y_mean_w) ** 2))
        else:
            tss_w = float(np.sum((y_pos - np.mean(y_pos)) ** 2))
        rss_floor = max(1e-300, 1e-12 * tss_w)

        best_ic = np.inf
        best: Optional[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = None
        singular_K: List[int] = []

        for K in range(1, k_max + 1):
            n_basis = comb(K + d, d)
            if n_basis >= n_pos:  # admissibility cap
                break
            basis, center, scale = _polynomial_sieve_basis(X_fit, K, return_stats=True)

            if w_fit is not None:
                gram = basis.T @ (w_fit[:, None] * basis)
            else:
                gram = basis.T @ basis
            with np.errstate(invalid="ignore", over="ignore"):
                gram_cond = float(np.linalg.cond(gram))
            if not np.isfinite(gram_cond) or gram_cond > cond_threshold:
                singular_K.append(K)
                continue

            coefs, _, _ = solve_ols(
                basis,
                y_fit,
                weights=w_fit,
                return_vcov=False,
                rank_deficient_action="warn",
            )
            if not np.all(np.isfinite(coefs)):
                singular_K.append(K)
                continue

            resid = y_fit - basis @ coefs
            if w_fit is not None:
                rss = float(np.sum(w_fit * resid**2))
            else:
                rss = float(np.sum(resid**2))
            rss = max(rss, rss_floor)
            ic_val = n_pos * float(np.log(rss / n_pos)) + c_n * n_basis
            if ic_val < best_ic:
                best_ic = ic_val
                best = (K, coefs, center, scale)

        if best is None:
            warnings.warn(
                "SieveLearner: every candidate degree was inadmissible "
                f"(support n_pos={n_pos}, skipped K={singular_K or 'none admissible'}). "
                "Falling back to an intercept-only (weighted-mean) fit.",
                UserWarning,
                stacklevel=2,
            )
            self.degree_ = 0
            self._coefs = np.array([fallback_mean])
            self.intercept_ = fallback_mean
            self.coef_ = np.array([])
            self.center_ = None
            self.scale_ = None
            return self

        if singular_K:
            warnings.warn(
                f"SieveLearner: skipped K={singular_K} due to rank-deficient or "
                "non-finite design; selected among the remaining degrees.",
                UserWarning,
                stacklevel=2,
            )
        K, coefs, center, scale = best
        self.degree_ = int(K)
        self._coefs = coefs
        self.intercept_ = float(coefs[0])  # basis carries its own intercept column
        self.coef_ = coefs[1:]
        self.center_ = center
        self.scale_ = scale
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.degree_ is None or self._coefs is None:
            raise ValueError("SieveLearner is not fitted; call fit first")
        X = np.asarray(X, dtype=np.float64)
        if self.degree_ == 0:
            # Intercept-only fallback: must not call the basis with None stats.
            return np.full(X.shape[0], float(self._coefs[0]))
        from diff_diff.efficient_did_covariates import _polynomial_sieve_basis

        basis = _polynomial_sieve_basis(X, self.degree_, center=self.center_, scale=self.scale_)
        return basis @ self._coefs

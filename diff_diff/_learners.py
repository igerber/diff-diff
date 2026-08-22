"""Duck-typed nuisance-learner protocol + native learners (private DML infra).

Learner contract
----------------
Learners ALWAYS receive a raw covariate matrix ``X`` with NO intercept column;
every learner manages the intercept internally (sklearn convention). Learners
are INSTANCES with sklearn fit-reset semantics: ``fit`` fully re-initializes
the fitted state and returns ``self``. A stateful/warm-start user learner that
violates fit-reset cannot be detected without taking a clone dependency —
documented accepted limitation.

Native learners (``"linear"``, ``"ridge"``, ``"logit"``) wrap the
``diff_diff.linalg`` solvers and expose sklearn-style fitted state
(``coef_``, ``intercept_``). Rank deficiency: the UNPENALIZED learners
(``LinearLearner``, ``LogitLearner``) FAIL CLOSED — an unpenalized
rank-deficient fit has no well-defined out-of-sample prediction; only
``RidgeLearner`` proceeds, predicting from the identified
(non-NaN-coefficient) columns after zero-variance drops (the penalty keeps
the fit well-posed and the intercept is preserved structurally).

Duck-typed user learners (e.g. sklearn estimators) plug in via the structural
Protocols below; ``validate_learner`` checks the required methods up front
(``TypeError`` per Python convention for wrong-typed objects), and
``_validate_predictions`` checks a learner's OUTPUT (``ValueError``, the
``conley._validate_callable_metric_result`` precedent).
"""

from typing import Any, Literal, Optional, Protocol, Union

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
]

LearnerName = Literal["linear", "ridge", "logit"]

_REGRESSOR_NAMES = ("linear", "ridge")
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

    Valid names per kind: regressor -> {"linear", "ridge"}; classifier ->
    {"logit"}. A wrong pairing or unknown string raises ``ValueError`` naming
    the valid names FOR THAT KIND.
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
        return LogitLearner(**options)
    validate_learner(spec, kind=kind, param_name="learner")
    return spec


def _predict_identified(X_aug: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Linear prediction using only identified (non-NaN) coefficients."""
    identified = np.isfinite(coefs)
    return X_aug[:, identified] @ coefs[identified]


class LinearLearner:
    """OLS regressor (``solve_ols``); intercept prepended internally."""

    def __init__(self) -> None:
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._coefs: Optional[np.ndarray] = None

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

    def __init__(self, alpha: Union[float, str] = "loocv") -> None:
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._coefs: Optional[np.ndarray] = None

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

    def __init__(self, max_iter: int = 25, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self._coefs: Optional[np.ndarray] = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> "LogitLearner":
        beta, _ = solve_logit(
            np.asarray(X, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            max_iter=self.max_iter,
            tol=self.tol,
            weights=sample_weight,
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

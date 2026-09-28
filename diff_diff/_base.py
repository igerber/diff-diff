"""Shared estimator parameter surface (4.0 program, Phase 2 PR 2(c)-i).

This module is a deliberate LEAF: it imports only the standard library, never
other ``diff_diff`` modules, so every estimator module can import it without
cycles (the ``results_base.py`` precedent).

One public symbol:

``BaseEstimator``
    Mixin providing the sklearn-compatible ``get_params`` / ``set_params``
    pair for every estimator class, replacing the 25 hand-rolled copies
    (see ``docs/v4-design.md`` section 7, "Constructor hygiene").

Design contract (the locked transactional rule):

- Parameter names are introspected from the concrete class's ``__init__``
  signature, so the parameter surface can never drift from the constructor.
- ``get_params`` returns ``{name: getattr(self, alias_or_name)}`` for every
  constructor parameter. ``deep`` is accepted for sklearn compatibility
  (``sklearn.base.clone`` calls ``get_params(deep=False)``) and ignored:
  no diff-diff estimator nests another estimator.
- ``set_params`` is TRANSACTIONAL via probe re-init: it validates the merged
  configuration by constructing a throwaway ``type(self)(**merged)`` - which
  raises before ``self`` is touched - then adopts the probe's parameter
  attributes plus each class's declared derived-config attributes. Fitted
  state (attributes ``fit()`` sets) is never touched, because only declared
  attributes are copied. The behavioral contract, stated post-normalization:

        est.set_params(**p)  ==  config state of
        type(est)(**{**est.get_params(), **cls._normalize_set_params(p)})

  Validation is therefore exactly ``__init__``'s validation, eagerly, and
  can never drift from it.

Per-class accommodation hooks (declarative; default no-op):

``_PARAM_ATTR_ALIASES``
    Maps a constructor parameter name to the attribute that stores its RAW
    value when the two differ (e.g. ``DifferenceInDifferences`` stores the
    raw ``vcov_type`` argument under ``_vcov_type_arg`` and the resolved
    value under ``vcov_type``).
``_DERIVED_CONFIG_ATTRS``
    Extra attributes ``__init__`` computes FROM the parameters that must be
    re-adopted from the probe after a successful ``set_params`` (e.g.
    ``_vcov_type_explicit``).
``_normalize_set_params``
    Classmethod rewriting the user-supplied ``params`` dict before the
    merge (e.g. the ``robust``-alone alias re-derivation on
    ``DifferenceInDifferences``).
"""

import inspect
from typing import Any, ClassVar, Dict, Mapping, Tuple, TypeVar

__all__ = ["BaseEstimator"]

TSelf = TypeVar("TSelf", bound="BaseEstimator")

# Signature kinds accepted as constructor parameters. VAR_POSITIONAL /
# VAR_KEYWORD are rejected loudly: an estimator constructor forwarding
# through *args/**kwargs has no introspectable parameter surface.
_ACCEPTED_KINDS = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
)


class BaseEstimator:
    """Mixin providing sklearn-compatible ``get_params`` / ``set_params``.

    See the module docstring for the full contract. Subclasses only declare
    the accommodation hooks when their constructor stores state under
    different names than its parameters; most estimators are plain drop-ins.
    """

    _PARAM_ATTR_ALIASES: ClassVar[Mapping[str, str]] = {}
    _DERIVED_CONFIG_ATTRS: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def _param_names(cls) -> Tuple[str, ...]:
        """Constructor parameter names, introspected once per class.

        Cached via ``cls.__dict__`` (never a plain class attribute, which a
        subclass with its own ``__init__`` would inherit; never an instance
        attribute, which would break ``vars(est)`` stability across fit).
        """
        cached = cls.__dict__.get("_param_names_cache")
        if cached is not None:
            return cached
        names = []
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if name == "self":
                continue
            if param.kind not in _ACCEPTED_KINDS:
                raise TypeError(
                    f"{cls.__name__}.__init__ uses *args/**kwargs "
                    f"(parameter {name!r}); its parameter surface cannot be "
                    "introspected for get_params/set_params."
                )
            names.append(name)
        result: Tuple[str, ...] = tuple(names)
        setattr(cls, "_param_names_cache", result)
        return result

    @classmethod
    def _normalize_set_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite the user-supplied params before the merge (default no-op)."""
        return params

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get estimator parameters (sklearn-compatible).

        Parameters
        ----------
        deep : bool, default True
            Accepted for sklearn compatibility (``sklearn.base.clone``
            calls ``get_params(deep=False)``) and ignored: no diff-diff
            estimator nests another estimator.

        Returns
        -------
        Dict[str, Any]
            Estimator parameters suitable for passing to ``__init__``.
            Keys follow the ``__init__`` signature order. Where a class
            stores a parameter's raw value under a different attribute
            (``_PARAM_ATTR_ALIASES``), the raw value is returned so that
            ``type(est)(**est.get_params())`` reconstructs the same
            configuration.
        """
        del deep
        aliases = self._PARAM_ATTR_ALIASES
        return {name: getattr(self, aliases.get(name, name)) for name in self._param_names()}

    def set_params(self: TSelf, **params: Any) -> TSelf:
        """
        Set estimator parameters (sklearn-compatible, transactional).

        Validates the merged configuration by constructing a throwaway
        instance of ``type(self)`` - so a rejected call raises the same
        error ``__init__`` would and leaves this estimator unchanged - then
        adopts the probe's parameter attributes and the class's declared
        derived-config attributes. Fitted state is never touched.

        Parameters
        ----------
        **params
            Estimator parameters. Unknown names raise ``ValueError``
            before any validation or mutation.

        Returns
        -------
        self
        """
        names = self._param_names()
        for key in params:
            if key not in names:
                raise ValueError(f"Unknown parameter: {key}")
        cls = type(self)
        normalized = cls._normalize_set_params(dict(params))
        merged = {**self.get_params(), **normalized}
        probe = cls(**merged)
        aliases = self._PARAM_ATTR_ALIASES
        for name in names:
            attr = aliases.get(name, name)
            setattr(self, attr, getattr(probe, attr))
        for attr in self._DERIVED_CONFIG_ATTRS:
            setattr(self, attr, getattr(probe, attr))
        return self

"""Shared deprecation-shim machinery for the 3.9 rename sweep (4.0 program, Phase 2(c)-ii).

This module is a deliberate LEAF: it imports only the standard library, never
other ``diff_diff`` modules, so every estimator module can import it without
cycles (the ``_base.py`` precedent). It is private and unexported. The 4.0
removal PRs delete the 3.9 shims that use it, but the module itself SURVIVES
4.0: the 4.0 -> 5.0 ``FutureWarning`` property shims of the results-storage
flip (ledger rows M-016 and M-050..M-058) ride ``deprecated_field_property``
through the 5.0 window (recorded in M-050's ``code_refs``).

The helpers implement the ledger's shim contract (``docs/v4-deprecations.yaml``,
groups ``renames-*``; rules in ``docs/v4-design.md`` section 8):

- Renamed parameter: the NEW name occupies the old name's signature position,
  the OLD name moves to the signature tail with the ``NOT_SUPPLIED`` sentinel
  default. ``resolve_renamed_kwarg`` maps old to new with a single
  ``FutureWarning``, rejects both-supplied calls loudly, and returns the
  caller-declared default when neither is passed.
- Dropped parameter (no successor name): the parameter keeps its position
  with a sentinel default; the owner calls ``warn_deprecated_kwarg`` when a
  value was actually supplied. (The 3.9 fit-time ``aggregate=`` shims were
  this shape; all of them were deleted at the 4.0 cut - CallawaySantAnna,
  EfficientDiD, StackedDiD, ChaisemartinDHaultfoeuille and
  HeterogeneousAdoptionDiD in PR 3b, ImputationDiD / TwoStageDiD /
  ContinuousDiD in PR 3c. ``warn_deprecated_kwarg`` has no library caller
  any more and is retained per section 8 rule 11 for the 5.0 window.)
- Renamed results field: the dataclass field takes the new name and
  ``deprecated_field_property`` builds the read-only warning alias under the
  old name (a plain class attribute, so it is a descriptor but NOT a
  ``__dataclass_fields__`` entry - the shape ``tests/test_v4_matrix.py``
  asserts for ``shimmed`` field rows).

Stacklevel contract: ``resolve_renamed_kwarg`` / ``warn_deprecated_kwarg``
must be called DIRECTLY from the public method that owns the parameter
(user frame -> public method -> helper -> ``warnings.warn`` = stacklevel 3).
``deprecated_field_property``'s getter warns at stacklevel 2 (user frame ->
property getter).

All warnings are ``FutureWarning`` (visible to end users by default) per the
section 2 category rule: new shims warn with ``FutureWarning`` (the
pre-existing M-001..M-003 ``DeprecationWarning`` sites were removed at the
4.0 cut).
"""

import warnings
from typing import Any, Dict, Mapping

__all__: "list[str]" = []


class _NotSupplied:
    """Sentinel for renamed/deprecated parameters (M-020 precedent).

    A plain ``None`` default cannot distinguish "not passed" from "passed
    None", so a bare ``None`` default would fire the FutureWarning on EVERY
    call (and break the warnings-as-errors ``cls(**est.get_params())``
    round-trip in ``tests/test_base_estimator.py``). The warning must fire
    only when the caller actually supplies the argument.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<not supplied>"

    # Parameter cloning must preserve this stateless marker's identity.
    def __copy__(self) -> "_NotSupplied":
        return self

    def __deepcopy__(self, memo: Any) -> "_NotSupplied":
        return self


NOT_SUPPLIED = _NotSupplied()


def deprecated_kwarg_message(qualname: str, name: str, instruction: str) -> str:
    """The single message template every shim warning and test pins."""
    return f"{qualname}({name}=) is deprecated and will be removed in 4.0; {instruction}."


def warn_deprecated_kwarg(
    qualname: str, name: str, instruction: str, *, stacklevel: int = 3
) -> None:
    """Emit the deprecation warning for a supplied deprecated parameter.

    The caller performs the was-it-supplied check (sentinel or ``None``
    depending on the parameter's legal value space) so that default and
    round-trip constructions stay silent.
    """
    warnings.warn(
        deprecated_kwarg_message(qualname, name, instruction),
        FutureWarning,
        stacklevel=stacklevel,
    )


def resolve_renamed_kwarg(
    qualname: str,
    old_name: str,
    old_value: Any,
    new_name: str,
    new_value: Any,
    *,
    default: Any = None,
    extra: str = "",
) -> Any:
    """Resolve a renamed keyword pair during the 3.9 shim window.

    Parameters
    ----------
    qualname : str
        The public surface the parameters belong to (``"Class.method"`` or
        a module-level function name) - used verbatim in messages.
    old_name, old_value : str, Any
        The deprecated parameter and what the caller passed (``NOT_SUPPLIED``
        when absent).
    new_name, new_value : str, Any
        The canonical parameter and what the caller passed (``NOT_SUPPLIED``
        when absent).
    default : Any
        Returned when neither name was supplied. Required parameters pass
        ``NOT_SUPPLIED`` here and re-validate with :func:`require_arg`.
    extra : str
        Appended to the warning message (e.g. a note that the old name
        survives with a different meaning on the same surface, as M-082's
        ``time=`` did through 3.x).

    Returns
    -------
    Any
        The effective value for the canonical parameter.
    """
    old_passed = not isinstance(old_value, _NotSupplied)
    new_passed = not isinstance(new_value, _NotSupplied)
    if old_passed and new_passed:
        raise ValueError(
            f"{qualname}() got both {old_name}= (deprecated) and {new_name}=; "
            f"pass only {new_name}=."
        )
    if old_passed:
        message = deprecated_kwarg_message(qualname, old_name, f"use {new_name}= instead")
        if extra:
            message = f"{message} {extra}"
        warnings.warn(message, FutureWarning, stacklevel=3)
        return old_value
    if new_passed:
        return new_value
    return default


def require_arg(qualname: str, name: str, value: Any) -> None:
    """Re-validate required-ness after the rename mapping.

    Sentinel-defaulted required parameters lose Python's built-in
    missing-argument ``TypeError``; this restores it, naming the NEW
    parameter.
    """
    if isinstance(value, _NotSupplied):
        raise TypeError(f"{qualname}() missing required argument: '{name}'")


def deprecated_field_property(
    cls_name: str, old: str, new: str, *, removed_in: str = "4.0"
) -> property:
    """Build the read-only warning alias for a renamed results field.

    Assigned WITHOUT an annotation in the dataclass body, so it stays a
    plain descriptor (never a ``__dataclass_fields__`` entry). Wording
    follows the 3.x ``SyntheticDiDResults.placebo_effects`` renamed-field
    alias precedent (M-003, removed at 4.0), with the FutureWarning category
    the ledger declares for renamed-field rows. ``removed_in`` names the
    release the ledger row schedules the removal for (``"4.0"`` for the 3.9
    rename sweep; ``"5.0"`` for aliases deprecated AT 4.0, e.g. the LWDiD
    ``pvalue``/``ci`` rows M-151/M-152) so the warning and the generated
    docstring never contradict the row.
    """

    def _get(self: Any) -> Any:
        warnings.warn(
            f"{cls_name}.{old} is deprecated; use {new} instead. "
            f"Will be removed in {removed_in}.",
            FutureWarning,
            stacklevel=2,
        )
        return getattr(self, new)

    _get.__name__ = old
    _get.__doc__ = f"Deprecated alias for :attr:`{new}` (removed in {removed_in})."
    return property(_get)


#: The 3.x ``overall_*`` storage names of the nine results containers flipped
#: at the 4.0 cut (ledger rows M-050..M-058), keyed to the canonical quintet
#: field that now stores each value. Shared by the containers' ``__setstate__``
#: pickle migration (spec section 5); ``ContinuousDiDResults`` keeps a
#: module-local map for its deviant ``overall_att_*`` spellings.
LEGACY_OVERALL_QUINTET: Dict[str, str] = {
    "overall_att": "att",
    "overall_se": "se",
    "overall_t_stat": "t_stat",
    "overall_p_value": "p_value",
    "overall_conf_int": "conf_int",
}


def migrate_renamed_state_keys(
    state: Mapping[str, Any], renames: Mapping[str, str]
) -> Dict[str, Any]:
    """Rewrite renamed storage keys of a pickled ``__dict__`` on load.

    Returns a NEW dict (the input is never mutated). For each ``old -> new``
    pair the legacy key is popped when present and its value seeds the
    canonical key only when that key is absent - an existing canonical
    value wins, and the legacy key never lingers in ``__dict__`` (a
    ``deprecated_field_property`` data descriptor would shadow it on read,
    but it would survive a re-pickle). Used by the ``__setstate__`` of the
    M-050..M-058 containers; the ``SyntheticDiDResults.__setstate__``
    precedent (``results.py``) is the hand-written form of the same rule.
    """
    out = dict(state)
    for old, new in renames.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    return out

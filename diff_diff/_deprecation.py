"""Private lifecycle helpers of the 4.0 program: pickle-key migration and required-arg sentinels.

This module is a deliberate LEAF: it imports only the standard library, never
other ``diff_diff`` modules, so every estimator module can import it without
cycles (the ``_base.py`` precedent). It is private and unexported.

History. Through the 3.9 rename sweep (Phase 2(c)-ii) this module carried the
``FutureWarning`` shim machinery of ``docs/v4-design.md`` section 8 - the
renamed-kwarg resolver, the dropped-kwarg warner and the renamed-results-field
warning property. The 4.0 cut deleted every shim that used them, and item 9b
(2026-11-01) deleted the helpers themselves together with their last consumers
(the ``overall_*`` / ``period_effects`` / ``pvalue`` / ``ci`` property shims of
ledger rows M-016, M-050..M-058, M-151, M-152, which had been scheduled for a
later removal and were instead retired AT 4.0 - no release ever carried them).
What remains:

- ``NOT_SUPPLIED`` / ``require_arg``: the sentinel default + required-ness
  re-validation that ``TwoWayFixedEffects.fit`` and ``TripleDifference.fit``
  use to express "``post`` is required in one mode, ``time`` in the other"
  (rows M-082/M-083, M-031/M-085).
- ``LEGACY_OVERALL_QUINTET`` / ``migrate_renamed_state_keys``: the
  ``__setstate__`` rewrite that loads 3.x pickles written under the
  ``overall_*`` storage names into the canonical quintet fields (spec
  section 5; rows M-050..M-058).
- ``migrate_event_study_reference_sentinel``: the ``__setstate__`` rewrite of
  the 3.x zero-count reference-row sentinel (row M-093).
"""

import math
from typing import Any, Dict, Mapping, Optional

__all__: "list[str]" = []


class _NotSupplied:
    """Sentinel default for a parameter whose required-ness depends on the call mode.

    A plain ``None`` default cannot distinguish "not passed" from "passed
    None"; the sentinel lets ``require_arg`` restore Python's missing-argument
    ``TypeError`` for the mode in which the parameter is required (the
    ``TwoWayFixedEffects.fit`` / ``TripleDifference.fit`` ``post`` vs ``time``
    contract) while the ``cls(**est.get_params())`` round-trip in
    ``tests/test_base_estimator.py`` stays silent.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<not supplied>"

    # Parameter cloning must preserve this stateless marker's identity.
    def __copy__(self) -> "_NotSupplied":
        return self

    def __deepcopy__(self, memo: Any) -> "_NotSupplied":
        return self


NOT_SUPPLIED = _NotSupplied()


def require_arg(qualname: str, name: str, value: Any) -> None:
    """Re-validate required-ness of a sentinel-defaulted parameter.

    Sentinel-defaulted required parameters lose Python's built-in
    missing-argument ``TypeError``; this restores it.
    """
    if isinstance(value, _NotSupplied):
        raise TypeError(f"{qualname}() missing required argument: '{name}'")


#: The 3.x ``overall_*`` storage names of the nine results containers flipped
#: at the 4.0 cut (ledger rows M-050..M-058; the names themselves were removed
#: at 4.0, item 9b), keyed to the canonical quintet
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
    value wins, and the legacy key never lingers in ``__dict__`` (it would
    otherwise survive a re-pickle). Used by the ``__setstate__`` of the
    M-050..M-058 containers; the ``SyntheticDiDResults.__setstate__``
    precedent (``results.py``) is the hand-written form of the same rule.
    """
    out = dict(state)
    for old, new in renames.items():
        if old in out:
            value = out.pop(old)
            out.setdefault(new, value)
    return out


def migrate_event_study_reference_sentinel(
    effects: Optional[Mapping[Any, Any]],
) -> Optional[Dict[Any, Any]]:
    """Rewrite 3.x zero-count reference rows of a pickled event-study dict.

    Through 3.x the native ``event_study_effects`` dicts marked their
    normalization (reference) row with a zero-count SENTINEL (``n_groups == 0``
    on CallawaySantAnna, ``n_obs == 0`` on StackedDiD / ImputationDiD /
    TwoStageDiD / SpilloverDiD). Since 4.0 (ledger row M-093, item 8a of the
    cut) the reference row carries an explicit ``"is_reference": True`` key and
    NO count key, and every remaining zero count is a genuine count. This
    helper is the ONLY place the retired sentinel is still recognized: the
    containers' ``__setstate__`` run it so 3.x pickles load in the 4.0 shape
    (the ``migrate_renamed_state_keys`` precedent).

    Returns a NEW dict (the input is never mutated); ``None`` passes through.
    A row is rewritten only when it is a mapping WITHOUT an ``is_reference``
    key whose count key (``n_groups`` or ``n_obs``) equals 0 and whose effect
    (``effect``, else ``att``) is finite and exactly ``0.0`` - the
    normalization anchor's signature. A NaN-effect zero-count row is a
    genuinely non-estimable horizon (TwoStageDiD all-filtered horizons,
    SpilloverDiD rectangular cells) and is copied verbatim.
    """
    if effects is None:
        return None
    out: Dict[Any, Any] = {}
    for key, row in effects.items():
        if not isinstance(row, Mapping) or "is_reference" in row:
            out[key] = row
            continue
        count_key = "n_groups" if "n_groups" in row else ("n_obs" if "n_obs" in row else None)
        if count_key is None:
            out[key] = row
            continue
        try:
            is_zero_count = float(row[count_key]) == 0.0
        except (TypeError, ValueError):
            is_zero_count = False
        effect = row.get("effect", row.get("att"))
        try:
            effect_f = float(effect)
        except (TypeError, ValueError):
            effect_f = float("nan")
        if is_zero_count and math.isfinite(effect_f) and effect_f == 0.0:
            new_row = dict(row)
            new_row.pop(count_key)
            new_row["is_reference"] = True
            out[key] = new_row
        else:
            out[key] = row
    return out

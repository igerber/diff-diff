### Behavioral Changes
- **Cross-fit learner isolation now fails closed**: DMLDiD replaces the prior
  warning-and-reuse fallback. A one-copy preflight raises `TypeError` before
  any group-time cell is fitted when a custom learner template's `deepcopy`
  fails or returns the original object; a per-fold clone failure later in the
  fit propagates as the same sanitized `TypeError` (never a NaN-cell skip).
  Implement `__deepcopy__` to return an independent instance; custom
  implementations remain responsible for nested mutable state. A failed
  re-fit now also clears any previous fitted result.

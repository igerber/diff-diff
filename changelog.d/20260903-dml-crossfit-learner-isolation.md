### Behavioral Changes
- **Cross-fit learner isolation now fails closed**: DMLDiD replaces the prior
  warning-and-reuse fallback. It raises `TypeError` before fitting any
  group-time cell when a custom learner template's `deepcopy` fails or returns
  the original object. Implement `__deepcopy__` to return an independent
  instance; custom implementations remain responsible for nested mutable state.
  A failed re-fit now also clears any previous fitted result.

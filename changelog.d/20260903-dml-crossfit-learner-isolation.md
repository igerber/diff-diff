### Fixed
- **Cross-fit learner isolation now fails closed**: DMLDiD rejects custom
  learner templates whose `deepcopy` fails or returns the original object
  before fitting any group-time cell, preventing reuse of the supplied
  template across folds. Custom `__deepcopy__` implementations remain
  responsible for isolating nested mutable state.

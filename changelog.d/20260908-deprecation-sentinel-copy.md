### Fixed
- **Preserve omitted-parameter sentinels when copying**: shallow and deep
  copies of the shared deprecation sentinel retain its identity, keeping
  StackedDiD parameter dictionaries equal after scikit-learn cloning.
  Deprecated-parameter warnings and alias resolution are unchanged.

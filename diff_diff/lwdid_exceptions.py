"""Backward-compatible exception aliases (deprecated).

All LWDiD exceptions now raise ValueError directly.
These aliases are kept only for isinstance() checks in user code.
"""

# Kept as thin aliases for any user code that catches them
LWDIDError = ValueError
LWDIDInferenceError = ValueError
BootstrapConvergenceError = ValueError
RandomizationError = ValueError
DiagnosticError = ValueError
InsufficientPrePeriodsError = ValueError
VisualizationError = ImportError

# Warning classes still needed for warnings.warn() categorization
LWDIDWarning = UserWarning
NumericalWarning = UserWarning
RandomizationWarning = UserWarning
DiagnosticWarning = UserWarning
SensitivityWarning = UserWarning
VisualizationWarning = UserWarning

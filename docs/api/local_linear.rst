Local-Linear Infrastructure
===========================

Kernels, kernel-moment constants, univariate local-linear regression at a
boundary, the MSE-optimal bandwidth selector, and the bias-corrected
local-linear fit.

This module ships the nonparametric building blocks that
:class:`~diff_diff.HeterogeneousAdoptionDiD` composes for its continuous-dose
fit paths (``continuous_at_zero`` and ``continuous_near_d_lower``); see
:doc:`had` for the estimator that consumes them.

Two scope tiers are exposed:

- **Generic helpers** (``local_linear_fit``, the three kernels,
  ``kernel_moments``, ``KERNELS``) - usable on their own for any one-sided
  boundary local-linear regression problem with a strictly nonnegative
  running variable.
- **HAD-scoped public wrappers** (``mse_optimal_bandwidth``,
  ``bias_corrected_local_linear``) - the configuration is hard-coded to
  HAD Phase 1b/1c (``p=1``, ``deriv=0``, ``interior=False``, ``vce="nn"``)
  and only those settings are parity-tested against R ``nprobust``. Other
  settings (``hc0``/``hc1``/``hc2``/``hc3`` variance, interior evaluation,
  higher polynomial orders) require dropping to the private
  ``diff_diff._nprobust_port`` module and accepting that parity has not
  been verified outside the HAD configuration. See ``docs/methodology/REGISTRY.md``
  ``HeterogeneousAdoptionDiD`` section for the full Phase 1b/1c contract.

The selector and bias correction are ports of the Calonico-Cattaneo-Farrell
(2018) plug-in bandwidth and Calonico-Cattaneo-Titiunik (2014) robust-bias
correction from the R ``nprobust`` package; methodology context lives in
:doc:`had` and ``docs/methodology/REGISTRY.md``.

Kernels
-------

Bounded one-sided kernels on ``[0, 1]`` for boundary-point nonparametric
regression. The library normalizes the Epanechnikov and triangular kernels
to ``int_0^1 k(u) du = 1/2``; the uniform kernel uses
``int_0^1 k(u) du = 1``.

.. autofunction:: diff_diff.epanechnikov_kernel

.. autofunction:: diff_diff.triangular_kernel

.. autofunction:: diff_diff.uniform_kernel

.. py:data:: diff_diff.KERNELS
   :type: dict[str, Callable[[np.ndarray], np.ndarray]]

   Mapping from kernel name (``"epanechnikov"`` / ``"triangular"`` / ``"uniform"``) to its callable evaluator on ``[0, 1]``. Pass the name string (not the callable) to ``local_linear_fit`` and ``mse_optimal_bandwidth`` via their ``kernel=`` argument.

.. autofunction:: diff_diff.kernel_moments

Boundary local-linear fit
-------------------------

Kernel-weighted OLS estimator of the conditional mean
``m(d0) := E[Y | D = d0]`` at the boundary of ``D``'s support.

.. autofunction:: diff_diff.local_linear_fit

.. autoclass:: diff_diff.LocalLinearFit
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

MSE-optimal bandwidth selector
------------------------------

Plug-in MSE-optimal bandwidth (Calonico-Cattaneo-Farrell 2018) for the
boundary local-linear fit. Returns ``BandwidthResult`` carrying the final
bandwidth ``h_mse`` plus the per-stage diagnostics needed to audit the
selector against the R ``nprobust`` reference.

.. autofunction:: diff_diff.mse_optimal_bandwidth

.. autoclass:: diff_diff.BandwidthResult
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Bias-corrected local-linear fit
-------------------------------

Calonico-Cattaneo-Titiunik (2014) robust-bias-corrected boundary
estimator. Composes the bandwidth selector and the local-linear fit and
returns a ``BiasCorrectedFit`` with point estimate, robust standard error,
and 95% confidence interval.

.. autofunction:: diff_diff.bias_corrected_local_linear

.. autoclass:: diff_diff.BiasCorrectedFit
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

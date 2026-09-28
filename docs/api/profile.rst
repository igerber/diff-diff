Panel Profiling
===============

Pre-fit description of a DiD panel's structural facts. ``profile_panel()``
inspects a long-format panel and returns a :class:`PanelProfile` dataclass
covering balance, treatment-type classification, outcome characteristics, and
a list of factual :class:`Alert` observations.

The profile is descriptive, not opinionated: alerts report what is (e.g.
"smallest cohort has 7 units"), never what to do about it. Estimator
selection is the caller's responsibility. For autonomous-agent consumption,
pair the profile output with the
`autonomous-agent reference guide <../llms-autonomous.txt>`_ (also accessible
at runtime via ``diff_diff.get_llm_guide("autonomous")``), which walks
through the estimator-support matrix and the per-design-feature reasoning
keyed off ``PanelProfile`` field values.

.. note::

   ``PanelProfile`` and its three supporting dataclasses
   (:class:`OutcomeShape`, :class:`TreatmentDoseShape`, :class:`Alert`) are
   re-exported at the top level of ``diff_diff`` so callers can construct
   or pattern-match against them without dotted-module access.

profile_panel
-------------

.. autofunction:: diff_diff.profile_panel

PanelProfile
------------

.. autoclass:: diff_diff.PanelProfile
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

OutcomeShape
------------

.. autoclass:: diff_diff.OutcomeShape
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

TreatmentDoseShape
------------------

.. autoclass:: diff_diff.TreatmentDoseShape
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Alert
-----

.. autoclass:: diff_diff.Alert
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

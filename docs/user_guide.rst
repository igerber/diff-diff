.. meta::
   :description: The diff-diff user guide — scholarly references, validation against R and Python packages, benchmarks, the 4.0 migration guide, and the full methodology registry.
   :keywords: DiD methodology, R did package comparison, DiD benchmarks, econometrics references, diff-diff 4.0 migration

User Guide
==========

The methods behind the library: scholarly references, validation against R
and Python implementations, performance benchmarks, the migration guide for
the upcoming 4.0 release, and the methodology registry documenting every
estimator's equations and edge cases.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: References
      :link: references
      :link-type: doc

      Scholarly citations for every estimator and diagnostic in the
      library.

   .. grid-item-card:: R Comparison
      :link: r_comparison
      :link-type: doc

      Coming from R? Side-by-side workflows and numerical validation
      against ``did``, ``synthdid``, and ``fixest``.

   .. grid-item-card:: Python Comparison
      :link: python_comparison
      :link-type: doc

      How diff-diff compares to other Python causal-inference libraries.

   .. grid-item-card:: Migrating to 4.0
      :link: migration-4.0
      :link-type: doc

      Every breaking change in the upcoming 4.0 release, with the one-line
      fix for each and a codemod table for the mechanical renames.

   .. grid-item-card:: Benchmarks
      :link: benchmarks
      :link-type: doc

      Validation results and performance benchmarks against reference
      implementations.

   .. grid-item-card:: Methodology Registry
      :link: methodology/REGISTRY
      :link-type: doc

      Academic foundations, equations, and documented edge cases for every
      estimator.

   .. grid-item-card:: Reporting
      :link: methodology/REPORTING
      :link-type: doc

      Conventions for reporting DiD results.

.. toctree::
   :maxdepth: 1
   :hidden:

   references
   R Comparison <r_comparison>
   Python Comparison <python_comparison>
   Migrating to 4.0 <migration-4.0>
   benchmarks
   Methodology Registry <methodology/REGISTRY>
   Reporting <methodology/REPORTING>

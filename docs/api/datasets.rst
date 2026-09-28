Datasets
========

Built-in datasets from published studies for examples, tutorials, and testing.

Canonical data are downloaded from public sources and verified against a pinned SHA-256
on every download and cache read, then cached locally at
``~/.cache/diff_diff/datasets/``. Pass ``force_download=True`` to any loader to refresh
the cache. ``load_card_krueger()``, ``load_castle_doctrine()`` and ``load_mpdta()`` pin an
immutable upstream commit, so the URL itself cannot change under them;
``load_prop99()`` and ``load_walmart()`` read a mutable SSC path over plain HTTP, where
the checksum alone establishes integrity.

Every loader reports its provenance through ``df.attrs["source"]``. A fresh download that
fails verification - transport error, size limit, or checksum mismatch - falls back to a
checksum-valid cache entry when one exists, so canonical data is never downgraded to
generated data while verified bytes are on disk. A checksum mismatch additionally emits a
``UserWarning`` naming the integrity failure, since it means the upstream file no longer
matches the pin. When no verified copy is available at all, the loader emits one
``UserWarning`` containing ``SYNTHETIC`` and returns a generated fallback frame marked
``df.attrs["source"] == "synthetic_fallback"``. For
``load_card_krueger()``, ``load_castle_doctrine()`` and ``load_mpdta()`` that fallback
also covers parse and schema-validation failures; ``load_prop99()`` and ``load_walmart()``
run their validation outside the fallback boundary and raise instead. Numbers computed on
a fallback frame are not replications of the published study. ``load_divorce_laws()`` has
no verified source configured and returns synthetic data on every call.

Dataset Loaders
---------------

load_card_krueger
~~~~~~~~~~~~~~~~~

Card & Krueger (1994) minimum wage study. Classic 2x2 DiD comparing fast-food
employment in New Jersey (treated) and Pennsylvania (control) around NJ's 1992
minimum wage increase.

.. autofunction:: diff_diff.load_card_krueger

Example
^^^^^^^

.. code-block:: python

   from diff_diff.datasets import load_card_krueger
   from diff_diff import DifferenceInDifferences

   ck = load_card_krueger()

   # Reshape to long format for DiD estimation
   ck_long = ck.melt(
       id_vars=['store_id', 'state', 'treated'],
       value_vars=['emp_pre', 'emp_post'],
       var_name='period', value_name='employment'
   )
   ck_long['post'] = (ck_long['period'] == 'emp_post').astype(int)

   # 26 store-waves have no employment reading in the source survey; estimators
   # reject missing outcomes rather than dropping them silently
   ck_long = ck_long.dropna(subset=['employment'])

   did = DifferenceInDifferences()
   results = did.fit(ck_long, outcome='employment', treatment='treated', post='post')

load_castle_doctrine
~~~~~~~~~~~~~~~~~~~~

Castle doctrine (Stand Your Ground) gun law study. Staggered adoption of
self-defense law expansions across U.S. states (2000--2010), suitable for
Callaway--Sant'Anna or Sun--Abraham estimation.

.. autofunction:: diff_diff.load_castle_doctrine

Example
^^^^^^^

.. code-block:: python

   from diff_diff.datasets import load_castle_doctrine
   from diff_diff import CallawaySantAnna

   castle = load_castle_doctrine()
   cs = CallawaySantAnna(control_group="never_treated")
   results = cs.fit(
       castle,
       outcome="homicide_rate",
       unit="state",
       time="year",
       first_treat="first_treat"
   )

load_divorce_laws
~~~~~~~~~~~~~~~~~

Always-synthetic teaching panel of unilateral (no-fault) divorce law reforms,
modelled on the staggered adoption studied by Stevenson & Wolfers (2006), with
outcomes for divorce rate, female labor force participation, and female suicide
rate. No verified source is configured for this loader, so it warns and returns
generated data on every call and is not suitable for replication.

.. autofunction:: diff_diff.load_divorce_laws

Example
^^^^^^^

.. code-block:: python

   from diff_diff.datasets import load_divorce_laws
   from diff_diff import CallawaySantAnna

   divorce = load_divorce_laws()
   cs = CallawaySantAnna(control_group="never_treated")
   results = cs.fit(
       divorce,
       outcome="divorce_rate",
       unit="state",
       time="year",
       first_treat="first_treat"
   )

load_mpdta
~~~~~~~~~~

County teen-employment panel (Callaway & Sant'Anna 2021): the canonical
2,500-row ``mpdta`` dataset distributed with the R ``did`` package, covering
staggered minimum wage increases (2003--2007).

.. autofunction:: diff_diff.load_mpdta

Example
^^^^^^^

.. code-block:: python

   from diff_diff.datasets import load_mpdta
   from diff_diff import CallawaySantAnna

   mpdta = load_mpdta()
   cs = CallawaySantAnna()
   results = cs.fit(
       mpdta,
       outcome="lemp",
       unit="countyreal",
       time="year",
       first_treat="first_treat"
   )

load_prop99
~~~~~~~~~~~

California Proposition 99 tobacco program study (Lee--Wooldridge cohort format
of the Abadie--Diamond--Hainmueller 2010 data). Log per capita cigarette sales
for 39 states (1970--2000) with a single treated unit (California, treated from
1989) -- the canonical small-sample DiD and synthetic control setting.

.. autofunction:: diff_diff.load_prop99

Example
^^^^^^^

.. code-block:: python

   from diff_diff.datasets import load_prop99
   from diff_diff import DifferenceInDifferences

   prop99 = load_prop99()
   prop99["treated_state"] = (prop99["first_year"] > 0).astype(int)
   prop99["post"] = (prop99["year"] >= 1989).astype(int)

   did = DifferenceInDifferences()
   results = did.fit(
       prop99, outcome="lcigsale", treatment="treated_state", post="post"
   )

load_walmart
~~~~~~~~~~~~

Walmart entry county panel (Lee & Wooldridge 2025 sample, derived from County
Business Patterns data as constructed by Brown & Butts). Log retail and
wholesale employment for 1,277 counties (1977--1999) with staggered first
store openings (1986--1999) and 391 never-treated counties.

.. autofunction:: diff_diff.load_walmart

Example
^^^^^^^

.. code-block:: python

   from diff_diff.datasets import load_walmart
   from diff_diff import CallawaySantAnna

   walmart = load_walmart()
   cs = CallawaySantAnna(control_group="never_treated")
   results = cs.fit(
       walmart,
       outcome="log_retail_emp",
       unit="cid",
       time="year",
       first_treat="first_year"
   )

Utility Functions
-----------------

load_dataset
~~~~~~~~~~~~

Generic loader that fetches a dataset by name.

.. autofunction:: diff_diff.load_dataset

list_datasets
~~~~~~~~~~~~~

List all available datasets with descriptions.

.. autofunction:: diff_diff.list_datasets

clear_cache
~~~~~~~~~~~~

Remove all cached dataset files from ``~/.cache/diff_diff/datasets/``.

.. autofunction:: diff_diff.clear_cache

Listing and Loading Datasets
----------------------------

.. code-block:: python

   from diff_diff.datasets import list_datasets, load_dataset

   # See what's available
   for name, description in list_datasets().items():
       print(f"{name}: {description}")

   # Load by name
   df = load_dataset("card_krueger")
   print(df.shape)
   print(df.columns.tolist())

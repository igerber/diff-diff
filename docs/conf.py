# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add repository root to sys.path so autodoc imports from checked-out source
# without needing pip install (which would require the Rust/maturin toolchain).
sys.path.insert(0, os.path.abspath(".."))

import diff_diff

# -- Project information -----------------------------------------------------
project = "diff-diff"
copyright = "2026, diff-diff contributors"
author = "diff-diff contributors"
release = diff_diff.__version__
version = ".".join(diff_diff.__version__.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "nbsphinx",
    "myst_parser",
    "sphinx_design",
]

# MyST renders the three in-site markdown pages (methodology/REGISTRY.md,
# methodology/REPORTING.md, migration-4.0.md) so cross-refs use :doc: instead
# of off-site blob/main URLs (stable-docs readers otherwise land on a different
# revision than their package version). dollarmath/amsmath cover the registry's
# LaTeX; heading anchors to depth 4 make its GitHub-style #section links resolve.
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 4


templates_path = ["_templates"]
# Only the two methodology pages and the 4.0 migration guide are published;
# every other repo-internal markdown under docs/ stays out of the build
# (performance/benchmark notes are deliberately NOT on RTD — see the repo
# convention — and un-toctree'd .md files would fail the -W build as orphans).
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "business-strategy.md",
    "dev-status.md",
    "performance-plan.md",
    "performance-scenarios.md",
    "practitioner-guide-evaluation.md",
    "survey-roadmap.md",
    "v4-design.md",
    "methodology/continuous-did.md",
    "methodology/rddensity-source-notes.md",
    "methodology/survey-theory.md",
    "methodology/variance-conventions.md",
    # Internal paper-review notes (methodology validation artifacts).
    "methodology/papers/*",
    "tutorials/README.md",
]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "diff-diff documentation"
# The homepage is a full-width card-grid landing page; suppress the (empty)
# primary sidebar rail there. Matches only the root docname.
html_sidebars = {"index": []}
# Use RTD's canonical URL when available; fall back to stable for local builds.
_canonical_url = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://diff-diff.readthedocs.io/en/stable/",
)
html_baseurl = _canonical_url
html_extra_path = [
    "../diff_diff/guides/llms.txt",
    "../diff_diff/guides/llms-full.txt",
    "../diff_diff/guides/llms-practitioner.txt",
    "../diff_diff/guides/llms-autonomous.txt",
    # Overrides RTD's allow-everything default at the domain root: keeps the
    # ~60 thin _modules/ source-view pages out of crawlers.
    "robots.txt",
]
sitemap_url_scheme = "{link}"

html_theme_options = {
    "logo": {
        "text": "diff-diff",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/igerber/diff-diff",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/diff-diff/",
            "icon": "fa-brands fa-python",
        },
    ],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # Version dropdown (stable/latest). json_url points at the latest build
    # so new entries propagate to every published version without rebuilds.
    # ONE-SELECTOR POLICY: this navbar switcher replaces the RTD flyout
    # (Settings -> Addons -> Flyout menu is disabled in the RTD dashboard,
    # decided 2026-07-20) - re-enabling the flyout would put two version
    # controls with different version lists on every page.
    # check_switcher=False: the build-time URL probe would fail -W on CI and
    # on the first RTD build (the URL only exists after this change ships);
    # the switcher itself is fetched client-side at page load.
    "switcher": {
        "json_url": "https://diff-diff.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": os.environ.get("READTHEDOCS_VERSION", "latest"),
    },
    "check_switcher": False,
    "navigation_depth": 3,
    "show_toc_level": 2,
    # Live-filtering search overlay. Safe to enable now that search-result
    # excerpt rendering survives notebook anchors (searchtools-css-escape.js).
    "search_as_you_type": True,
}

# -- Options for sphinxext-opengraph -----------------------------------------
ogp_site_url = _canonical_url
ogp_site_name = "diff-diff"
ogp_description_length = 200
ogp_type = "website"
ogp_enable_meta_description = True
ogp_social_cards = {
    "line_color": "#1f77b4",
}

# -- Options for nbsphinx ---------------------------------------------------
nbsphinx_execute = "never"
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
    <p class="admonition-title">Interactive notebook</p>
    <p>
    This tutorial is a Jupyter notebook. You can
    <a href="https://github.com/igerber/diff-diff/blob/main/docs/{{ docname }}">view it on GitHub</a>
    or download it to run locally.
    </p>
    </div>
"""

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- ReadTheDocs version-aware banner ----------------------------------------
# Shows a warning on development builds so users know they may be reading
# docs for unreleased features. Uses PyData theme's announcement bar on RTD,
# falls back to rst_prolog for local builds.
rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
rtd_version_type = os.environ.get("READTHEDOCS_VERSION_TYPE", "")

if rtd_version == "latest" or rtd_version_type == "branch":
    html_theme_options["announcement"] = (
        "This documentation is for the <strong>development version</strong> of diff-diff. "
        "It may describe features not yet available in the latest PyPI release. "
        'Use the version selector to switch to <a href="/en/stable/">stable</a>.'
    )


# -- Custom CSS / JS ---------------------------------------------------------
def setup(app):
    app.add_css_file("custom.css")
    # CSS.escape()s section anchors so notebook heading ids containing
    # ' ( ) : etc. don't crash search-result excerpt rendering. NB: on
    # search.html this file is emitted BEFORE searchtools.js (custom js
    # renders with the early script group; searchtools comes from the
    # search page template), so the wrapper defers to DOMContentLoaded -
    # see the file's header comment before touching load order.
    app.add_js_file("searchtools-css-escape.js")
    # Keeps pre-rename numbered section deep links ("#3.-Fit-Event-Study")
    # working after the 2026-07 heading-number strip: rewrites the hash to
    # the renamed fragment when the legacy one no longer exists.
    app.add_js_file("legacy-fragment-redirect.js")

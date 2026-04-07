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
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "llms.txt", "llms-full.txt"]

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
html_title = "diff-diff: Difference-in-Differences Causal Inference for Python"
# Use RTD's canonical URL when available; fall back to stable for local builds.
_canonical_url = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://diff-diff.readthedocs.io/en/stable/",
)
html_baseurl = _canonical_url
html_extra_path = ["llms.txt", "llms-full.txt"]
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
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_depth": 3,
    "show_toc_level": 2,
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

# -- Custom CSS --------------------------------------------------------------
def setup(app):
    app.add_css_file("custom.css")

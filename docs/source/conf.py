# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Expose the package source so autodoc can import it without installation.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
from ndict_tools import __version__  # noqa: E402

project = "Nested Dictionary Tools"
copyright = "2024-2026, biface"
author = "biface"
release = __version__
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# autodoc
autoclass_content = "both"
autodoc_member_order = "bysource"
autosummary_generate = True

# napoleon (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# todo
todo_include_todos = True

# doctest: `sphinx-build -b doctest` runs the explicit ``.. doctest::`` blocks of
# the pages only. Docstring examples rendered by autodoc are not collected.
doctest_test_doctest_blocks = ""

language = "en"

# -- Internationalisation (DD-028) -------------------------------------------
# Catalogs live in docs/source/locales/<lang>/LC_MESSAGES/, one per source
# document. Source locations are left out of the catalogs so that editing a page
# does not rewrite the line references of every translation.
locale_dirs = ["locales/"]
gettext_compact = False
gettext_location = False

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

html_logo = "_static/images/logo.svg"
html_favicon = "_static/images/logo.svg"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# -- External links ----------------------------------------------------------
# Defines :issue:`N` and :pr:`N` shorthand roles.
extensions.append("sphinx.ext.extlinks")
extlinks = {
    "issue": ("https://github.com/biface/ndt/issues/%s", "issue #%s"),
    "pr": ("https://github.com/biface/ndt/pull/%s", "PR #%s"),
}

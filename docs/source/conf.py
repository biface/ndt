# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
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
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# autodoc
# Class docstrings only: each class documents its parameters, so the private
# __init__ of _StackedDict does not appear on the pages of the public classes.
autoclass_content = "class"
autodoc_member_order = "bysource"
autosummary_generate = True

# Public pages show public class names in signatures. The code keeps its
# annotations on the private base classes, which accept every class of the
# family, including a future one that does not derive from NestedDictionary.
# The rewrite applies to objects documented under ``ndict_tools.<Name>`` only;
# the internal pages (``ndict_tools.tools...``) keep the real annotations.
_PUBLIC_NAMES = [
    (re.compile(r"\b(?:ndict_tools\.tools\.)?_StackedDict\b"), "NestedDictionary"),
    (re.compile(r"\b(?:ndict_tools\.tools\.)?_CPaths\b"), "CompactPathsView"),
    (re.compile(r"\b(?:ndict_tools\.tools\.)?_Paths\b"), "PathsView"),
]
_INTERNAL_MODULES = {"tools", "core", "serialize", "exception", "_compat"}


def _public_signature(app, what, name, obj, options, signature, return_annotation):
    parts = name.split(".")
    if len(parts) < 2 or parts[0] != "ndict_tools" or parts[1] in _INTERNAL_MODULES:
        return None
    for pattern, public in _PUBLIC_NAMES:
        if signature:
            signature = pattern.sub(public, signature)
        if return_annotation:
            return_annotation = pattern.sub(public, return_annotation)
    return signature, return_annotation


def setup(app):
    app.connect("autodoc-process-signature", _public_signature)


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

# Read the Docs builds each translation as a separate project and sets
# READTHEDOCS_LANGUAGE (e.g. "fr"); local and GitHub Pages builds pass
# -D language=<lang>. English otherwise.
language = os.environ.get("READTHEDOCS_LANGUAGE", "en").replace("-", "_")

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

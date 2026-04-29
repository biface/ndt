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
    "sphinx_multiversion",
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

language = "en"

# -- sphinx-multiversion configuration ---------------------------------------
# Build one doc set per release tag and keep the master branch as dev.
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = r"^master$"
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/tags/v\d+\.\d+\.\d+$"
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = False

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

html_logo = "_static/images/logo.svg"
html_favicon = "_static/images/logo.svg"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Version selector injected by sphinx-multiversion into the sidebar.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "versioning.html",
    ]
}

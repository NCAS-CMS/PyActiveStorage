# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "p5rem"
copyright = "2026, Bryan Lawrence"
author = "Bryan Lawrence"
release = "0.1.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST-parser settings: enable useful extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "paramiko": ("https://docs.paramiko.org/en/stable", None),
}

# ---------------------------------------------------------------------------
# Autodoc defaults
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# ---------------------------------------------------------------------------
# Napoleon (Google / NumPy-style docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# ---------------------------------------------------------------------------
# HTML output — Furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "p5rem"
html_theme_options = {
    "sidebar_hide_name": False,
}

"""Sphinx configuration for the FOOTSIES PPO Agent project."""

import os
import sys

# Make the project root importable so autodoc can import each module
sys.path.insert(0, os.path.abspath(".."))

# ── Project metadata ──────────────────────────────────────────────────────────
project   = "FOOTSIES PPO Agent"
copyright = "2026, Cosmin"
author    = "Cosmin"
release   = "0.1"

# ── Extensions ────────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",      # pull docstrings from source
    "sphinx.ext.viewcode",     # add [source] links to every object
    "sphinx.ext.napoleon",     # understand Google / NumPy docstring style
    "sphinx.ext.intersphinx",  # cross-reference Python stdlib docs
]

intersphinx_mapping = {
    "python":    ("https://docs.python.org/3", None),
    "numpy":     ("https://numpy.org/doc/stable", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
}

# ── Mock heavy runtime deps so autodoc works without installing torch/SB3 ─────
# These packages are not needed to render our own docstrings; mocking them keeps
# the CI build fast (~30 s) and avoids SDL/GPU library requirements on the runner.
# Our documented classes inherit from gymnasium (not mocked), so inheritance
# display is unaffected.
autodoc_mock_imports = [
    "torch",
    "stable_baselines3",
    "footsies_gym",
    "pygame",
    "tensorboard",
]

# ── autodoc defaults ──────────────────────────────────────────────────────────
autodoc_default_options = {
    "members":          True,
    "undoc-members":    True,   # include functions/classes with no docstring
    "show-inheritance": True,
    "private-members":  False,
    "special-members":  "__init__",
}
autodoc_member_order = "bysource"   # preserve the order in the source file
add_module_names   = False          # show Class instead of module.Class in headers

# ── Napoleon (docstring style) ────────────────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring  = False
napoleon_use_param        = True
napoleon_use_rtype        = True

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme = "alabaster"
html_theme_options = {
    "description":       "PPO agent trained on the FOOTSIES fighting game",
    "github_user":       "Cosmin",
    "fixed_sidebar":     True,
    "sidebar_collapse":  False,
}
html_static_path = ["_static"]
html_title       = "FOOTSIES PPO Agent"

# ── Source ────────────────────────────────────────────────────────────────────
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path   = ["_templates"]

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from mock import Mock as MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()
MOCK_MODULES = ['jax', 'jax.lax', 'jax.numpy', 'jaxtyping', 
    'jax_verify', 'jax_verify.src', 'jax_verify.src.linear', 'sympy2jax', 'diffrax', 'equinox', 'equinox.nn']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

project = 'immrax'
copyright = '2023, Akash Harapanahalli'
author = 'Akash Harapanahalli'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = []

autoclass_content = 'both'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

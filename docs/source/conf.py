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
# MOCK_MODULES = ['jax', 'jax.lax', 'jax.numpy', 'jax.core', 'jax.experimental.compilation_cache', 'jax._src', 'jax._src.util', 'jax._src.api', 'jax._src.traceback_util', 'jax.tree_util',
#                 'jax.typing', 'jaxtyping', 'sympy', 'jax_verify', 'jax_verify.src', 'jax_verify.src.linear', 'sympy2jax', 'diffrax', 'equinox', 'equinox.nn', 
#                 'numpy', 'shapely', 'shapely.geometry', 'shapely.ops']
# MOCK_MODULES = ['sympy', 'jax_verify', 'jax_verify.src', 'jax_verify.src.linear', 'sympy2jax', 'diffrax', 'equinox', 'equinox.nn', 
#                 'numpy', 'shapely', 'shapely.geometry', 'shapely.ops']
MOCK_MODULES = ['jax_verify', 'jax_verify.src', 'jax_verify.src.linear']
# MOCK_MODULES = []
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

project = 'immrax'
copyright = '2023, Akash Harapanahalli'
author = 'Akash Harapanahalli'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.mathjax',
    # 'nbsphinx',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = []

autoclass_content = 'both'

autodoc_member_order = 'bysource'

nb_execution_mode = 'off'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

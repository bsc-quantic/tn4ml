# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tn4ml'
copyright = '2022, Barcelona Supercomputing Center - Centro Nacional de Supercomputación'
author = 'Sergio Sánchez Ramírez, Ema Puljak, Sergi Masor Llima, Jofre Valles Muns'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx"
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autodoc_member_order = 'bysource'
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', 'test', '.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'quimb': ('https://quimb.readthedocs.io', None),
    'scipy': ('https://scipy.org/', None),
    'functools': ('https://docs.python.org/3/library/functools.html#module-functools', None),
    'tensorflow': ('https://www.tensorflow.org/', None)
}

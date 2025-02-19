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
import inspect
import sys
sys.path.insert(0, os.path.abspath('./source'))

def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    try:
        # Ensure the filename reflects the path after `tn4ml`
        module_path = info['module'].split('tn4ml.', 1)[-1]  # Keep everything after 'tn4ml.'
        filename = module_path.replace('.', '/')
        
        # Import the module and get the target object
        module = __import__(info['module'], fromlist=[''])
        obj = module
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)
        
        # Retrieve the line number for the specific function or class
        lineno = inspect.getsourcelines(obj)[1]

        if filename == "models":
            filename = f"models/model"
        
        # Format the GitHub URL
        return f"https://github.com/bsc-quantic/tn4ml/blob/master/tn4ml/{filename}.py#L{lineno}"
    except Exception:
        return None

# Define the canonical URL if you are using a custom domain on Read the Docs
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True

# import sys
# sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tn4ml'
copyright = '2024, Barcelona Supercomputing Center - Centro Nacional de Supercomputación'
author = 'Ema Puljak, Sergio Sánchez Ramírez, Sergi Masor Llima, Jofre Vallès-Muns'
release = '1.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_copybutton",
    'sphinx.ext.viewcode',
    'sphinx.ext.linkcode'  # For custom links to source code
    # "sphinx_gallery.gen_gallery",
]

# path to the examples scripts
# sphinx_gallery_conf = {
#     'examples_dirs': ['examples'],   # path to your example scripts
#     'gallery_dirs': ['auto_examples'],  # path to where to save gallery generated output
#     'filename_pattern': r'\.ipynb$'
# }

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

nbsphinx_thumbnails = {
    'examples/mnist_classification': '_static/class.png',
    'examples/mnist_ad': '_static/ad.png',
    'examples/mnist_ad_sweeps': '_static/ad_sweeps.png',
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autodoc_member_order = 'bysource'
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
copybutton_prompt_text = r">>> |\.\.\. "  # Regex to match the prompts
copybutton_only_copy_prompt_lines = False  # Copy all code lines, not just the ones with prompts
copybutton_remove_prompts = True  # Remove the prompts before copying

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', 'test', '.ipynb_checkpoints']

mathjax3_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS', 'useLabelIds': True}},
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinx_book_theme'
html_title = 'tn4ml'
html_logo = "_static/logo.png"
html_static_path = ['_static']
html_css_files = ['custom.css']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'quimb': ('https://quimb.readthedocs.io', None),
    'scipy': ('https://scipy.org/', None),
    'functools': ('https://docs.python.org/3/library/functools.html#module-functools', None),
    'tensorflow': ('https://www.tensorflow.org/', None)
}
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import inspect
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath("./source"))  # noqa: PTH100


def linkcode_resolve(domain, info):  # noqa: D103
    if domain != "py" or not info["module"]:
        return None

    try:
        # Ensure the filename reflects the path after `tn4ml`
        module_path = info["module"].split("tn4ml.", 1)[
            -1
        ]  # Keep everything after 'tn4ml.'
        filename = module_path.replace(".", "/")

        # Import the module and get the target object
        module = __import__(info["module"], fromlist=[""])
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)

        # Retrieve the line number for the specific function or class
        lineno = inspect.getsourcelines(obj)[1]

        if filename == "models":
            filename = "models/model"

        # Format the GitHub URL
        return f"https://github.com/bsc-quantic/tn4ml/blob/master/tn4ml/{filename}.py#L{lineno}"
    except Exception:  # noqa: BLE001
        return None


# Define the canonical URL if you are using a custom domain on Read the Docs
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True

# import sys  # noqa: ERA001
# sys.path.insert(0, os.path.abspath('../../'))  # noqa: ERA001

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tn4ml"
copyright = (  # noqa: A001
    "2026, Barcelona Supercomputing Center - Centro Nacional de Supercomputación"
)
author = "tn4ml contributors"
try:
    release = _pkg_version("tn4ml")
except PackageNotFoundError:
    release = "unknown"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",  # For custom links to source code
    # "sphinx_gallery.gen_gallery",
]

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

nbsphinx_thumbnails = {
    "source/examples/mnist_classification": "_static/class.png",
    "source/examples/mnist_ad": "_static/ad.png",
    "source/examples/mnist_ad_sweeps": "_static/ad_sweeps.png",
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autodoc_member_order = "bysource"
nbsphinx_allow_errors = True
nbsphinx_execute = "never"
copybutton_prompt_text = r">>> |\.\.\. "  # Regex to match the prompts
copybutton_only_copy_prompt_lines = (
    False  # Copy all code lines, not just the ones with prompts
)
copybutton_remove_prompts = True  # Remove the prompts before copying

templates_path = ["_templates"]
exclude_patterns = [
    "build",
    "Thumbs.db",
    ".DS_Store",
    "test",
    ".ipynb_checkpoints",
    "examples/tnad_latent/README.md",
]

mathjax3_config = {
    "TeX": {"equationNumbers": {"autoNumber": "AMS", "useLabelIds": True}},
}

source_suffix = {
    ".rst": "restructuredtext",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = "<span>#</span>"
html_theme = "pydata_sphinx_theme"
html_title = "tn4ml"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "use_edit_page_button": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bsc-quantic/tn4ml",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Paper",
            "url": "https://arxiv.org/abs/2502.13090",
            "icon": "fa-solid fa-file-pdf",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/tn4ml/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "logo": {
        "text": "tn4ml",
        "image_dark": "_static/logo_dark.png",
    },
    "show_toc_level": 1,
    # place icons in the top-right navbar next to the theme toggle
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "search_as_you_type": True,
}


intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "quimb": ("https://quimb.readthedocs.io", None),
    "scipy": ("https://scipy.org/", None),
    "functools": (
        "https://docs.python.org/3/library/functools.html#module-functools",
        None,
    ),
    "tensorflow": ("https://www.tensorflow.org/", None),
}

# Ensure github edit links work (required for use_edit_page_button)
html_context = globals().get("html_context", {})
html_context.update(
    {
        "github_user": "bsc-quantic",
        "github_repo": "tn4ml",
        "github_version": "master",
        "doc_path": "docs",
    }
)

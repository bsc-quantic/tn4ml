from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tn4ml",
    version="1.0",
    author="Ema Puljak, Sergio Sanchez Ramirez, Sergi Masot Llima, Jofre Vallès-Muns",
    description="Tensor Networks for Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bsc-quantic/tn4ml/tree/master",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(),
    python_requires='>=3.8',
    setup_requires=[
        "setuptools >= 38.3.0"
    ],
    install_requires=[
        "autoray>=0.3.0",
        "dask",
        "funcy",
        "numpy",
        "opt_einsum",
        "quimb>=1.4.1",
        "jaxlib",
        "jax",
        "optax",
        "flax",
        "pandas",
    ],
    extras_require={
        "docs": [
            "sphinx>=2.0",
            "sphinx-book-theme",
            "ipykernel",
            "nbsphinx",
            "sphinxcontrib-bibtex",
            "sphinxcontrib-devhelp",
            "sphinxcontrib-htmlhelp",
            "sphinxcontrib-jsmath",
            "sphinxcontrib-qthelp",
            "sphinxcontrib-serializinghtml",
            "sphinx-rtd-theme",
            "sphinx-copybutton",
            "sphinx-gallery",
            "tensorflow",
        ],
        "test": [
            "pytest",
        ],
    },
)

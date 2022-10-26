# Tensor Networks for Anomaly Detection

## Usage

First create a virtualenv using `pyenv` or `conda`. Then install this package using,
```bash
pip install --editable .
```

This will install the package and its dependencies while you can still edit it and the changes will be reflected.

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20

This code requires a patch in `quimb` that is already upstream but that it is not available in release yet. Please install `quimb` from the HEAD:
```bash
pip install git+https://github.com/jcmgray/quimb.git@develop
```
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

### Warning

The Jax library that is used for automatic differentiation is not officially supported on Windows. There are two ways around it:
- Use Windows Subsystem for Linux (WSL - https://docs.microsoft.com/en-us/windows/wsl/install), which can then be used directly in Visual studio code and other development environments (https://code.visualstudio.com/docs/remote/wsl)
- Install from a community built wheel (and not confirmed to be fully stable) from https://github.com/cloudhan/jax-windows-builder. You must choose the right version for your backend (CPU or CUDA) and python version.
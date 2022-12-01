Installation
************

First create a virtualenv using `pyenv` or `conda`. Then install this package using,

``pip install --editable`` .

This will install the package and its dependencies while you can still edit it and the changes will be reflected.

Requirements
------------

- Python_ ≥ 3.8
- NumPy_ ≥ 1.20
- Quimb_ ≥ 1.4.1

.. warning:: 
    The Jax library that is used for automatic differentiation is not officially supported on Windows.
    
    There are two ways around it:
    
    #. Use Windows Subsystem for Linux (WSL_), which can then be used directly in Visual studio code and other development environments.
    #. Install from a community built wheel (and not confirmed to be fully stable) from jax-windows-builder_. You must choose the right version for your backend (CPU or CUDA) and python version.

.. _WSL: https://docs.microsoft.com/en-us/windows/wsl/install
.. _jax-windows-builder: https://github.com/cloudhan/jax-windows-builder>
.. _Python: https://www.python.org/
.. _Numpy: https://numpy.org/
.. _Quimb: https://github.com/jcmgray/quimb
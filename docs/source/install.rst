Installation
************

First create a virtualenv using `pyenv` or `conda`. 

**Install using** `pip`:

.. code-block:: bash

    pip install tn4ml

**Install directly from the git repository**:

.. code-block:: bash

    pip install git+https://github.com/bsc-quantic/tn4ml.git

**Install by cloning the repository and navigate to the root directory of the repository and run**:

.. code-block:: bash

    pip install .

**Or install the package in development mode**:

.. code-block:: bash

    pip install -e .

**For tests, install the package with the test dependencies**:

.. code-block:: bash

    pip install .[test]

**Run the tests**:

.. code-block:: bash

    pytest


**Accelerated runtime**

(Optional) To improve runtime precision set these flags:
.. code-block:: python

    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_default_matmul_precision', 'highest')


**Running on GPU**
Before everything install `JAX` version that supports CUDA and its suitable for runs on GPU.
Checkout how to install here: `jax\[cuda\] <https://docs.jax.dev/en/latest/installation.html#pip-installation-nvidia-gpu-cuda-installed-via-pip-easier>`_.

Next, at the beginning of your script set:

.. code-block:: python

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 - or set any GPU ID
    import jax
    jax.config.update("jax_platform_name", 'gpu')

Then when training `Model` set:
.. code-block:: python

    device = 'gpu'
    model.configure(device=device)

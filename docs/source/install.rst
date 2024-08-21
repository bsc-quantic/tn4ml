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
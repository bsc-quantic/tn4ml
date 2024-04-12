Installation
************

First create a virtualenv using `pyenv` or `conda`. 

Install directly from the git repository using,

``pip install git+https://github.com/bsc-quantic/tn4ml```

Or clone the repository and navigate to the root directory. Then install this package using,

``pip install .``

To install the package in development mode, use,

``pip install -e .``

For tests, install the package with the test dependencies using,

``pip install .[test]``

and run the tests using
``pytest``
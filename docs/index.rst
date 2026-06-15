Tensor Networks for Machine Learning
====================================

**tn4ml** is a Python library for using Tensor Networks in Machine Learning
applications.

It is built on top of `Quimb <https://quimb.readthedocs.io>`_ for
the tensor network objects and `JAX <https://docs.jax.dev>`_ for the optimization
pipeline.

The library currently supports 1D tensor network structures:

- **Matrix Product State** (MPS)
- **Matrix Product Operator** (MPO)
- **Spaced Matrix Product Operator** (SMPO)

together with a range of **embedding** functions, **initialization** techniques,
**objective functions**, and **optimization strategies** that can be mixed and
matched to build and train your own models.

Quickstart
----------

Install from PyPI:

.. code-block:: bash

   pip install tn4ml

Then train a model in a few lines:

.. code-block:: python

   from tn4ml.models import MPS_initialize

   # 1. Initialize a tensor network model (e.g. a Matrix Product State)
   model = MPS_initialize(L=n_features, ...)

   # 2. Configure the optimization pipeline (optimizer, loss, strategy, device)
   model.configure(...)

   # 3. Train on your embedded data
   history = model.train(data)

See :doc:`source/install` for GPU setup and development installs, and
:doc:`source/examples` for end-to-end notebooks.

Where to go next
----------------

- :doc:`source/install` — installation, accelerated runtime, and GPU support.
- :doc:`source/api` — full API reference for models, embeddings, initializers,
  metrics, strategies, and evaluation.
- :doc:`source/examples` — worked examples for classification, anomaly
  detection, and the experiments from our paper.
- :doc:`source/changelog` — release notes and version history.

Citation
--------

If you use **tn4ml** in your work, please cite
`arXiv:2502.13090 <https://arxiv.org/abs/2502.13090>`_:

.. code-block:: bibtex

   @article{puljak2025tn4mltensornetworktraining,
         title={tn4ml: Tensor Network Training and Customization for Machine Learning},
         author={Ema Puljak and Sergio Sanchez-Ramirez and Sergi Masot-Llima and Jofre Vallès-Muns and Artur Garcia-Saez and Maurizio Pierini},
         year={2025},
         eprint={2502.13090},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2502.13090},
   }

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   source/install
   source/api
   source/examples
   source/changelog

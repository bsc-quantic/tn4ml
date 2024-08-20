.. tn4ml documentation master file, created by
   sphinx-quickstart on Wed Nov 30 09:42:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tensor Networks for Machine Learning
====================================

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Getting started

   install

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Modules

   tn4ml.models
   initializers
   embeddings
   metrics
   strategy

.. nbgallery::
   :caption: Examples
   :name: example-gallery

   source/notebooks/mnist_classification.ipynb
   source/notebooks/mnist_ad.ipynb
   source/notebooks/mnist_ad_sweeps.ipynb
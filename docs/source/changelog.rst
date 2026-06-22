Changelog
*********

All notable changes to **tn4ml** are documented here.

v1.1.1
======

**Fixed**

- ``metrics.SemiSupervisedLoss`` indexed a 0-dim scalar with ``[0]`` (``IndexError``);
  it now returns the scalar loss directly.
- ``metrics.MeanSquaredError`` accessed ``output.tensors[0]`` even when
  ``model.apply(data)`` had already collapsed to a single ``Tensor`` (when
  ``len(model) == len(data)``), raising ``AttributeError``; it now reads the contracted
  output correctly in both contraction paths.
- ``metrics.CombinedLoss`` with NumPy-array input was broken: the "missing embedding"
  ``ValueError`` was never raised, and the embedded samples were passed to the error
  function as an unusable list. It now raises when no embedding is given and averages
  the error over the embedded batch.

**Added**

- **Test coverage raised from 65% to ~80%.** New tests across ``eval.py`` (ROC/PR
  plotting and the ``compare_AUC`` / ``compare_TPR_per_FPR`` / ``compare_FPR_per_TPR``
  hyperparameter-sweep helpers), ``metrics.py`` (``OptaxWrapper``,
  ``CrossEntropyWeighted``, ``CombinedLoss``, ``SemiSupervisedNLL``, ``MeanSquaredError``,
  ``SemiSupervisedLoss``, and error paths), the classification and option paths of
  ``mps.py`` / ``mpo.py`` / ``smpo.py`` (``class_index``, ``add_identity``, ``insert``,
  ``compress``, ``canonical_center``), ``tn.py``, the ``ComplexEmbedding`` branch of
  ``embeddings.embed()``, and ``model.py`` (``save`` / ``load_model`` round-trip,
  ``update_tensors``, ``compute_entropy``).
- **README badges** for code coverage, last commit, and PyPI version, alongside the
  existing CI, pre-merge, and docs badges.

**Changed**

- The CI coverage job now publishes a self-hosted coverage badge to a dedicated
  ``badges`` branch (no Codecov account required).
- Raised the CI coverage gate (``--cov-fail-under``) from 50 to 80.

v1.1.0
======

**Fixed**

- **Windows installation.** Pinned ``orbax-checkpoint>=0.11.34`` so the transitive
  ``uvloop`` dependency (which has no Windows wheels) is correctly skipped on
  Windows. ``orbax-checkpoint==0.11.33`` declared ``uvloop`` unconditionally, which
  broke ``pip install tn4ml`` on Windows; ``0.11.34+`` guards it with
  ``platform_system != "Windows"``. The previous ``pip install --no-deps tn4ml``
  workaround is no longer needed.
- Fixed ``mypy`` errors across the codebase (#39).

**Changed**

- **Removed the hardcoded** ``softmax`` from the model — this affects model output
  and evaluation (#40).
- Updated the batching function in ``model.py`` (#37).
- Dropped support for Python < 3.10.
- Renamed the ``test/`` directory to ``tests/``.
- Updated example notebooks and refined extra dependencies.

**Added**

- Developer tooling: pre-commit hooks and CI pre-merge checks (``ruff``, ``mypy``,
  ``bandit``), plus automatic version inheritance in the CI/CD pipeline (#37).

v1.0.5
======

**Changed**

- Refined ``extras_require`` in ``setup.py`` to ensure correct installation of the
  example dependencies. Install them with ``pip install "tn4ml[examples]"``.

**Added**

- New example scripts from the paper
  *"tn4ml: Tensor Network Training and Customization for Machine Learning"*.
- Updated documentation.

v1.0.4
======

**Fixed**

- Normalization issue for large systems (#28).
- Corrected canonization in ``Strategy.Sweeps``.

**Added**

- ``device`` option in ``Model.configure()`` (#26).
- Updated example notebooks.

v1.0.3
======

**Fixed**

- Model training and evaluation (#19).
- Validation and model saving issues (#17).
- ``metrics.CombinedLoss()``.

**Added**

- New ``PatchAmplitudeEmbedding`` embedding (#20, thanks @gabrieledangeli).
- ``model.forward()`` function.

v1.0.2
======

**Fixed**

- Fix in ``initializers.py``.
- Fix in ``model.py`` — affects model evaluation.

v1.0.1
======

**Added**

- A few new features for embeddings.

**Fixed**

- Bug fixes and performance improvements.
- Minor issues found in version 1.0.0.

v1.0.0
======

**Added**

- Initial release of **tn4ml**: tensor networks for machine learning built on top
  of Quimb and JAX, with support for 1D tensor network structures (Matrix Product
  State, Matrix Product Operator, Spaced Matrix Product Operator), embeddings,
  initializers, objective functions, and optimization strategies.

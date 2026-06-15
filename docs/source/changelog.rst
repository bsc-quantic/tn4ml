Changelog
*********

All notable changes to **tn4ml** are documented here.

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

import warnings

from .embeddings import (
    Embedding,
    trigonometric,
    fourier,
    embed
)

from .loss import (
    no_reg,
    reg_norm_logrelu,
    reg_norm_quad,
    error_logquad,
    error_quad,
    loss
)

from .strategy import (
    Strategy,
    Sweeps,
    Global
)

from .util import (
    EarlyStopping
)
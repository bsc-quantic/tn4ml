import warnings

from .embeddings import (
    Embedding,
    trigonometric,
    fourier,
    physics_embedding,
    embed
)

from .loss import (
    no_reg,
    reg_norm_logrelu,
    reg_norm_quad,
    error_logquad,
    error_quad,
    error_cross_entropy,
    loss_fn,
)

from .strategy import (
    Strategy,
    Sweeps,
    Global
)

from .util import (
    EarlyStopping,
    ExponentialDecay,
    ExponentialGrowth,)
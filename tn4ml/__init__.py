import warnings

from .embeddings import (
    Embedding,
    trigonometric,
    fourier,
    physics_embedding,
    whatever_encoding,
    original_inverse,
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
    ExponentialGrowth,
    squeeze_dimensions,
    squeeze_image,
    rearange_image,
    rearanged_dimensions, 
    unsqueeze_image_pooling,
    gramschmidt,
    gramschmidt_row,
    gramschmidt_col)

from .initializers import (
    ones_init,
    zeros_init,
    gramschmidt_init,
    identity_init,
    noise_init
)
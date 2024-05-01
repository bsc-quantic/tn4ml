import warnings

from .initializers import (
    ones_init,
    zeros_init,
    gramschmidt_init,
    identity_init,
    noise_init
)
from .embeddings import (
    Embedding,
    trigonometric,
    fourier,
    original_inverse,
    gaussian_rbf,
    jax_arrays,
    embed
)

from .loss import (
    neg_log_likelihood,
    transformed_squared_norm,
    no_reg,
    reg_log_norm,
    reg_log_norm_relu,
    reg_norm_quad,
    error_logquad,
    error_quad,
    softmax,
    MSE,
    loss_wrapper_optax,
    combined_loss,
)

from .strategy import (
    Strategy,
    Sweeps,
    Global
)

from .util import (
    gramschmidt_row,
    gramschmidt_col,
    return_digits,
    zigzag_order,
    integer_to_one_hot
)
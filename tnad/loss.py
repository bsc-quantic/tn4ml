from numbers import Number
from typing import Callable, Optional

import quimb.tensor as qtn
import autoray
from autoray import do

from tnad.embeddings import Embedding, embed


def no_reg(x):
    return 0

def reg_norm_logrelu(P):
    """Regularization cost using ReLU of the log of the Frobenius-norm of `P`."""
    return do("maximum", 0.0, do("log", P.H.apply(P)))

def reg_norm_quad(P):
    """Regularization cost using the quadratic formula centered in 1 of the Frobenius-norm of `P`."""
    return do("power", do("add", P.H.apply(P), -1.0), 2)

def error_logquad(P, data):
    mps = P.apply(data)
    return do("power", do("add", do("log", mps.H & mps ^ all), -1.0), 2)

def error_quad(P, data):
    mps = P.apply(data)
    return do("power", do("add", mps.H & mps ^ all, -1.0), 2)

def loss(model, data, error: Callable = error_logquad, reg: Callable = no_reg, embedding: Optional[Embedding] = None) -> Number:
    with autoray.backend_like("jax"):
        return do("mean", [error(model, embed(sample, embedding)) for sample in data] if embedding else error(model, data)) + reg(model)
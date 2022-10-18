from numbers import Number
from typing import Callable, Optional
import quimb.tensor as qtn
from autoray import do
from tnad.embeddings import embed, Embedding


def no_reg(x):
    return 0


def reg_norm_logrelu(P):
    """Regularization cost using ReLU of the log of the Frobenius-norm of `P`."""
    return do("maximum", 0.0, do("log", P.H & P ^ all))


def reg_norm_quad(P):
    """Regularization cost using the quadratic formula centered in 1 of the Frobenius-norm of `P`."""
    return do("power", P.H & P ^ all - 1, 2)


def error_logquad(P, data):
    mps = qtn.tensor_network_apply_op_vec(P, data)
    return do("power", do("add", do("log", mps.H & mps ^ all), -1.0), 2)


def error_quad(P, data):
    mps = qtn.tensor_network_apply_op_vec(P, data)
    return do("power", do("add", mps.H & mps ^ all, -1.0), 2)


def loss(model, data, error: Callable = error_logquad, reg: Callable = no_reg, embedding: Optional[Embedding] = None) -> Number:
    return do("mean", [error(model, embed(sample, embedding) if embedding else sample) for sample in data]) + reg(model)

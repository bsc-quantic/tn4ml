from numbers import Number
import quimb.tensor as qtn
from autoray import do
import tnad.embeddings


def no_reg(x):
    return 0


def reg_norm_logrelu(P):
    """Regularization cost using ReLU of the log of the Frobenius-norm of `P`."""
    return do("maximum", 0.0, do("log", P.H & P ^ all))


def reg_norm_quad(P):
    """Regularization cost using the quadratic formula centered in 1 of the Frobenius-norm of `P`."""
    return do("power", do("add", P.H & P ^ all, -1.0), 2)


def loss(model, batch_data, error=None, reg=no_reg) -> Number:
    acc = tnad.loss.reg_norm_quad(model)
    n = len(batch_data)
    for sample in batch_data:
        acc = acc + tnad.loss.error_quad(model, sample) / n

    return acc


def error_logquad(P, data):
    mps = qtn.tensor_network_apply_op_vec(P, data)
    return do("power", do("add", do("log", mps.H & mps ^ all), -1.0), 2)


def error_quad(P, data):
    mps = qtn.tensor_network_apply_op_vec(P, data)
    return do("power", do("add", mps.H & mps ^ all, -1.0), 2)

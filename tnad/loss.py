from numbers import Number
import quimb.tensor as qtn
from autoray import do


def no_reg(x):
    return 0


def reg_norm_logrelu(P):
    """Regularization cost using ReLU of the log of the Frobenius-norm of `P`."""
    return max(0, do("log", P.H & P ^ all))


def reg_norm_quad(P):
    """Regularization cost using the quadratic formula centered in 1 of the Frobenius-norm of `P`."""
    return do("power", P.H & P ^ all - 1, 2)


def loss(model, batch, error=None, reg=no_reg) -> Number:
    return do("mean", [error(model, sample) for sample in batch]) + reg(model)


def error_logquad(P, phi):
    mps = qtn.tensor_network_apply_op_vec(P, phi)
    return do("power", do("log", mps.H & mps ^ all) - 1, 2)


def error_quad(P, phi):
    mps = qtn.tensor_network_apply_op_vec(P, phi)
    return do("power", mps.H & mps ^ all - 1, 2)

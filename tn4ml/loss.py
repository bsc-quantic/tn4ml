from numbers import Number
from typing import Callable, Optional

from autoray import do
import jax.numpy as jnp
import math

from .embeddings import Embedding, embed

""" Examples of Loss functions """

def clustering_l1(model):
    l1 = do("log", ((model.H & model)^all))
    return l1

def clustering_l2(model, input_mps):
    l2 = do("log", do("power", (model.H&input_mps)^all, 2))
    return -l2

def clustering_logquad(model, input_mps):
    return do("power", do("add", do("log", (model.H & input_mps)^all), -1.0), 2)

def no_reg(x):
    return 0

def reg_norm_logrelu(P):
    """Regularization cost using ReLU of the log of the Frobenius-norm of `P`.

    Parameters
    ----------
    P : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator

    Returns
    -------
    float
    """
    return do("maximum", 0.0, do("log", P.H.apply(P)))

def reg_norm_quad(P):
    """Regularization cost using the quadratic formula centered in 1 of the Frobenius-norm of `P`.

    Parameters
    ----------
    P : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator

    Returns
    -------
    float
    """
    return do("power", do("add", P.H.apply(P), -1.0), 2)

def error_logquad(P, data):
    """Example of error calculation when applying :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` `P` to `data`.

    Parameters
    ----------
    P : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    if len(P.tensors) < len(data.tensors):
        inds_contract = []
        for i in range(len(data.tensors)):
            inds_contract.append(f'k{i}')
        mps = (P.H&data)
        for index in inds_contract:
            mps.contract_ind(index)
    else:
        mps = P.apply(data)
    return do("power", do("add", do("log", mps.H & mps ^ all), -1.0), 2) # TODO missing power on mps.H&mps^all

def error_quad(P, data):
    """Example of error calculation when applying :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` `P` to `data`.

    Parameters
    ----------
    P : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    mps = P.apply(data)
    return do("power", do("add", mps.H & mps ^ all, -1.0), 2)

def error_distance_to_origin(P, data):
    """Example of error calculation when applying :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` `P` to `data`.

    Parameters
    ----------
    P : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    mps = P.apply(data)
    return - (mps.H & mps ^ all)

def softmax(z):
    sum_z = do("sum", jnp.asarray([do("power", math.e, z_j) for z_j in z]))
    return jnp.asarray([do("power", math.e, z_i)/sum_z for z_i in z])

def error_cross_entropy(P, data, mps_target):
    """Example of supervised error calculation when applying :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` `P` to `data`.
    
    Parameters
    ----------
    P : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    mps_target: :class:`numpy.ndarray`
        Target data.
    Returns
    -------
    float
    """
    mps = P.apply(data)
    class_vector = mps.tensors[0].data
    class_vector = do("reshape", class_vector, (5,))
    prob_dist = softmax(class_vector)
    return - do("sum", mps_target * do("log", do("power", prob_dist, 2)))

def loss_fn(model, data, error: Callable = error_logquad, reg: Callable = no_reg, embedding: Optional[Embedding] = None) -> Number:
    """Example of Loss function with calculation of error on input data and regularization.

    Parameters
    ----------
    model : :class:`tn4ml.models.Model`
    data: :class:`numpy.ndarray`
        Data used for computing the loss value.
    error: function
        Function for error calculation.
    reg: function
        Function for regularization value calculation.
    embedding: :class:`tn4ml.embeddings.Embedding`
        Data embedding function.

    Returns
    -------
    float
    """
    return do("mean", [error(model, embed(sample, embedding)) for sample in data] if embedding else error(model, data)) + reg(model)
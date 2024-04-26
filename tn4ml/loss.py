from numbers import Number
from typing import Callable, Optional, Callable

#from autoray import do
import jax.numpy as jnp
import jax
import numpy as np
import quimb.tensor as qtn

from .models.model import Model
from .embeddings import Embedding, embed
from .models.smpo import SpacedMatrixProductOperator
from .models.mps import MatrixProductState
from .models.mpo import MatrixProductOperator

def neg_log_likelihood(model: qtn.MatrixProductState, data: qtn.MatrixProductState) -> Number:
    """Negative Log-Likelihood loss.

    Parameters
    ----------
    model : :class:`quimb.tensor.MatrixProductState`
        Matrix Product State model
    data: :class:`quimb.tensor.MatrixProductState`
        Input MPS
    Returns
    -------
    float
    """
    # check if physical dimensions match 
    assert model.tensors[0].shape[-1] == data.tensors[0].shape[-1]

    if len(model.tensors) < len(data.tensors):
        inds_contract = []
        for i in range(len(data.tensors)):
            inds_contract.append(f'k{i}')

        output = (model.H & data)
        for index in inds_contract:
            output.contract_ind(index)

        output = output^all

    elif len(model.tensors) == len(data.tensors):
        output = (model.H & data)^all

    return - jax.lax.log(jax.lax.pow((model.H & data)^all, 2))

def transformed_squared_norm(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """Squared norm of transformed input data.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    assert type(model) == SpacedMatrixProductOperator
    if len(model.tensors) < len(data.tensors):
        inds_contract = []
        for i in range(len(data.tensors)):
            inds_contract.append(f'k{i}')

        mps = (model.H & data)
        for index in inds_contract:
            mps.contract_ind(index)
    else:
        mps = model.apply(data)
    return jax.lax.pow(mps.H & mps ^ all, 2)

def no_reg(x):
    return 0

def reg_log_norm(model) -> Number:
    """Regularization cost - log(Frobenius-norm of `model`)

    Parameters
    ----------
    model : :class:`quimb.tensor.MatrixProductState`
        Matrix Product State model
    Returns
    -------
    float
    """
    assert type(model) in [SpacedMatrixProductOperator,
                           MatrixProductState,
                           MatrixProductOperator,
                           qtn.MatrixProductState,
                           qtn.MatrixProductOperator]
    
    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()
    return jax.lax.log(norm)

def reg_log_norm_pow(model) -> Number:
    """Regularization cost - log(Frobenius-norm of `model`)

    Parameters
    ----------
    model : :class:`quimb.tensor.MatrixProductState`
        Matrix Product State model
    Returns
    -------
    float
    """
    assert type(model) in [SpacedMatrixProductOperator,
                           MatrixProductState,
                           MatrixProductOperator,
                           qtn.MatrixProductState,
                           qtn.MatrixProductOperator]
    
    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()
    return jax.lax.log(jax.lax.pow(norm, 2))

def reg_log_norm_relu(model) -> Number:
    """Regularization cost using ReLU of the log of the Frobenius-norm of `model`.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator

    Returns
    -------
    float
    """
    assert type(model) in [SpacedMatrixProductOperator,
                           MatrixProductState,
                            MatrixProductOperator,
                             qtn.MatrixProductState,
                            qtn.MatrixProductOperator]
    
    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()

    return jax.lax.max(0.0, jax.lax.log(norm))

def reg_norm_quad(model) -> Number:
    """Regularization cost using the quadratic formula centered in 1 of the Frobenius-norm of `model`.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator

    Returns
    -------
    float
    """
    assert type(model) in [SpacedMatrixProductOperator,
                           MatrixProductState,
                           MatrixProductOperator,
                           qtn.MatrixProductState,
                           qtn.MatrixProductOperator]

    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()

    return jax.lax.pow(jax.lax.log(norm) - 1.0, 2)

def error_logquad(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """Example of error calculation when applying :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` `P` to `data`.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    return jax.lax.pow((jax.lax.log(transformed_squared_norm(model, data)) - 1.0), 2)

def error_quad(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState) -> Number:
    """Example of error calculation when applying :class:`tn4ml.models.smpo.SpacedMatrixProductOperator` `P` to `data`.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input mps.
    Returns
    -------
    float
    """
    return jax.lax.pow((transformed_squared_norm(model, data) - 1.0), 2)

def softmax(z, position) -> Number:
    """ Softmax function.

    Parameters
    ----------
    z : :class:`jnp.array``
        Predicted probabilities.
    position: int
        Indicates for which class we are calculating softmax value.
    Returns
    -------
    float
    """
    return jnp.exp(z[position]) / jnp.sum(jnp.exp(z))

def cross_entropy_softmax(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState, targets: jnp.array) -> Number:
    """Cross-entropy loss function for supervised learning.
    
    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input Matrix Product State
    targets: :class:`numpy.ndarray`
        Target class vector. Example = [1 0 0 0] for n_classes = 4.
    
    Returns
    -------
    float
    
    """
    if len(model.tensors) < len(data.tensors):
        inds_contract = []
        for i in range(len(data.tensors)):
            inds_contract.append(f'k{i}')

        output = (model.H & data)
        for index in inds_contract:
            output.contract_ind(index)

        output = output^all
    elif len(model.tensors) == len(data.tensors):
        if hasattr(model, 'apply'):
            output = model.apply(data)^all
        else:
            output = (model.H & data)^all
    else:
        raise ValueError('Number of tensors for input data MPS needs to be higher or equal number of tensors in model.')
    
    output = output.data.reshape((len(targets), ))
    output = output/jnp.linalg.norm(output)

    return - jnp.log(softmax(output, jnp.argmax(targets)))

def MSE(model: SpacedMatrixProductOperator, data: qtn.MatrixProductState, targets: jnp.array) -> Number:
    """Mean Squared Error loss function for supervised learning.
    
    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input Matrix Product State
    targets: :class:`numpy.ndarray`
        Target class vector. Example = [1 0 0 0] for n_classes = 4.
    
    Returns
    -------
    float
    """
    if len(model.tensors) < len(data.tensors):
        inds_contract = []
        for i in range(len(data.tensors)):
            inds_contract.append(f'k{i}')

        output = (model.H & data)
        for index in inds_contract:
            output.contract_ind(index)

        output = output^all
    elif len(model.tensors) == len(data.tensors):
        if hasattr(model, 'apply'):
            output = model.apply(data)^all
        else:
            output = (model.H & data)^all
    else:
        raise ValueError('Number of tensors for input data MPS needs to be higher or equal number of tensors in model.')
    
    output = output.data.reshape((len(targets), ))
    output = output/jnp.linalg.norm(output)

    return jnp.mean(jnp.square(output - targets))

def loss_wrapper_optax(optax_loss = None) -> Callable:
    """Wrapper around optax loss functions for supervised learning. Make sure you got all inputs to loss function correct.
    Refer to documentation for each loss to https://optax.readthedocs.io/en/latest/api/losses.html .
    Make sure SMPO has only one output with dimension = number of classes.
    
    Parameters
    ----------
    model : :class:`tn4ml.models.model.Model`
        Tensor Network model.
    data: :class:`quimb.tensor.MatrixProductState`
        Input Matrix Product State
    y_true: :class:`numpy.ndarray`
        Target class vector. Example = [1 0 0 0] for n_classes = 4.
    kwargs : dict
        Additional arguments for optax loss function.
    Returns
    -------
    float
    """

    assert optax_loss != None

    def loss_optax(model: Model, data: qtn.MatrixProductState, y_true: Optional[jnp.array] = None, **kwargs) -> Number:

        """Loss function for learning. Make sure you got all inputs to loss function correct.
        
        Parameters
        ----------
        model : :class:`tn4ml.models.model.Model`
            Tensor Network model.
        data: :class:`quimb.tensor.MatrixProductState`
            Input Matrix Product State
        y_true: :class:`numpy.ndarray`
            Target class vector. Example = [1 0 0 0] for n_classes = 4.
        kwargs : dict
            Additional arguments for optax loss function.

        Returns
        -------
        float
        """

        # if not callable(getattr(model, "apply", None)):
        #     raise AttributeError("Model should have 'apply' method.")

        # if hasattr(model, 'lower_inds') and len(list(model.lower_inds)) > 1:
        #     raise ValueError('Model has more than one output! Rethink your contraction path to contract to vector size = (n_classes,).')
        
        if isinstance(model, SpacedMatrixProductOperator):
            if len(model.tensors) < len(data.tensors):
                inds_contract = []
                for i in range(len(data.tensors)):
                    inds_contract.append(f'k{i}')

                output = (model.H & data)
                for index in inds_contract:
                    output.contract_ind(index)
                
                output = output^all
                output = output.data.reshape((len(y_true), ))
                
                y_pred = jnp.log(output)
                # normalize
                y_pred = y_pred/jnp.linalg.norm(y_pred)
            else:
                output = model.apply(data)
                
                if len(output.tensors) > 1:
                    output = output^all
                    y_pred = output.data
                else:
                    y_pred = jnp.expand_dims(jnp.squeeze(output.tensors[0].data), axis=0)
                
                if y_true is not None:
                    y_true = jnp.expand_dims(jnp.squeeze(y_true), axis=0)

        elif isinstance(model, MatrixProductState):
            y_pred = model & data ^ all
        
        # normalize
        y_pred = y_pred/jnp.linalg.norm(y_pred)

        if y_true is not None:
            return optax_loss(y_pred, y_true, **kwargs)
        else:
            return optax_loss(y_pred, **kwargs)
    return loss_optax

def combined_loss(model: Model, data: np.ndarray = None, error: Callable = error_logquad, reg: Callable = no_reg, embedding: Optional[Embedding] = None) -> Number:
    """Example of Loss function with calculation of error on input data and regularization.

    Parameters
    ----------
    model : :class:`tn4ml.models.Model`
        Tensor Network with parametrized tensors.
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
    if not data:
        raise ValueError('Provide input data!')
    
    return np.mean([error(model, embed(sample, embedding)) for sample in data] if embedding else error(model, data)) + reg(model)
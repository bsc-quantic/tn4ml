# Examples of loss functions for supervised and unsupervised learning.

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import quimb.tensor as qtn

from .embeddings import Embedding, embed
from .models.model import Model
from .models.mps import MatrixProductState
from .models.smpo import SpacedMatrixProductOperator


def NegLogLikelihood(  # noqa: N802
    model: qtn.MatrixProductState, data: qtn.MatrixProductState
) -> jax.Array:
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
        inds_contract = [f"k{i}" for i in range(len(data.tensors))]

        output = model.H & data
        for index in inds_contract:
            output.contract_ind(index)

        output = output ^ all

    elif len(model.tensors) == len(data.tensors):
        # assuming that model and data has same names for physical indices
        output = (model.H & data) ^ all
    else:
        raise ValueError(
            "Number of tensors for input data MPS needs to be higher or equal number of tensors in model."
        )

    return -jax.lax.log(jax.lax.pow(output, 2))


def TransformedSquaredNorm(  # noqa: N802
    model: SpacedMatrixProductOperator, data: qtn.MatrixProductState
) -> jax.Array:
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
    if len(model.tensors) < len(data.tensors):
        inds_contract = [f"k{i}" for i in range(len(data.tensors))]

        mps = model.H & data
        for index in inds_contract:
            mps.contract_ind(index)
    else:
        mps = model.apply(data)

    return jax.lax.pow((mps.H & mps) ^ all, 2)


def NoReg(_x):  # noqa: N802
    """Return zero regularization."""
    return 0


def LogFrobNorm(model) -> jax.Array:  # noqa: N802
    """Regularization cost - log(Frobenius-norm of `model`).

    Parameters
    ----------
    model : :class:`quimb.tensor.MatrixProductState`
        Matrix Product State model

    Returns
    -------
    float
    """
    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()
    return jax.lax.log(norm)


def LogPowFrobNorm(model) -> jax.Array:  # noqa: N802
    """Regularization cost - log(squared(Frobenius-norm of `model`)).

    Parameters
    ----------
    model : :class:`quimb.tensor.MatrixProductState`
        Matrix Product State model

    Returns
    -------
    float
    """
    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()
    return jax.lax.log(jax.lax.pow(norm, 2))


def LogReLUFrobNorm(model) -> jax.Array:  # noqa: N802
    """Regularization cost using ReLU of the log of the Frobenius-norm of `model`.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator

    Returns
    -------
    float
    """
    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()

    return jax.lax.max(0.0, jax.lax.log(norm).astype(jnp.float64))


def QuadFrobNorm(model) -> jax.Array:  # noqa: N802
    """Regularization cost using the quadratic formula centered in 1 of the Frobenius-norm of `model`.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator

    Returns
    -------
    float
    """
    if type(model) in [SpacedMatrixProductOperator]:
        tn = model.H.apply(model)
        norm = tn.contract_cumulative(tn.site_tags)
    else:
        norm = model.norm()

    return jax.lax.pow(jax.lax.log(norm) - 1.0, 2)


def LogQuadNorm(  # noqa: N802
    model: SpacedMatrixProductOperator, data: qtn.MatrixProductState
) -> jax.Array:
    """Calculate log-quadratic error after applying `model` to `data`.

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
    return jax.lax.pow((jax.lax.log(TransformedSquaredNorm(model, data)) - 1.0), 2)


def QuadNorm(  # noqa: N802
    model: SpacedMatrixProductOperator, data: qtn.MatrixProductState
) -> jax.Array:
    """Calculate quadratic error after applying `model` to `data`.

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
    return jax.lax.pow((TransformedSquaredNorm(model, data) - 1.0), 2)


def SemiSupervisedLoss(  # noqa: N802
    model: SpacedMatrixProductOperator,
    data: qtn.MatrixProductState,
    y_true: float,
    **_kwargs,
) -> jax.Array:
    """Loss function for semi-supervised learning.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input Matrix Product State
    y_true: :class:`Number`
        Target class percentage.

    Returns
    -------
    float
    """
    norm = LogQuadNorm(model, data) + 0.3 * LogReLUFrobNorm(model)
    loss_value = jax.lax.pow(y_true * (1 / norm) + (1 - y_true) * norm, 2)
    return loss_value[0]


def SemiSupervisedNLL(  # noqa: N802
    model: SpacedMatrixProductOperator,
    data: qtn.MatrixProductState,
    y_true: jnp.ndarray | None = None,
    **_kwargs,
) -> jax.Array:
    """Loss function for semi-supervised learning.

    Parameters
    ----------
    model : :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        Spaced Matrix Product Operator
    data: :class:`quimb.tensor.MatrixProductState`
        Input Matrix Product State
    y_true: :class:`Number`
        Target class percentage.

    Returns
    -------
    float
    """
    mps = model.apply(data)
    norm = jnp.array(mps.arrays).sum()
    norm = jax.lax.pow((jax.lax.log(norm) - 1.0), 2)

    output = (model.H & data) ^ all
    output = output.data.reshape((2,))
    class_error = optax.softmax_cross_entropy_with_integer_labels(
        output, jnp.squeeze(y_true)
    )

    return (
        class_error
        + output * (1 / (norm))
        + (1 - output) * (norm)
        + 0.3 * LogReLUFrobNorm(model)
    )


def Softmax(z, position) -> jax.Array:  # noqa: N802
    """Softmax function.

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


def CrossEntropySoftmax(  # noqa: N802
    model: SpacedMatrixProductOperator,
    data: qtn.MatrixProductState,
    targets: jnp.ndarray,
) -> jax.Array:
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
        inds_contract = [f"k{i}" for i in range(len(data.tensors))]

        output = model.H & data
        for index in inds_contract:
            output.contract_ind(index)

        output = output ^ all
    elif len(model.tensors) == len(data.tensors):
        if hasattr(model, "apply"):
            output = model.apply(data) ^ all
        else:
            output = (model.H & data) ^ all
    else:
        raise ValueError(
            "Number of tensors for input data MPS needs to be higher or equal number of tensors in model."
        )

    output = output.data.reshape((len(targets),))
    output = output / jnp.linalg.norm(output)

    return -jnp.log(Softmax(output, jnp.argmax(targets)))


def MeanSquaredError(  # noqa: N802
    model: SpacedMatrixProductOperator,
    data: qtn.MatrixProductState,
    targets: jnp.ndarray,
) -> jax.Array:
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
        inds_contract = [f"k{i}" for i in range(len(data.tensors))]

        output = model.H & data
        for index in inds_contract:
            output.contract_ind(index)

        output = output ^ all
    elif len(model.tensors) == len(data.tensors):
        if hasattr(model, "apply"):
            output = model.apply(data) ^ all
        else:
            output = model | data
            for ind in data.outer_inds():
                output.contract_ind(ind=ind)

            tags = list(qtn.tensor_core.get_tags(output))
            tags_to_drop = []
            for j in range(len(model.tensors) // 2 - 1):
                output.contract_between(tags[j], tags[j + 1])
                tags_to_drop.extend([tags[j]])
            output.drop_tags(tags_to_drop)
            output.fuse_multibonds_()

            tags_to_drop = []
            for j in range(len(model.tensors) - 1, len(model.tensors) // 2 - 1, -1):
                output.contract_between(tags[j], tags[j - 1])
                tags_to_drop.extend([tags[j]])
            output.drop_tags(tags_to_drop)
    else:
        raise ValueError(
            "Number of tensors for input data MPS needs to be higher or equal number of tensors in model."
        )

    output = output.tensors[0].data.reshape((len(targets),))
    output = output / jnp.linalg.norm(output)

    return jnp.mean(jnp.square(output - targets))


def OptaxWrapper(optax_loss=None) -> Callable:  # noqa: N802
    """Wrap optax loss functions for supervised learning.

    Make sure you got all inputs to loss function correct.
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
    assert optax_loss is not None

    def loss_optax(
        model: Model,
        data: qtn.MatrixProductState,
        y_true: jnp.ndarray | None = None,
        **kwargs,
    ) -> jax.Array:
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
        if isinstance(model, SpacedMatrixProductOperator):
            if len(model.tensors) < len(data.tensors):
                inds_contract = [f"k{i}" for i in range(len(data.tensors))]

                output = model.H & data
                for index in inds_contract:
                    output.contract_ind(index)

                output = output ^ all
                output = output.data.reshape((len(y_true),))

                y_pred = jnp.log(output)
            else:
                output = model.apply(data)

                if len(output.tensors) > 1:
                    output = output ^ all
                    y_pred = output.data
                else:
                    y_pred = jnp.expand_dims(
                        jnp.squeeze(output.tensors[0].data), axis=0
                    )

                if y_true is not None:
                    y_true = jnp.expand_dims(jnp.squeeze(y_true), axis=0)

        elif isinstance(model, MatrixProductState):
            y_pred = (model & data) ^ all
        else:
            y_pred = (model & data) ^ all
            y_pred = jnp.expand_dims(jnp.squeeze(y_pred.data), axis=0)

        # normalize
        y_pred = y_pred / jnp.linalg.norm(y_pred)

        if y_true is not None:
            if len(y_true.shape) == 1:
                y_true = jnp.expand_dims(y_true, axis=0)
            return optax_loss(y_pred, y_true, **kwargs)
        return optax_loss(y_pred, **kwargs)

    return loss_optax


def CrossEntropyWeighted(class_weights: jnp.ndarray = None) -> Callable:  # noqa: N802
    """Build a weighted cross-entropy loss function."""

    def cross_entropy(
        model: Model,
        data: MatrixProductState,
        y_true: jnp.ndarray = None,
        **_kwargs,
    ) -> jax.Array:
        """
        Compute the weighted cross-entropy loss.

        Parameters
        ----------
            model : :class:`tn4ml.models.model.Model`
                Tensor Network model.
            data: :class:`quimb.tensor.MatrixProductState`
                Input Matrix Product State
            y_true: :class:`jax.numpy.ndarray`
                Target class vector. Example = [1 0 0 0] for n_classes = 4.
            class_weights: :class:`numpy.ndarray`
                Class weights, shape (num_classes,).
            kwargs : dict
                Additional arguments for optax loss function.

        Returns
        -------
            Weighted cross-entropy loss.
        """
        # Compute per-sample weights based on class labels
        sample_weights = jnp.sum(
            y_true * jnp.array(class_weights), axis=-1
        )  # Shape: (batch_size,)

        logits = OptaxWrapper(optax.softmax_cross_entropy)(model, data, y_true)

        # Apply sample weights
        weighted_loss = logits * jnp.array(sample_weights)  # Shape: (batch_size,)

        # Return mean weighted loss
        return jnp.array(weighted_loss)

    return cross_entropy


def CombinedLoss(  # noqa: N802
    model: Model,
    data: qtn.MatrixProductState | np.ndarray,
    y_true: jnp.ndarray | None = None,
    error: Callable = LogQuadNorm,
    reg: Callable = NoReg,
    embedding: Embedding | None = None,
) -> jax.Array:
    """
    Unified Loss function combining error computation and regularization.

    Parameters
    ----------
    model : :class:`tn4ml.models.Model`
        Tensor Network with parametrized tensors.
    data : Union[:class:`qtn.MatrixProductState`, :class:`numpy.ndarray`]
        Data used for computing the loss value.
    y_true : Optional[:class:`jnp.array`]
        True labels for supervised learning tasks (only applicable for MatrixProductState data).
    error : function
        Function for error calculation.
    reg : function
        Function for regularization value calculation.
    embedding : :class:`tn4ml.embeddings.Embedding`, optional
        Data embedding function.

    Returns
    -------
    float
        Computed loss value.
    """
    if data is None:
        raise ValueError("Provide input data!")

    if isinstance(data, np.ndarray):
        # Compute loss for NumPy array data
        if embedding:
            data = [embed(sample, embedding) for sample in data]
        else:
            ValueError(
                "Provide embedding function for NumPy array data to embed it into Tensor Network representation."
            )

    if y_true is not None:
        loss = jnp.mean(error(model, data, y_true)) + reg(model)
    else:
        loss = jnp.mean(error(model, data)) + reg(model)

    return loss

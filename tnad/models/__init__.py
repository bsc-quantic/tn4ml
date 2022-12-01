import functools
import operator
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any, Callable, Collection, Optional, Sequence, NamedTuple

import autoray
import funcy
import jax
import numpy as np
import tnad
import tnad.embeddings as embeddings
from tnad.util import EarlyStopping
from quimb import tensor as qtn
import quimb as qu
from scipy.optimize import OptimizeResult
from tnad.strategy import *
from tqdm import tqdm

class Model(qtn.TensorNetwork):
    """:class:`tnad.models.Model` class models training model of class :class:`quimb.tensor.tensor_core.TensorNetwork`.

    Attributes
    ----------
    loss_fn : `Callable`, or `None`
        Loss function. See :mod:`tnad.loss` for examples.
    strategy : :class:`tnad.strategy.Strategy`
        Strategy for computing gradients.
    optimizer : :class:`quimb.tensor.optimize.TNOptimizer`, or different possibilities of optimizers from :func:`quimb.tensor.optimize`. 
    """
    
    def __init__(self):
        """Constructor
        """
        self.loss_fn = None # dict()
        self.strategy = Global()
        self.optimizer = qtn.optimize.ADAM()
    
    def save(self, model_name, dir_name='~'):

        """Saves :class:`tnad.models.Model` to pickle file.

        Parameters
        ----------
        model_name : str
            Name of Model.
        dir_name: str
            Directory for saving Model.
        """

        qu.save_to_disk(self, f'{dir_name}/{model_name}.pkl')
    
    def configure(self, **kwargs):

        """Configures :class:`tnad.models.Model` for training setting the arguments.
        """

        for key, value in kwargs.items():
            if key == "strategy":
                if isinstance(value, Strategy):
                    self.strategy = value
                elif value in ["sweeps", "local", "dmrg"]:
                    self.strategy = Sweeps()
                elif value in ["global"]:
                    self.strategy = Global()
                else:
                    raise ValueError(f'Strategy "{value}" not found')
            elif key in ["optimizer", "loss_fn", "learning_rate"]:
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found")

    def train(self, 
            data: Collection,
            batch_size: Optional[int] = None,
            nepochs: Optional[int] = 1,
            embedding: embeddings.Embedding = embeddings.trigonometric(),
            callbacks: Optional[Sequence[tuple[str, Callable]]] = None,
            earlystop: Optional[EarlyStopping] = None,
            **kwargs):
        
        """Performs the training procedure of :class:`tnad.models.Model`.

        Parameters
        ----------
        data : sequence of :class:`numpy.ndarray`
            Data used for training procedure. 
        batch_size : int, or default `None`
            Number of samples per gradient update.
        nepochs : int
            Number of epochs for training the Model.
        embedding : :class:`tnad.embeddings.Embedding`
            Data embedding function.
        callbacks : sequence of callbacks (metrics) - ``tuple(callback name, `Callable`)``, or default `None`
            List of metrics for monitoring training progress. Each metric function receives (:class:`tnad.models.Model`, :class:`scipy.optimize.OptimizeResult`, :class:`quimb.tensor.optimize.Vectorizer`).
        earlystop : :class:`tnad.util.EarlyStopping`
            Early stopping training when monitored metric stopped improving.
        
        Returns
        -------
        history: dict
            Records training loss and metric values.
        """
        
        num_batches = (len(data)//batch_size)
        
        history = dict()
        history['loss'] = []
        if callbacks:
            for name, _ in callbacks:
                history[name] = []
        
        if earlystop:
            if earlystop.monitor not in history.keys():
                raise ValueError(f'This metric {earlystop.monitor} is not monitored. Change metric for EarlyStopping.')
            if earlystop.mode not in ['min', 'max']:
                raise ValueError(f'EarlyStopping mode can be either "min" or "max".')
                
            memory = dict()
            memory['best'] = np.Inf if earlystop.mode == 'min' else -np.Inf
            memory['best_epoch'] = 0 # track on each epoch
            if earlystop.mode == 'min': 
                min_delta = earlystop.min_delta*(-1)
                operator = np.less
            else:
                min_delta = earlystop.min_delta*1
                operator = np.greater
            memory['wait'] = 0
            
        with tqdm(total=nepochs, desc="epoch") as outerbar, tqdm(total=(len(data)//batch_size)-1, desc="batch") as innerbar:
            for epoch in range(nepochs):
                innerbar.reset()
                
                for batch in funcy.partition(batch_size, data):
                    batch = jax.numpy.asarray(batch)

                    loss_cur, res, vectorizer = _fit_jax_vmap(self, self.loss_fn, batch, strategy=self.strategy, optimizer=self.optimizer, epoch=epoch, embedding=embedding, learning_rate=self.learning_rate)
                    history['loss'].append(loss_cur)
                    # model.normalize()
                    
                    if callbacks:
                        for name, fn in callbacks:
                            history[name].append(fn(self, res, vectorizer))

                    innerbar.update()
                    innerbar.set_postfix(loss=history["loss"][-1])
                    
                if earlystop:
                    current = sum(history[earlystop.monitor][-num_batches:])/num_batches

                    if memory['wait'] == 0 and epoch == 0:
                        memory['best'] = current
                        memory['best_model'] = self
                        memory['best_epoch'] = epoch
                        #memory['wait'] += 1
                    if epoch > 0: memory['wait'] += 1
                    if operator(current - min_delta, memory['best']):
                        memory['best'] = current
                        memory['best_model'] = self
                        memory['best_epoch'] = epoch
                        memory['wait'] = 0
                    if memory['wait'] >= earlystop.patience and epoch > 0:
                        best_epoch = memory['best_epoch']
                        print(f'Training stopped by EarlyStopping on epoch: {best_epoch}', flush=True)
                        self = memory['best_model']
                        return history
                    print('Waiting for ' + str(memory['wait']) + ' epochs.', flush=True)
                outerbar.update()
        return history

    def predict(self, x):
        """Performs transformation on input data.

        Parameters
        ----------
        x : :class:`quimb.tensor.tensor_1d.MatrixProductState`, or :class:`quimb.tensor.tensor_core.TensorNetwork`
            Embedded data in MatrixProductState form.
        
        Returns
        -------
        :class:`quimb.tensor.tensor_1d.MatrixProductState`
            Matrix product state of result
        """

        return (self @ x)
    
    def predict_norm(self, x):

        """Computes norm for output of ``predict(x)``.

        Parameters
        ----------
        x : :class:`quimb.tensor.tensor_1d.MatrixProductState`, or :class:`quimb.tensor.tensor_core.TensorNetwork`
            Embedded data in MatrixProductState form.
        
        Returns
        -------
        float
            Norm of `predict(x)`
        """

        return self.predict(x).norm()

def load_model(dir_name, model_name):
    """Loads the Model from pickle file.

    Parameters
    ----------
    dir_name : str
        Directory where model is stored.
    model_name : str
        Name of model.
    """

    return qu.load_from_disk(f'{dir_name}/{model_name}.pkl')

class LossWrapper:
    """Wrapper of loss function to make it compatible with JAX.

    Attributes
    ----------
    tn : :class:`quimb.tensor.tensor_core.TensorNetwork`
        Tensor network on which training is performed.
    loss_fn: function
        Loss function.
    """

    def __init__(self, loss_fn, tn):
        """Constructor

        Attributes
        ----------
        tn : :class:`quimb.tensor.tensor_core.TensorNetwork``
            Tensor Network.
        loss_fn : function
            Loss function.
        """
        self.tn = tn
        self.loss_fn = loss_fn

    def __call__(self, tensor_arrays, **kwargs):
        """Wraps and executes loss function.

        Parameters
        ----------
        tensor_arrays : sequence of :class:`numpy.ndarray``
            
        Returns
        -------
        :class:`functools.partial`
        """
        tn = self.tn.copy()

        loss_fn = functools.partial(self.loss_fn, **kwargs)

        for tensor, array in zip(tn.tensors, tensor_arrays):
            tensor.modify(data=array)

        with qtn.contract_backend("jax"):
            return loss_fn(tn)


def _fit(
    model: Model,
    loss_fn: Callable,
    data: Collection,
    strategy: Strategy = Global(),
    optimizer: Optional[Callable] = None,
    epoch: Optional[int] = None,
    embedding: embeddings.Embedding = embeddings.trigonometric(),
    **hyperparams,
):
    """Perfoms training procedure with using JAX to compute gradients of loss function.

    Parameters
    ----------
    model : :class:`tnad.models.Model`
        Model for training.
    loss_fn : `Callable`
        Loss function.
    data : sequence` of :class:`numpy.ndarray`
        Data for training Model.
    strategy : :class:`tnad.strategy.Strategy`
        Strategy for computing gradients.
    optimizer : :class:`quimb.tensor.optimize.TNOptimizer`, or different possibilities of optimizers from :func:`quimb.tensor.optimize`,or `None`
        Optimizer.
    epoch : int
        Current epoch.
    embedding : :class:`tnad.embeddings.Embedding`
        Data embedding function.
    
    Returns
    -------
    float
        Value of loss function
    :class:`scipy.optimize.OptimizeResult`
        See :class:`scipy.optimize.OptimizeResult` for more information.
    :class:`quimb.tensor.optimize.Vectorizer`
        Vectorizer data of Tensor Network.
    """

    if not isinstance(strategy, Global):
        raise NotImplementedError("non-`Global` strategies are not implemented yet for function `_fit`")

    if optimizer is None:
        optimizer = qtn.optimize.SGD()

    for sites in strategy.iterate_sites(model):
        # contract sites in groups
        strategy.prehook(model, sites)

        vectorizer = qtn.optimize.Vectorizer(model.arrays)

        def jac(x):
            arrays = vectorizer.unpack(x)

            def foo(x, *model_arrays):
                tn = model.copy()
                for tensor, array in zip(tn.tensors, model_arrays):
                    tensor.modify(data=array)

                phi = tnad.embeddings.embed(x, embedding)
                return loss_fn(tn, phi)

            with autoray.backend_like("jax"), qtn.contract_backend("jax"):
                x = jax.vmap(jax.grad(foo, argnums=[i + 1 for i in range(model.L)]), in_axes=[0] + [None] * model.L)(jax.numpy.asarray(data), *arrays)
                x = [jax.numpy.sum(xi, axis=0) / data.shape[0] for xi in x]

            return np.concatenate(x, axis=None)

        # call quimb's optimizers with vectorizer
        def loss(x):
            arrays = vectorizer.unpack(x)

            def foo(sample, *model_arrays):
                tn = model.copy()
                for tensor, array in zip(tn.tensors, model_arrays):
                    tensor.modify(data=array)

                phi = tnad.embeddings.embed(sample, embedding)
                return loss_fn(model, phi)

            with autoray.backend_like("jax"), qtn.contract_backend("jax"):
                x = jax.vmap(foo, in_axes=[0] + [None] * model.L)(jax.numpy.asarray(data), *arrays)
                return jax.numpy.sum(x) / data.shape[0]

        # prepare hyperparameters
        hyperparams = {key: value(epoch) if callable(value) else value for key, value in hyperparams.items()}
        if "maxiter" not in hyperparams:
            hyperparams["maxiter"] = 1

        x = vectorizer.pack(model.arrays)
        res = optimizer(loss, x, jac, **hyperparams)

        opt_arrays = vectorizer.unpack(res.x)

        for tensor, array in zip(model.tensors, opt_arrays):
            tensor.modify(data=array)

        # split sites
        strategy.posthook(model, sites)

        return res.fun, res, vectorizer

def _fit_test(
    model: Model,
    loss_fn: Callable,
    data: Collection,
    strategy: Strategy = Global(),
    optimizer: Optional[Callable] = None,
    executor: Optional[Executor] = None,
    epoch: Optional[int] = None,
    callbacks: Optional[Sequence[Callable[[Model, OptimizeResult, qtn.optimize.Vectorizer], Any]]] = None,
    **hyperparams,
):

    """Performs training procedure. Currently in testing stage. Not used.
    """

    if not isinstance(strategy, Global):
        raise NotImplementedError("non-`Global` strategies are not implemented yet for function `_fit`")

    if optimizer is None:
        optimizer = qtn.optimize.SGD()

    if executor is None:
        executor = ProcessPoolExecutor()

    metrics = None
    if callbacks:
        metrics = []

    for sites in strategy.iterate_sites(model):
        # contract sites in groups
        strategy.prehook(model, sites)

        arrays = model.arrays
        vectorizer = qtn.optimize.Vectorizer(arrays)

        error_wrapper = LossWrapper(loss_fn, model)

        def jac(x):
            # compute grad of error term
            error_grad = jax.grad(error_wrapper)

            arrays = tuple(map(jax.numpy.asarray, vectorizer.unpack(x)))
            futures = executor.map(lambda sample: error_grad(arrays, data=sample), data)

            # tree fold for parallelization
            futures = list(futures)
            while len(futures) > 1:
                futures = [executor.submit(operator.add, *chunk) if len(chunk) > 1 else chunk[0] for chunk in funcy.chunks(2, futures)]

            # normalize gradients
            n = len(data)
            grad_arrays = tuple(array / n for array in futures[0].result())

            return vectorizer.pack(grad_arrays, name="grad")

        # call quimb's optimizers with vectorizer
        def loss(x):
            arrays = vectorizer.unpack(x)
            futures = executor.map(lambda sample: error_wrapper(arrays, data=sample), data)

            # tree fold for parallelization
            futures = list(futures)
            while len(futures) > 1:
                futures = [executor.submit(operator.add, *chunk) if len(chunk) > 1 else chunk[0] for chunk in funcy.chunks(2, futures)]

            return futures[0].result() / len(data)

        # prepare hyperparameters
        hyperparams = {key: value(epoch) if callable(value) else value for key, value in hyperparams.items()}
        if "maxiter" not in hyperparams:
            hyperparams["maxiter"] = 1

        x = vectorizer.pack(arrays)
        res = optimizer(loss, x, jac, **hyperparams)

        opt_arrays = vectorizer.unpack(res.x)

        for tensor, array in zip(model.tensors, opt_arrays):
            tensor.modify(data=array)

        # split sites
        strategy.posthook(model, sites)

    if callbacks:
        metrics = tuple(fn(model, res, vectorizer) for fn in callbacks)  # type: ignore
        return (res.fun, *metrics)  # type: ignore

    return res.fun

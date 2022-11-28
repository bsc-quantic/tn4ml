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
    
    """
    `Model` class models training model of the class Tensor Network
    
    Args:
        loss_fn: Loss function. See `tnad/loss.py` for examples. Default `None`.
        strategy: Strategy for computing gradients. See `Strategy` class in `strategy.py`. Default `strategy.Global()`
        optimizer: `quimb.tensor.optimize.TNOptimizer` instance or different possibilities of optimizers from `quimb.tensor.optimize`. 
    """
    
    def __init__(self):
        self.loss_fn = None # dict()
        self.strategy = Global()
        self.optimizer = qtn.optimize.ADAM()
    
    def save(self, model_name, dir_name='~'):
        """
        Saves the Model to pickle file.
        Args
            model_name: name of Model. `str`.
            dir_name: directory for saving Model. `str`.
        """
        qu.save_to_disk(self, f'{dir_name}/{model_name}.pkl')
    
    def configure(self, **kwargs):
        """
        Configures Model for training setting the arguments.
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
        
        """
        Performs the training procedure of the model
        
        Args:
            data: Data used for training procedure. `Sequence` of `numpy.ndarray`
            batch_size: Number of samples per gradient update. `Integer` or default `None` 
            nepochs: Number of epochs for training the Model. `Integer`.
            embedding: Data embedding function. `embeddings.Embedding` instance.
            callbacks: Metrics for monitoring training progress. `Sequence` of callbacks (metrics) - tuple(callback name, function - `Callable`) or default `None`. Each metric function receives (Model, return value of `scipy.optimize.OptimizeResult`, `quimb.tensor.optimize.Vectorizer`).
            earlystop: Early stopping training when monitored metric stopped improving. `EarlyStopping` instance.

        Return
            history: Records training loss and metric values. `dictionary`.
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

    def fit_step(self, loss_fn, niter=1, **kwargs):
        for sites in self.strategy.iterate_sites(self):
            # contract tensors (if needed)
            self.strategy.prehook(self, sites)

            if isinstance(loss_fn, dict):
                error = loss_fn["error"]
                data = kwargs["loss_constants"].pop("batch_data")

                loss_fn = [lambda model: error(model, sample) for sample in data]

                if "reg" in loss_fn:
                    loss_fn.append(loss_fn["reg"])

            target_site_tags = tuple(self.site_tag(site) for site in funcy.flatten(sites))
            opt = qtn.TNOptimizer(
                self,
                loss_fn=loss_fn,
                optimizer=self.optimizer,
                tags=target_site_tags,
                **kwargs,
            )

            if isinstance(self.optimizer, str):
                optself = opt.optimize(niter)
                self._tensors = optself.tensors
            else:
                x = opt.vectorizer.vector
                _, grads = opt.vectorized_value_and_grad(x)
                grads = opt.vectorizer.unpack(grads)

                tensors = self.select_tensors(target_site_tags, which="any")
                for tensor, grad in zip(tensors, grads):
                    tensor.modify(data=self.optimizer(tensor.data, grad))

            # split tensors (if needed) & renormalize (if configured)
            self.strategy.posthook(self, sites)

    def predict(self, x):
        """
        Performs action of Model on input data.
        Args
            x: Embedded data in MatrixProductState form.
        Return
            MatrixProductState of result
        """
        return (self @ x)
    
    def predict_norm(self, x):
        """
        Computed norm for 'action of Model on input data'.
        Args
            x: Embedded data in MatrixProductState form.
        Return
            norm of `predict(x)`
        """
        return self.predict(x).norm()

def load_model(dir_name, model_name):
    """
    Loads the Model from pickle file.
    Args
        dir_name: Directory where model is stored. `str`.
        model_name: Name of model. `str`.
    """
    return qu.load_from_disk(f'{dir_name}/{model_name}.pkl')

class LossWrapper:
    """
    Wrapper of loss function to make it compatible with JAX.
    
    Args:
        tn: Tensor network on which training is performed.
        loss_fn: Loss function.
    """
    def __init__(self, loss_fn, tn):
        self.tn = tn
        self.loss_fn = loss_fn

    def __call__(self, arrays, **kwargs):
        tn = self.tn.copy()

        kwargs = qtn.optimize.parse_constant_arg(kwargs, jax.numpy.asarray)
        loss_fn = functools.partial(self.loss_fn, **kwargs)

        for tensor, array in zip(tn.tensors, arrays):
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
    """
    Perfoms training procedure with using JAX to compute gradients of loss function.
    
    Args
        model: Model for training. `Model`.
        loss_fn: Loss function. `Callable`.
        data: Data for training Model. `Sequence` of `numpy.ndarray`.
        strategy: Strategy for computing gradients. `Strategy`.
        optimizer: `quimb.tensor.optimize.TNOptimizer` instance or different possibilities of optimizers from `quimb.tensor.optimize` or default `None`.
        epoch: Current epoch. `Integer`.
        embedding: Data embedding function. `embeddings.Embedding` instance.
    
    Return
        res.fun - Value of loss function. `float`.
        res - return value of `scipy.optimize.OptimizeResult`. See `scipy.optimize.OptimizeResult`
        vectorizer - Vectorizer data of Tensor Network. `quimb.tensor.optimize.Vectorizer` instance.
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
    """
    Performs training procedure. Currently in testing stage. Not used.
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

from .smpo import SpacedMatrixProductOperator
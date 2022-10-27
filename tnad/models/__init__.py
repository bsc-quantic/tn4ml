import functools
import operator
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any, Callable, Collection, Optional, Sequence

import autoray
import funcy
import jax
import numpy as np
import tnad.embeddings
from quimb import tensor as qtn
from scipy.optimize import OptimizeResult
from tnad.strategy import *
from tqdm import tqdm


class Model(qtn.TensorNetwork):
    def __init__(self):
        self.loss_fn = None
        self.strategy = Global()
        self.optimizer = qtn.optimize.ADAM()

    def configure(self, **kwargs):
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
            elif key in ["optimizer", "loss_fn"]:
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found")

    def train(self, data, batch_size=None, nepochs=1, callbacks=None, **kwargs):
        if self.loss_fn is None:
            raise ValueError("`loss_fn` not yet configured. Call `Model.configure(loss_fn=...)` first.")

        # regularization parameter
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]

        # split data in batches
        if batch_size:
            data = np.split(data, data.shape[0] // batch_size)
        else:
            data = [data]  # NOTE fixes `for batch in data`

        metrics = []

        for epoch in (pbar := tqdm(range(nepochs))):
            pbar.set_description(f"Epoch #{epoch}")
            for batch in data:
                metrics_cur = _fit(self, self.loss_fn, batch, strategy=self.strategy, optimizer=self.optimizer, epoch=epoch, callbacks=callbacks, **kwargs)
                metrics.append(metrics_cur)

        return metrics

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
        return (self @ x).norm()


class LossWrapper:
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
    executor: Optional[Executor] = None,
    epoch: Optional[int] = None,
    callbacks: Optional[Sequence[Callable[[Model, OptimizeResult, qtn.optimize.Vectorizer], Any]]] = None,
    **hyperparams,
):
    """
    ## Arguments
    - model: `Model`
    - loss_fn: `Callable`
    - data: `Sequence` of `numpy.ndarray`
    - reg_fn: `Callable`
    - strategy: `Strategy`
    - optimizer: `Callable` or `None`
    - executor: `concurrent.futures.Executor` or `None`
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


def _fit_jax_vmap(
    model: Model,
    loss_fn: Callable,
    data: Collection,
    strategy: Strategy = Global(),
    optimizer: Optional[Callable] = None,
    epoch: Optional[int] = None,
    callbacks: Optional[Sequence[Callable[[Model, OptimizeResult, qtn.optimize.Vectorizer], Any]]] = None,
    embedding: tnad.embeddings.Embedding = tnad.embeddings.trigonometric(),
    **hyperparams,
):
    """
    ## Arguments
    - model: `Model`
    - loss_fn: `Callable`
    - data: `Sequence` of `numpy.ndarray`
    - reg_fn: `Callable`
    - strategy: `Strategy`
    - optimizer: `Callable` or `None`
    """

    if not isinstance(strategy, Global):
        raise NotImplementedError("non-`Global` strategies are not implemented yet for function `_fit`")

    if optimizer is None:
        optimizer = qtn.optimize.SGD()

    metrics = None
    if callbacks:
        metrics = []

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

        if callbacks:
            metrics.append(tuple(fn(model, res, vectorizer)) for fn in callbacks)  # type: ignore
            return (res.fun, *metrics)  # type: ignore

        return res.fun


from .smpo import SpacedMatrixProductOperator

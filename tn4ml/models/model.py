from tqdm import tqdm
from typing import Any, Callable, Collection, Optional, Sequence, Tuple
import autoray
import funcy
import jax
import numpy as np
from quimb import tensor as qtn
import quimb as qu
from ..embeddings import Embedding, trigonometric, embed, physics_embedding
from ..util import EarlyStopping, ExponentialDecay, ExponentialGrowth
from ..strategy import *

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def choose_optimizer(optimizer):
    # helper function
    # adam, nadam, ADABELIEF, RMSPROP, SGD
    if isinstance(optimizer, qtn.optimize.SGD):
        return qtn.optimize.SGD()
    elif isinstance(optimizer, qtn.optimize.RMSPROP):
        return qtn.optimize.RMSPROP()
    elif isinstance(optimizer, qtn.optimize.ADAM):
        return qtn.optimize.ADAM()
    elif isinstance(optimizer, qtn.optimize.NADAM):
        return qtn.optimize.NADAM()
    
class Model(qtn.TensorNetwork):
    """:class:`tn4ml.models.Model` class models training model of class :class:`quimb.tensor.tensor_core.TensorNetwork`.

    Attributes
    ----------
    loss_fn : `Callable`, or `None`
        Loss function. See :mod:`tn4ml.loss` for examples.
    strategy : :class:`tn4ml.strategy.Strategy`
        Strategy for computing gradients.
    optimizer : :class:`quimb.tensor.optimize.TNOptimizer`, or different possibilities of optimizers from :func:`quimb.tensor.optimize`.
    """

    def __init__(self):
        """Constructor
        """
        self.loss_fn = None
        self.strategy = Global()
        self.optimizer = qtn.optimize.ADAM()

    def save(self, model_name, dir_name='~'):

        """Saves :class:`tn4ml.models.Model` to pickle file.

        Parameters
        ----------
        model_name : str
            Name of Model.
        dir_name: str
            Directory for saving Model.
        """

        qu.save_to_disk(self, f'{dir_name}/{model_name}.pkl')
    
    def set_model(self, model):
        self.model = model
    
    def set_smpo(self, smpo):
        self.smpo = smpo
        
    def return_mps_sample(self, sample):
        phi = physics_embedding(sample, trigonometric())
        mps = self.smpo.apply(phi)
        return mps
    

    def configure(self, **kwargs):

        """Configures :class:`tn4ml.models.Model` for training setting the arguments.
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
            inputs: Collection,
            targets: Optional[Collection] = None,
            batch_size: Optional[int] = None,
            nepochs: Optional[int] = 1,
            embedding: Embedding = trigonometric(),
            callbacks: Optional[Sequence[Tuple[str, Callable]]] = None,
            normalize: Optional[bool] = False,
            earlystop: Optional[EarlyStopping] = None,
            exp_decay: Optional[ExponentialDecay] = None,
            exp_growth: Optional[ExponentialGrowth] = None,
            **kwargs):

        """Performs the training procedure of :class:`tn4ml.models.Model`.

        Parameters
        ----------
        inputs : sequence of :class:`numpy.ndarray`
            Data used for training procedure.
        targets: sequence of :class:`numpy.ndarray`
            Targets for training procedure (if training is supervised).
        batch_size : int, or default `None`
            Number of samples per gradient update.
        nepochs : int
            Number of epochs for training the Model.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        callbacks : sequence of callbacks (metrics) - ``tuple(callback name, `Callable`)``, or default `None`
            List of metrics for monitoring training progress. Each metric function receives (:class:`tn4ml.models.Model`, :class:`scipy.optimize.OptimizeResult`, :class:`quimb.tensor.optimize.Vectorizer`).
        normalize : bool
            If True, the model is normalized after each iteration.
        earlystop : :class:`tn4ml.util.EarlyStopping`
            Early stopping training when monitored metric stopped improving.
        exp_decay : `ExponentialDecay` instance
            Exponential decay of the learning rate.
        exp_growth : `ExponentialGrowth` instance
            Exponential growth of the learning rate.

        Returns
        -------
        history: dict
            Records training loss and metric values.
        """

        num_batches = (len(inputs)//batch_size)

        history = dict()
        history['loss'] = []
        if callbacks:
            for name, _ in callbacks:
                history[name] = []

        if earlystop:
            earlystop.on_begin_train(history)
         
        if self.optimizer is None:
            self.optimizer = qtn.optimize.SGD()
            
        if isinstance(self.strategy, Sweeps):
            iterate = self
            self.optimizers = []
            for sites in self.strategy.iterate_sites(iterate):
                self.optimizers.append(choose_optimizer(self.optimizer))

        with tqdm(total=nepochs, desc="epoch") as outerbar, tqdm(total=(len(inputs)//batch_size)-1, desc="batch") as innerbar:
            for epoch in range(nepochs):
                    
                if exp_decay and epoch >= exp_decay.start_decay:
                    self.learning_rate = exp_decay(epoch)
                if exp_growth and epoch >= exp_growth.start_step:
                    self.learning_rate = exp_growth(epoch)
                
                # supervised learning
                if targets is not None:
                    if targets.ndim == 1:
                        targets = np.expand_dims(targets, axis=1)
                    data = np.concatenate([inputs, targets], axis=1)
                else: data = inputs

                loss_batch = 0
                for batch in funcy.partition(batch_size, data):
                    #data = shuffle_along_axis(data, axis=1)
                    batch = jax.numpy.asarray(batch)
                    if isinstance(self.strategy, Sweeps):
                        loss_cur, res, vectorizer = _fit_sweeps(self, self.loss_fn, batch, strategy=self.strategy, epoch=epoch, embedding=embedding, learning_rate=self.learning_rate)
                    else:
                        loss_cur, res, vectorizer = _fit(self, self.loss_fn, batch, strategy=self.strategy, epoch=epoch, embedding=embedding, learning_rate=self.learning_rate)
                        
                    loss_batch += loss_cur
                                        
                    if normalize:
                        self.canonize(0)
                        self.normalize()

                    if callbacks:
                        for name, fn in callbacks:
                            history[name].append(fn(self, res, vectorizer))
                
                history['loss'].append(loss_batch/num_batches)
                
                outerbar.update()
                outerbar.set_postfix({'loss': loss_batch/num_batches})

                if earlystop:
                    if earlystop.monitor == 'loss':
                        current = loss_batch/num_batches
                    else:
                        current = sum(history[earlystop.monitor][-num_batches:])/num_batches
                    return_value = earlystop.on_end_epoch(current, epoch)
                    if return_value==0: continue
                    else: return history

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


def _fit(
    model: Model,
    loss_fn: Callable,
    data: Collection,
    strategy: Strategy = Global(),
    epoch: Optional[int] = None,
    embedding: Embedding = trigonometric(),
    **hyperparams,
):
    """Perfoms training procedure with using JAX to compute gradients of loss function.

    Parameters
    ----------
    model : :class:`tn4ml.models.Model`
        Model for training.
    loss_fn : `Callable`
        Loss function.
    data : sequence` of :class:`numpy.ndarray`
        Data for training Model. Can contain targets (if training is supervised).
    strategy : :class:`tn4ml.strategy.Strategy`
        Strategy for computing gradients.
    epoch : int
        Current epoch.
    embedding : :class:`tn4ml.embeddings.Embedding`
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
        raise NotImplementedError("This is _fit method for Global optimization strategy.")
    
    L = model.L

    for sites in strategy.iterate_sites(model):
        # contract sites in groups
        strategy.prehook(model, sites)
        
        vectorizer = qtn.optimize.Vectorizer(model.arrays)

        def jac(x):
            # x = model
            arrays = vectorizer.unpack(x)

            def foo(sample, *model_arrays):
                #unpack
                # sample = data input
                tn = model.copy()
                for tensor, array in zip(tn.tensors, model_arrays):
                    tensor.modify(data=array)
                    
                if sample.shape[0] > L:
                    sample, target = sample[:L], sample[L:]
                    phi = embed(sample, embedding)
                    return loss_fn(tn, phi, target) # if training is supervised
                else:
                    phi = embed(sample, embedding)
                    return loss_fn(tn, phi)

            with autoray.backend_like("jax"), qtn.contract_backend("jax"):
                x = jax.vmap(jax.grad(foo, argnums=[i + 1 for i in range(L)]), in_axes=[0] + [None] * L)(jax.numpy.asarray(data), *arrays)
                x = [jax.numpy.sum(xi, axis=0) / data.shape[0] for xi in x]
            return np.concatenate(x, axis=None)

        # call quimb's optimizers with vectorizer
        def loss(x):
            # x = model
            arrays = vectorizer.unpack(x)

            def foo(sample, *model_arrays):
                # sample = data input
                tn = model.copy()
                for tensor, array in zip(tn.tensors, model_arrays):
                    tensor.modify(data=array)
                
                if sample.shape[0] > L:
                    sample, target = sample[:L], sample[L:]
                    phi = embed(sample, embedding)
                    return loss_fn(tn, phi, target) # if training is supervised
                else:
                    phi = embed(sample, embedding)
                    return loss_fn(tn, phi)

            with autoray.backend_like("jax"), qtn.contract_backend("jax"):
                x = jax.vmap(foo, in_axes=[0] + [None] * L)(jax.numpy.asarray(data), *arrays)
                return jax.numpy.sum(x) / data.shape[0]
        
        # prepare hyperparameters
        hyperparams = {key: value(epoch) if callable(value) else value for key, value in hyperparams.items()}
        if "maxiter" not in hyperparams:
            hyperparams["maxiter"] = 1

        x = vectorizer.pack(model.arrays)
        res = model.optimizer(loss, x, jac, **hyperparams)

        opt_arrays = vectorizer.unpack(res.x)
        
        for tensor, array in zip(model.tensors, opt_arrays):
            tensor.modify(data=array)
            
        # split sites
        strategy.posthook(model, sites)
        
    return res.fun, res, vectorizer

def _fit_sweeps(
    model: Model,
    loss_fn: Callable,
    data: Collection,
    strategy: Strategy = Sweeps(),
    epoch: Optional[int] = None,
    embedding: Embedding = trigonometric(),
    **hyperparams,
):
    """Perfoms training procedure with using JAX to compute gradients of loss function for having input MPS dataset.

    Parameters
    ----------
    model : :class:`tn4ml.models.Model`
        Model for training.
    loss_fn : `Callable`
        Loss function.
    data : sequence` of :class:`numpy.ndarray`
        Data for training Model. Can contain targets (if training is supervised).
    strategy : :class:`tn4ml.strategy.Strategy`
        Strategy for computing gradients.
    epoch : int
        Current epoch.
    embedding : :class:`tn4ml.embeddings.Embedding`
        Data embedding function.

    Returns
    -------
    float
        Value of loss function
    :class:`scipy.optimize.OptimizeResult`
        See :class:`scipy.optimize.OptimizeResult` for more information.
    :class:`quimb.tensor.optimize.Vectorizer`
        Vectorizer data of Tensor Network.
    :class:`Model`
        Model which is training
    """

    if not isinstance(strategy, Sweeps):
        raise NotImplementedError("Only for `Sweeps` strategy")
    
    if strategy.grouping > 2:
        raise NotImplementedError("Only implemented for grouping <= 2")
    
    L = model.L

    s = 0
    for sites in strategy.iterate_sites(model):
        # contract sites in groups
        strategy.prehook(model, sites)

        if strategy.grouping == 1:
            sites = [sites]
        sitetags = [model.site_tag(site) for site in sites]
        
        tensor = model.select_tensors(sitetags)[0]
        vectorizer = qtn.optimize.Vectorizer(tensor.data)

        def jac(x):
            target_array = vectorizer.unpack(x)
            
            def foo(sample, x):
                #unpack
                # sample = data input
                tn = model.copy()
                tn.select_tensors(sitetags)[0].modify(data=x)
                
                # TODO IMPLEMENT FOR SUPERVISED
                if model.smpo:
                    # if using SMPO for dimensionality reduction
                    phi = physics_embedding(sample, trigonometric())
                    mps = model.smpo.apply(phi)
                    return loss_fn(tn, mps)
                else:
                    phi = embed(sample, embedding)
                    return loss_fn(tn, phi)
            
            with autoray.backend_like("jax"), qtn.contract_backend("jax"):
                x = jax.vmap(jax.grad(foo, argnums=[1]), in_axes=[0,None])(jax.numpy.asarray(data), target_array)
                x = [jax.numpy.sum(xi, axis=0) / data.shape[0] for xi in x]

            return np.concatenate(x, axis=None)

        # call quimb's optimizers with vectorizer
        def loss(x):
            # x = model
            target_array = vectorizer.unpack(x)

            def foo(sample, x):
                # sample = data input
                tn = model.copy()
                tn.select_tensors(sitetags)[0].modify(data=x)
                
                # TODO IMPLEMENT FOR SUPERVISED
                if model.smpo:
                    # if using SMPO for dimensionality reduction
                    phi = physics_embedding(sample, trigonometric())
                    mps = model.smpo.apply(phi)
                    return loss_fn(tn, mps)
                else:
                    phi = embed(sample, embedding)
                    return loss_fn(tn, phi)

            with autoray.backend_like("jax"), qtn.contract_backend("jax"):
                x = jax.vmap(foo, in_axes=[0, None])(jax.numpy.asarray(data), target_array)
                return jax.numpy.sum(x) / data.shape[0]

        # prepare hyperparameters
        hyperparams = {key: value(epoch) if callable(value) else value for key, value in hyperparams.items()}
        if "maxiter" not in hyperparams:
            hyperparams["maxiter"] = 1
        
        tensor = model.select_tensors(sitetags)[0]
        x = vectorizer.pack(tensor.data)
        res = model.optimizers[s](loss, x, jac, **hyperparams)
        
        opt_array = vectorizer.unpack(res.x) #len 1
        
        tensor = model.select_tensors(sitetags)[0]
        tensor.modify(data = opt_array)
            
        # split sites
        strategy.posthook(model, sites)
        # counting how many combinations of sites
        s+=1

    return res.fun, res, vectorizer
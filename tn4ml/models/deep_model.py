import functools
import operator
from tqdm import tqdm
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any, Callable, Collection, Optional, Sequence, NamedTuple
import autoray
import funcy
import jax
import numpy as np
from scipy.optimize import OptimizeResult
# from .smpo import SpacedMatrixProductOperator

import quimb.tensor as qtn
import quimb as qu
import autoray as a
from quimb.tensor import TensorNetwork
from .model import Model, LossWrapper
from .smpo import SpacedMatrixProductOperator
from ..embeddings import Embedding, trigonometric, embed
from ..util import EarlyStopping, ExponentialDecay, ExponentialGrowth
from ..strategy import *

class DeepTensorNetwork(TensorNetwork):

    def __init__(self, nfeatures_start, n_mpo, spacing, bond_dim, seed=123, init_func='normal', ind_array = ["k{}", "i{}", "j{}", "l{}", "m{}"]):
        self.nfeatures_start = nfeatures_start
        self.n_mpo = n_mpo
        self.spacing = spacing
        self.bond_dim = bond_dim
        self.seed = seed
        self.init_func = init_func
        self.loss_fn = None
        self.strategy = Global()
        self.optimizer = qtn.optimize.ADAM()

        model = SpacedMatrixProductOperator.rand_orthogonal(self.nfeatures_start, spacing=self.spacing, init_func=self.init_func, bond_dim=self.bond_dim, seed=self.seed, upper_ind_id=ind_array[0], lower_ind_id=ind_array[1])
        num_outputs = len(list(model.lower_inds))
        output_inds = list(model.lower_inds)

        # atm naive implementation
        for i in range(1, n_mpo):
            next_model = SpacedMatrixProductOperator.rand_orthogonal(num_outputs, spacing=self.spacing, init_func=self.init_func, bond_dim=self.bond_dim, seed=self.seed, upper_ind_id=ind_array[i], lower_ind_id=ind_array[i+1])
            input_inds = list(next_model.upper_inds)
            num_outputs = len(list(next_model.lower_inds))
            dict_to_rename=dict()
            for ind in range(len(input_inds)):
                dict_to_rename[input_inds[ind]] = output_inds[ind]
            next_model.reindex(dict_to_rename, inplace=True)
            output_inds = list(next_model.lower_inds)
            
            model = model&next_model
        
        self.model = model
        TensorNetwork.__init__(self, model.tensors, virtual=True)
    
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
    
    def normalize(self):
        """Function for normalizing tensors of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

        Parameters
        ----------
        insert : int
            Index of tensor divided by norm. *Default = None*. When `None` the norm division is distributed across all tensors.
        """
        norm = self.norm()
        n_tensors = len(self.tensors)
        for tensor in self.tensors:
            tensor.modify(data=tensor.data / a.do("power", norm, 1 / n_tensors))
    
    def norm(self, **contract_opts):
        """Calculates norm of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`.

        Parameters
        ----------
        contract_opts : Optional
            Arguments passed to ``contract()``.

        Returns
        -------
        float
            Norm of :class:`tn4ml.models.smpo.SpacedMatrixProductOperator`
        """
        norm = self.model.H & self.model
        return norm.contract(**contract_opts) ** 0.5
    
    def train(self,
            inputs: Collection,
            targets: Optional[Collection] = None,
            batch_size: Optional[int] = None,
            nepochs: Optional[int] = 1,
            embedding: Embedding = trigonometric(),
            callbacks: Optional[Sequence[tuple[str, Callable]]] = None,
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

        with tqdm(total=nepochs, desc="epoch") as outerbar, tqdm(total=(len(inputs)//batch_size)-1, desc="batch") as innerbar:
            for epoch in range(nepochs):
                # innerbar.reset()
                if exp_decay and epoch >= exp_decay.start_decay:
                    self.learning_rate = exp_decay(epoch)
                if exp_growth and epoch >= exp_growth.start_step:
                    self.learning_rate = exp_growth(epoch)

                if targets is not None:
                    if targets.ndim == 1: 
                        targets = np.expand_dims(targets, axis=1)
                    data = np.concatenate([inputs, targets], axis=1)
                else: data = inputs

                loss_batch = 0
                for batch in funcy.partition(batch_size, data):
                    batch = jax.numpy.asarray(batch)

                    # TODO - implement _fit (for this TN architecture)
                    loss_cur, res, vectorizer = _fit(self, self.loss_fn, batch, strategy=self.strategy, optimizer=self.optimizer, epoch=epoch, embedding=embedding, learning_rate=self.learning_rate)
                    loss_batch += loss_cur

                    if normalize:
                        self.normalize()

                    if callbacks:
                        for name, fn in callbacks:
                            history[name].append(fn(self, res, vectorizer))

                    #innerbar.update()
                    #innerbar.set_postfix({'loss': loss_batch/(batch_num+1)})
                
                history['loss'].append(loss_batch/num_batches)
                outerbar.update()
                # innerbar.reset()
                outerbar.set_postfix({'loss': loss_batch/num_batches})

                if earlystop:
                    if earlystop.monitor == 'loss':
                        current = loss_batch/num_batches
                    else:
                        current = sum(history[earlystop.monitor][-num_batches:])/num_batches
                    return_value = earlystop.on_end_epoch(current, epoch)
                    if return_value==0: continue
                    else: return history
                #print(f'Current loss: {loss_batch/num_batches}')

        return history

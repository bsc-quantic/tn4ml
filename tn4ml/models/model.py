from typing import Any, Collection, Optional, Sequence, Tuple, Callable
from tqdm import tqdm
import funcy
import math
from time import time
import numpy as np

import quimb.tensor as qtn
import quimb as qu
import autoray
import optax
import jax

from ..embeddings import *
from ..strategy import *
from ..util import gradient_clip, EarlyStopping, TrainingType

class Model(qtn.TensorNetwork):
    """:class:`tn4ml.models.Model` class models training model of class :class:`quimb.tensor.tensor_core.TensorNetwork`.

    Attributes
    ----------
    loss : `Callable`, or `None`
        Loss function. See :mod:`tn4ml.metrics` for examples.
    strategy : :class:`tn4ml.strategy.Strategy`
        Strategy for computing gradients.
    optimizer : str
        Type of optimizer matching names of optimizers from optax.
    learning_rate : float
        Learning rate for optimizer.
    train_type : int
        Type of training: 0 = 'unsupervised' or 1 ='supervised', 2 = 'target TN'.
    gradient_transforms : sequence
        Sequence of gradient transformations.
    opt_state : Any
        State of optimizer.
    """

    def __init__(self):
        """ Constructor method for :class:`tn4ml.models.Model` class."""
        self.loss: Callable = None
        self.strategy : Any = 'global'
        self.optimizer : optax.GradientTransformation = optax.adam
        self.learning_rate : float = 1e-2
        self.train_type : int = TrainingType.UNSUPERVISED
        self.gradient_transforms : Sequence = None
        self.opt_state : Any = None
        self.device: str = 'cpu'

    def save(self, model_name: str, dir_name: str = '~', tn: bool = False):
        """ Saves :class:`tn4ml.models.Model` to pickle file.

        Parameters
        ----------
        model_name : str
            Name of Model.
        dir_name: str
            Directory for saving Model.
        tn : bool
            If True, model object is TensorNetwork.
        """
        exec(compile('from ' + self.__class__.__module__ + ' import ' + self.__class__.__name__, '<string>', 'single'))
        arrays = tuple(map(lambda x: np.array(jax.device_get(x)), self.arrays))
        if tn:
            tensors = []
            for i, array in enumerate(arrays):
                tensors.append(qtn.Tensor(array, inds=self.tensors[i].inds, tags=self._site_tag_id.format(i)))
            model = type(self)(tensors)
        else:
            model = type(self)(arrays)
        qu.save_to_disk(model, f'{dir_name}/{model_name}.pkl')
    
    def nparams(self) -> int:
        """ Returns number of parameters of the model.
        
        Returns
        -------
        int
        """
        return sum([np.prod(tensor.data.shape) for tensor in self.tensors])

    def configure(self, **kwargs):

        """ Configures model for training with specific parameters.

        Parameters
        ----------
        kwargs : dict
            Configuration parameters.

            Including:
            - strategy: str or Strategy object
            Training strategy ('global', 'sweeps', 'local', 'dmrg', 'dmrg-like')
            - optimizer: callable or optax optimizer
                Optimization algorithm to use
            - loss: callable
                Loss function for training
            - train_type: int
                Type of training (from :class:`tn4ml.util.TrainingType`)
            - learning_rate: float
                Learning rate for optimizer
            - gradient_transforms: sequence
                Sequence of gradient transformations for optax
            - device: str
                Computation device ('cpu' or 'gpu')
        
        Examples
        --------
        >>> model.configure(strategy='global', optimizer=optax.adam, learning_rate=0.01, loss=tn4ml.metrics.LogQuadNorm, train_type=TrainingType.UNSUPERVISED)
        """

        for key, value in kwargs.items():
            if key == "strategy":
                if isinstance(value, Strategy):
                    self.strategy = value
                elif value in ['sweeps', 'local', 'dmrg', 'dmrg-like']:
                    self.strategy = Sweeps()
                elif value in ['global', 'sgd', 'gd', 'gradient_descent']:
                    self.strategy = 'global'
                else:
                    raise ValueError(f'Strategy "{value}" not found')
            elif key in ["optimizer", "loss", "train_type", "learning_rate", "gradient_transforms", "device"]:
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found")

        if self.train_type not in [TrainingType.UNSUPERVISED, TrainingType.SUPERVISED, TrainingType.TARGET_TN]:
            raise AttributeError(f"Specify type of training: {TrainingType.UNSUPERVISED.name}, {TrainingType.SUPERVISED.name}, or {TrainingType.TARGET_TN.name}!") 
               
        if not hasattr(self, 'optimizer') or not hasattr(self, 'gradient_transforms'):
            raise AttributeError("Provide 'optimizer' or sequence of 'gradient_transforms'! ")
        
        if self.gradient_transforms:
            self.optimizer = optax.chain(*self.gradient_transforms)
        else:
            if hasattr(self, 'optimizer') and callable(self.optimizer):
                self.optimizer = self.optimizer(learning_rate = self.learning_rate)
            else:
                self.optimizer = optax.adam(learning_rate=self.learning_rate)
        
        if self.device not in ['cpu', 'gpu']:
            raise AttributeError("Device must be 'cpu' or 'gpu'!")

    def predict(self, sample: np.ndarray, embedding: Embedding = TrigonometricEmbedding(), return_tn: bool = False, normalize: bool = False) -> Union[np.ndarray, qtn.TensorNetwork]:
        """ Predicts the output of the model.
        
        Parameters
        ----------
        sample : :class:`numpy.ndarray`
            Input data.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        return_tn : bool   
            If True, returns tensor network, otherwise returns data. Useful when you want to vmap over predict function.
        
        Returns
        -------
        :class:`quimb.tensor.tensor_core.TensorNetwork`
            Output of the model.
        """

        if len(sample.flatten()) < self.L:
            raise ValueError(f"Input data must have at least {self.L} elements!")

        tn_sample = embed(sample, embedding)
        
        if callable(getattr(self, "apply", None)):
            output = self.apply(tn_sample)
        else:
            output = self & tn_sample
        
        if return_tn:
            return output
        else:
            output = output.contract(all, optimize='auto-hq')
            y_pred = output.squeeze().data
            if normalize:
                y_pred = y_pred/jnp.linalg.norm(y_pred)
            return y_pred
        
    def forward(self, data: jnp.ndarray, embedding: Embedding = TrigonometricEmbedding(), batch_size: int=64, normalize: bool = False, dtype: Any = jnp.float_, seed: int = 42, alternate_flip: bool = False) -> Collection:
        """ Forward pass of the model.

        Parameters
        ----------
        data : :class:`jax.numpy.ndarray`
            Input data.
        y_true: :class:`jax.numpy.ndarray`
            Target class vector.
        embedding: :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        batch_size: int
            Batch size for data processing.
        normalize: bool
            If True, the model output is normalized.
        dtype: Any
            Data type of input data.
        seed: int
            Random seed for data shuffling.
        
        Returns
        -------
        :class:`jax.numpy.ndarray`
            Output of the model.
        """
        outputs = []
        for batch_data in _batch_iterator(data, batch_size=batch_size, shuffle=False, dtype=dtype, seed=seed, alternate_flip=alternate_flip):
            x = jnp.array(batch_data, dtype=jnp.float64)
            
            output = jnp.squeeze(jnp.array(jax.vmap(self.predict, in_axes=(0, None, None, None))(x, embedding, False, normalize)))
            outputs.append(output)
        
        return jnp.concatenate(outputs, axis=0)
    
    def accuracy(self, data: jnp.ndarray, y_true: jnp.array = None, embedding: Embedding = TrigonometricEmbedding(), batch_size: int=64, shuffle: bool = False, normalize: bool = False, dtype:Any = jnp.float_, seed: int = 42, alternate_flip: bool = False) -> Number:
        """ Calculates accuracy for supervised learning.
        
        Parameters
        ----------
        model : :class:`tn4ml.models.Model`
            Tensor Network model.
        data: :class:`numpy.ndarray`
            Input data.
        y_true: :class:`numpy.ndarray`
            Target class vector.
        embedding: :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        batch_size: int
            Batch size for data processing.
        normalize: bool
            If True, the model output is normalized in predict function.
        dtype: Any
            Data type of input data.
        seed: int
            Random seed for data shuffling.
        
        Returns
        -------
        float
        """

        if y_true is None:
            raise ValueError("For unsupervised learning you must provide target data!")
        
        correct_predictions = 0
        num_samples = 0
        for batch_data in _batch_iterator(data, y_true, batch_size=batch_size, shuffle=shuffle, dtype=dtype, seed=seed, alternate_flip=alternate_flip):
            x, y = batch_data
            x, y = jnp.array(x, dtype=dtype), jnp.array(y)

            y_pred = jnp.squeeze(jnp.array(jax.vmap(self.predict, in_axes=(0, None, None, None))(x, embedding, False, normalize)))
            predicted = jnp.argmax(y_pred, axis=-1)
            true = jnp.argmax(y, axis=-1)

            correct_predictions += jnp.sum(predicted == true).item()
            num_samples += y_pred.shape[0]

        accuracy = correct_predictions / num_samples
        return accuracy

    def update_tensors(self, params):
        """ Updates tensors of the model with new parameters.
        
        Parameters
        ----------
        params : sequence of :class:`jax.numpy.ndarray`
            New parameters of the model.
        sitetags : sequence of str, or default `None`
            Names of tensors for differentiation (for Sweeping strategy).
        
        Returns
        -------
        None
        """
        if isinstance(self.strategy, Sweeps):
            if self.sitetags is None:
                raise ValueError("For Sweeping strategy you must provide names of tensors for differentiation.")
            tensor = self.select_tensors(self.sitetags)[0]
            tensor.modify(data = params[0])
        else:
            for tensor, array in zip(self.tensors, params):
                tensor.modify(data=array)
    
    def compute_entropy(self, data, embedding):
        """ Computes entropy of the model.

        Parameters
        ----------
        data : :class:`jax.numpy.ndarray`
            Input data.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.

        Returns
        -------
        float
            Entropy of the model.
        """
        data_embeded = embed(jnp.array(data), embedding)
        mps = self.apply(data_embeded)
        e = mps.entropy(len(mps.tensors)//2)
        return e

    def compute_entropy_batch(self, data, embedding):
        """ Computes entropy of the model for a batch of data.

        Parameters
        ----------
        data : :class:`jax.numpy.ndarray`
            Input data.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        
        Returns
        -------
        float
            Entropy of the model.
        """
        data = jnp.array(data)
        entropy = self.compute_entropy(data[0], embedding)
        return entropy
        
    def create_train_step(self, params, loss_func):

        """ Creates function for calculating value and gradients of loss, and function for one step in training procedure.
        Initializes the optimizer and creates optimizer state.

        Parameters
        ----------
        params : sequence of :class:`jax.numpy.ndarray`
            Parameters of the model.
        loss_func : function
            Loss function.
        grads_func : function
            Function for calculating gradients of loss.
        
        Returns
        -------
        train_step : function
            Function to perform one training step.
        opt_state : tuple
            State of optimizer at the initialization. 
        """
        init_params = {
            i: jnp.array(data)
            for i, data in enumerate(params)
        }
        opt_state = self.optimizer.init(init_params)

        def value_and_grad(params, data=None, targets=None):
            """ Calculates loss value and gradient. """

            loss_scalar_fn = lambda data, targets, *params: loss_func(data, targets, *params)
            loss, grads = jax.value_and_grad(loss_scalar_fn, argnums=range(2, 2 + len(params)))(data, targets, *params)
            return loss, grads

        def train_step(params, opt_state, data=None, grad_clip_threshold=None):
            """ Performs one training step.

            Parameters
            ----------
            params : sequence of :class:`jax.numpy.ndarray`
                Parameters of the model.
            opt_state : tuple
                State of optimizer.
            data : sequence of :class:`jax.numpy.ndarray`
                Input data.
            sitetags : sequence of str
                Names of tensors for differentiation (for Sweeping strategy).
            
            Returns
            -------
            float, :class:`jax.numpy.ndarray`
            """

            if data is not None:
                if type(data) == tuple and len(data) == 2:
                    data, targets = data
                    data, targets = jnp.array(data), jnp.array(targets)
                else:
                    data = jnp.array(data)
                    targets = None
            else:
                targets = None
            
            loss, grads = jax.jit(value_and_grad, backend=self.device)(params, data, targets)

            if grad_clip_threshold:
                grads = gradient_clip(grads, grad_clip_threshold)
            
            # convert to pytree structure
            grads = {i: jnp.array(data)
                    for i, data in enumerate(grads)}
            params = {i: jnp.array(data)
                    for i, data in enumerate(params)}
            
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # convert back to arrays
            params = tuple(jnp.array(v) for v in params.values())

            # update TN inplace
            self.update_tensors(params)
            
            # for numerical stability
            #self.normalize()

            return params, opt_state, loss
        
        return train_step, opt_state
    
    def train(self,
            inputs: Collection = None,
            val_inputs: Optional[Collection] = None,
            targets: Optional[Collection] = None,
            val_targets: Optional[Collection] = None,
            tn_target: Optional[qtn.TensorNetwork] = None,
            batch_size: Optional[int] = None,
            epochs: Optional[int] = 1,
            embedding: Embedding = TrigonometricEmbedding(),
            normalize: Optional[bool] = False,
            canonize: Optional[Tuple] = tuple([False, None]),
            time_limit: Optional[int] = None,
            earlystop: Optional[EarlyStopping] = None,
            # callbacks: Optional[Sequence[Tuple[str, Callable]]] = None,
            gradient_clip_threshold: Optional[float] = None,
            val_batch_size: Optional[int] = None,
            eval_metric: Optional[Callable] = None,
            display_val_acc: Optional[bool] = False,
            dtype: Any = jnp.float_,
            shuffle: Optional[bool] = False,
            seed: Optional[int] = 42,
            alternate_flip: bool = False):
        
        """Performs the training procedure of :class:`tn4ml.models.Model`.

        Parameters
        ----------
        inputs : sequence of :class:`numpy.ndarray`
            Data used for training procedure.
        val_inputs : sequence of :class:`numpy.ndarray`
            Data used for validation.
        targets: sequence of :class:`numpy.ndarray`
            Targets for training procedure (if training is supervised).
        val_targets: sequence of :class:`numpy.ndarray`
            Targets for validation (if training is supervised).
        tn_target: :class:`quimb.tensor.tensor_core.TensorNetwork` or any specialized TN class from `quimb.tensor` module
            Target tensor network for training.
        batch_size : int, or default `None`
            Number of samples per gradient update.
        epochs : int
            Number of epochs for training.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        normalize : bool
            If True, the model is normalized after each iteration.
        canonize: tuple([bool, int])
            tuple indicating is model canonized after each iteration. Example: (True, 0) - model is canonized in canonization center = 0.
        time_limit: int
            Time limit on model's training in seconds.
        earlystop : :class:`tn4ml.util.EarlyStopping`
            Early stopping training when monitored metric stopped improving.
        gradient_clip_threshold : float
            Threshold for gradient clipping.
        val_batch_size : int
            Number of samples per validation batch.
        display_val_acc : bool
            If True, displays validation accuracy.
        alternate_flip : bool
            If True, flips every other batch along axis=1.
            
        Returns
        -------
        history: dict
            Records training loss and metric values.
        """

        num_batches = (len(inputs)//batch_size)
        
        if targets is not None:
            if targets.ndim == 1:
                targets = np.expand_dims(targets, axis=-1)
        
        if val_inputs is not None and eval_metric is None:
            eval_metric = self.loss

        self.batch_size = batch_size

        if inputs is not None:
            n_batches = (len(inputs)//self.batch_size)

        if not hasattr(self, 'history'):
            self.history = dict()
            self.history['loss'] = []
            self.history['epoch_time'] = []
            self.history['unfinished'] = False
            if val_inputs is not None:
                if val_batch_size is None:
                    raise ValueError("Validation batch size must be provided!")
                self.history['val_loss'] = []
                if display_val_acc:
                    self.history['val_acc'] = []
        
        if earlystop:
            return_value = 0
            earlystop.on_begin_train(self.history, self)
        
        self.sitetags = None # for sweeping strategy
        
        def loss_fn(data=None, targets=None, *params):
            """
            Batches embedding + loss computation internally, with model params fixed externally.
            """
            tn = self.copy()

            if hasattr(self, 'sitetags') and self.sitetags is not None:
                tn.select_tensors(self.sitetags)[0].modify(data = params[0])
            else:
                for tensor, array in zip(tn.tensors, params):
                    tensor.modify(data=array)

            # Define batched version of embed + loss logic
            def single_loss(x, y = None):
                tn_i = embed(x, embedding) # create TN from data

                if self.train_type == TrainingType.UNSUPERVISED:
                    return self.loss(tn, tn_i)
                elif self.train_type == TrainingType.SUPERVISED:
                    return self.loss(tn, tn_i, y)
                else:
                    assert self.train_type == TrainingType.TARGET_TN, "Train type must be TARGET_TN!"
                    return self.loss(tn, tn_target)

            # Choose vmap axis setup
            if self.train_type == TrainingType.UNSUPERVISED:
                vmapped_loss = jax.jit(jax.vmap(single_loss, in_axes=(0,)), backend=self.device)  # only data needed
                return jnp.mean(vmapped_loss(data))
            elif self.train_type == TrainingType.SUPERVISED:
                vmapped_loss = jax.jit(jax.vmap(single_loss, in_axes=(0, 0)), backend=self.device)  # both data and targets batched
                return jnp.mean(vmapped_loss(data, targets))
            else:  # TARGET_TN
                vmapped_loss = jax.jit(jax.vmap(single_loss, in_axes=(0,)), backend=self.device)  # only data needed
                return jnp.mean(vmapped_loss(data))

        if isinstance(self.strategy, Sweeps):
            
            # initialize optimizers
            self.loss_func = jax.jit(loss_fn, backend=self.device)
            self.opt_states = []

            for s, sites in enumerate(self.strategy.iterate_sites(self)):
                self.strategy.prehook(self, sites)
                
                self.sitetags = [self.site_tag(site) for site in sites]
                
                params_i = self.select_tensors(self.sitetags)[0].data
                params_i = jnp.expand_dims(params_i, axis=0) # add batch dimension
                print(params_i.shape)

                self.step, opt_state = self.create_train_step(params=params_i, loss_func=self.loss_func)

                self.opt_states.append(opt_state)

                self.strategy.posthook(self, sites)
        else:
            if self.strategy != 'global':
                raise ValueError("Only Global Gradient Descent and DMRG Sweeping strategy is supported for now!")
            
            # initialize optimizer
            params = self.arrays
            self.loss_func = jax.jit(loss_fn, backend=self.device)
            self.step, self.opt_state = self.create_train_step(params=params, loss_func=self.loss_func)
        
        finish = False
        start_train = time()
        with tqdm(total = epochs, desc = "epoch") as outerbar:
            for epoch in range(epochs):
                time_epoch = time()
                
                if self.train_type == TrainingType.TARGET_TN:
                    params = self.arrays
                    _, self.opt_state, loss_epoch = self.step(params, self.opt_state, None, grad_clip_threshold=gradient_clip_threshold)
                    
                    self.history['loss'].append(loss_epoch)
                    self.history['epoch_time'].append(time() - time_epoch)
                else:
                    loss_batch = 0
                    for batch_data in _batch_iterator(inputs, targets, batch_size, dtype=dtype, shuffle=shuffle, seed=seed, alternate_flip=alternate_flip):
                        if isinstance(self.strategy, Sweeps):
                            # Sweeping strategy
                            loss_curr = 0
                            for s, sites in enumerate(self.strategy.iterate_sites(self)):
                                self.strategy.prehook(self, sites)
                                
                                self.sitetags = [self.site_tag(site) for site in sites]
                                
                                params_i = self.select_tensors(self.sitetags)[0].data
                                params_i = jnp.expand_dims(params_i, axis=0) # add batch dimension

                                _, self.opt_states[s], loss_group = self.step(params_i, self.opt_states[s], batch_data, grad_clip_threshold=gradient_clip_threshold)
                                
                                self.strategy.posthook(self, sites)

                                loss_curr += loss_group
                            loss_curr /= (s+1)
                        else:
                            # Global strategy
                            params = self.arrays
                            _, self.opt_state, loss_curr = self.step(params, self.opt_state, batch_data, grad_clip_threshold=gradient_clip_threshold)

                        loss_batch += loss_curr

                        if normalize:
                            self.normalize()

                        if canonize[0] and not isinstance(self.strategy, Sweeps):
                            self.canonicalize(canonize[1], inplace=True)

                    loss_epoch = loss_batch/n_batches
                    loss_epoch = loss_epoch.item()

                    self.history['loss'].append(loss_epoch)

                    self.history['epoch_time'].append(time() - time_epoch)

                    if finish: break

                    # if for some reason you have a limited amount of time to train the model
                    if time_limit is not None and (time() - start_train + np.mean(self.history['epoch_time']) >= time_limit):
                        self.history["unfinished"] = True
                        return self.history
                    
                    # evaluate validation loss
                    if val_inputs is not None:

                        loss_val_epoch = self.evaluate(val_inputs, val_targets, batch_size=val_batch_size, embedding=embedding, evaluate_type=self.train_type, metric=eval_metric, dtype=dtype, shuffle=shuffle, seed=seed, alternate_flip=alternate_flip)
                        
                        self.history['val_loss'].append(loss_val_epoch)
                        if display_val_acc:
                            accuracy_val_epoch = self.accuracy(val_inputs, val_targets, batch_size=val_batch_size, embedding=embedding, shuffle=shuffle, dtype=dtype, seed=seed, alternate_flip=alternate_flip)
                            self.history['val_acc'].append(accuracy_val_epoch)

                        if earlystop:
                            if earlystop.monitor == 'val_loss':
                                current = loss_val_epoch
                                return_value = earlystop.on_end_epoch(current, epoch, self)
                    else:
                        if earlystop:
                            if earlystop.monitor == 'loss':
                                    current = loss_epoch
                            else:
                                current = sum(self.history[earlystop.monitor][-num_batches:])/num_batches
                            return_value = earlystop.on_end_epoch(current, epoch, self)
                if epoch == 0:
                    outerbar.bar_format = "{l_bar}{bar} {n_fmt}/{total_fmt} {postfix}"
                
                if val_inputs is not None:
                    if display_val_acc:
                        outerbar.set_postfix({'loss': f'{loss_epoch:.4f}', 'val_loss': f'{self.history["val_loss"][-1]:.4f}', 'val_acc': f'{self.history["val_acc"][-1]:.4f}'})
                    else:
                        outerbar.set_postfix({'loss': loss_epoch, 'val_loss': f'{self.history["val_loss"][-1]:.4f}'})
                else:
                    outerbar.set_postfix({'loss': f'{loss_epoch:.4f}'})
                
                outerbar.update()

                if earlystop:
                    if return_value == 1:
                        self = earlystop.memory['best_model']
                        return self.history

        return self.history

    def evaluate(self, 
                 inputs: Collection = None,
                 targets: Optional[Collection] = None,
                 tn_target: Optional[qtn.TensorNetwork] = None,
                 batch_size: Optional[int] = None,
                 embedding: Embedding = TrigonometricEmbedding(),
                 evaluate_type: int = TrainingType.UNSUPERVISED,
                 return_list: bool = False,
                 metric: Optional[Callable] = None,
                 dtype: Any = jnp.float_,
                 shuffle: Optional[bool] = False,
                 seed: Optional[int] = 42,
                 alternate_flip: Optional[bool] = False) -> Union[float, np.ndarray]:
        
        """ Evaluates the model on the data.

        Parameters
        ----------
        inputs : sequence of :class:`numpy.ndarray`
            Data used for evaluation.
        targets: sequence of :class:`numpy.ndarray`
            Targets for evaluation (if evaluation is supervised).
        tn_target: :class:`quimb.tensor.tensor_core.TensorNetwork` or any specialized TN class from `quimb`
            Target tensor network for evaluation.
        batch_size : int, or default `None`
            Number of samples per evaluation.
        embedding : :class:`tn4ml.embeddings.Embedding`
            Data embedding function.
        evaluate_type : int
            Type of evaluation: 0 = 'unsupervised' or 1 ='unsupervised'.
        return_list : bool
            If True, returns list of loss values for each batch.
        metric : function
            Metric function for evaluation.
        dtype : Any
            Data type of input data.
        shuffle : bool
            If True, data is shuffled.
        seed : int
            Random seed for data shuffling.
        
        Returns
        -------
        float
            Loss value.
        """

        if evaluate_type not in [TrainingType.UNSUPERVISED, TrainingType.SUPERVISED, TrainingType.TARGET_TN]:
            raise ValueError(f"Specify type of evaluation: {TrainingType.UNSUPERVISED.name}, {TrainingType.SUPERVISED.name}, or {TrainingType.TARGET_TN.name}!")

        if hasattr(self, 'batch_size'):
            if batch_size is None:
                batch_size = self.batch_size

        if not hasattr(self, 'batch_size'):
            self.batch_size = batch_size
        
        if return_list:
            loss = []
        
        def loss_fn(data = None, targets = None, *params):
            """
            Batches embedding + loss computation internally, with model params fixed externally.
            """
            tn = self.copy()

            if hasattr(self, 'sitetags') and self.sitetags is not None:
                tn.select_tensors(self.sitetags)[0].modify(data = params[0])
            else:
                for tensor, array in zip(tn.tensors, params):
                    tensor.modify(data=array)

            # Define batched version of embed + loss logic
            def single_loss(x, y = None):
                tn_i = embed(x, embedding) # create TN from data

                if evaluate_type == TrainingType.UNSUPERVISED:
                    return self.loss(tn, tn_i)
                elif evaluate_type == TrainingType.SUPERVISED:
                    return self.loss(tn, tn_i, y)
                else:
                    assert evaluate_type == TrainingType.TARGET_TN, "Train type must be TARGET_TN!"
                    return self.loss(tn, tn_target)

            # Choose vmap axis setup
            if evaluate_type == TrainingType.UNSUPERVISED:
                vmapped_loss = jax.jit(jax.vmap(single_loss, in_axes=(0,)), backend=self.device)   # only data needed
                return vmapped_loss(data)
            elif evaluate_type == TrainingType.SUPERVISED:
                vmapped_loss = jax.jit(jax.vmap(single_loss, in_axes=(0, 0)), backend=self.device)   # both data and targets batched
                return vmapped_loss(data, targets)
            else:  # TARGET_TN
                vmapped_loss = jax.jit(jax.vmap(single_loss, in_axes=(0,)), backend=self.device) # only data needed
                return vmapped_loss(data)
        
        if inputs is not None:
            loss_value = 0
            for batch_data in _batch_iterator(inputs, targets, batch_size, dtype=dtype, shuffle=shuffle, seed=seed, alternate_flip=alternate_flip):
                if type(batch_data) == tuple and len(batch_data) == 2:
                    x, y = batch_data
                    x, y = jnp.array(x, dtype=dtype), jnp.array(y)
                else:
                    x = jnp.array(batch_data, dtype=dtype)
                    y = None

                if isinstance(self.strategy, Sweeps):
                    
                    loss_curr = np.zeros((x.shape[0],))
                    for s, sites in enumerate(self.strategy.iterate_sites(self)):
                        self.strategy.prehook(self, sites)
                        
                        self.sitetags = [self.site_tag(site) for site in sites]
                        
                        params_i = self.select_tensors(self.sitetags)[0].data
                        params_i = jnp.expand_dims(params_i, axis=0)

                        loss_group = loss_fn(x, y, *params_i)

                        self.strategy.posthook(self, sites)

                        loss_curr += loss_group
                    loss_curr /= (s+1)
                else:
                    params = self.arrays
                    loss_curr = loss_fn(x, y, *params)
                
                loss_value += np.mean(loss_curr)

                if return_list:
                    loss.extend(loss_curr)

            if return_list:
                return np.array(loss)
            
            loss_value = loss_value / (len(inputs)//batch_size)
        else:
            assert evaluate_type == TrainingType.TARGET_TN # If inputs are not provided, evaluation type must be 2!
            assert tn_target is not None # If inputs are not provided, target tensor network must be provided!

            loss_value = loss_fn(None, None, *params)
        return loss_value.item()
    
    def convert_to_pytree(self):
        """ Converts tensor network to pytree structure and returns its skeleon.
        Reference to :func:`quimb.tensor.pack`.
        
        Returns
        -------
        pytree (dict)
        skeleton (Tensor, TensorNetwork, or similar) - A copy of obj with all references to the original data removed.
        """
        params, skeleton = qtn.pack(self)
        return params, skeleton

def load_model(model_name, dir_name=None):
    """ Loads the Model from pickle file.

    Parameters
    ----------
    model_name : str
        Name of the model.
    dir_name : str
        Directory where model is stored.
    
    Returns
    -------
    :class:`tn4ml.models.Model` or subclass
    """
    if dir_name == None:
        return qu.load_from_disk(f'{model_name}.pkl')
    return qu.load_from_disk(f'{dir_name}/{model_name}.pkl')

def _check_chunks(chunked: Collection, batch_size: int = 2) -> Collection:
    """ Checks if the last chunk has lower size then batch size.
    
    Parameters
    ----------
    chunked : sequence
        Sequence of chunks.
    batch_size : int
        Size of batch.
    
    Returns
    -------
    sequence
    """
    if len(chunked[-1]) < batch_size:
        chunked = chunked[:-1]
    return chunked

def _batch_iterator(x: Collection, 
                    y: Optional[Collection] = None, 
                    batch_size:int = 2, 
                    dtype: Any = jnp.float_, 
                    shuffle: bool = True, 
                    seed: int = 0,
                    alternate_flip: bool = False):
    """ Iterates over batches of data with optional alternating batch flipping.
    
    Parameters
    ----------
    x : sequence
        Input data.
    batch_size : int
        Size of batch.
    y : sequence, or default `None`
        Target data.
    dtype : Any
        Data type of input data.
    shuffle : bool
        If True, shuffles the data.
    seed : int
        Seed for shuffling.
    alternate_flip : bool
        If True, flips every other batch along axis=1.
    
    Yields
    ------
    tuple
        Batch of input and target data (if target data is provided)
    """

    key = jax.random.PRNGKey(seed)

    # Convert to JAX array
    x = jax.numpy.asarray(x, dtype=dtype)

    if shuffle:
        perm = jax.random.permutation(key, len(x))
        x = x[perm]  # Shuffle x
        if y is not None:
            y = jax.numpy.asarray(y)  # Keep dtype as is
            y = y[perm]

    # Chunk the data
    x_chunks = _check_chunks(list(funcy.chunks(batch_size, x)), batch_size)

    if y is not None:
        y_chunks = _check_chunks(list(funcy.chunks(batch_size, y)), batch_size)
        
        # Track batch number for alternating flips
        for batch_idx, (x_chunk, y_chunk) in enumerate(zip(x_chunks, y_chunks)):
            # Flip every other batch if alternate_flip is enabled
            if alternate_flip and batch_idx % 2 == 1:  # For odd-indexed batches (0-indexed)
                x_chunk = jax.numpy.asarray(x_chunk, dtype=dtype)
                y_chunk = jax.numpy.asarray(y_chunk)
                # Flip each sample in the batch along axis=1
                x_chunk = jax.numpy.flip(x_chunk, axis=1)
            
            yield x_chunk, y_chunk
    else:
        for batch_idx, x_chunk in enumerate(x_chunks):
            # Flip every other batch if alternate_flip is enabled
            if alternate_flip and batch_idx % 2 == 1:  # For odd-indexed batches (0-indexed)
                x_chunk = jax.numpy.asarray(x_chunk, dtype=dtype)

                # Flip each sample in the batch along axis=1
                x_chunk = jax.numpy.flip(x_chunk, axis=1)
                
            yield x_chunk
import re
from typing import NamedTuple
import numpy as np

def return_digits(array):
    """Helper function to convert array of string numbers to integers.
    """
    digits=[]
    for text in array:
        split_text = re.split(r'(\d+)', text)
        for t in split_text:
            if t.isdigit(): digits.append(int(t))
            else: continue
    return digits

def gramschmidt(A):
    """Function that creates an orthogonal basis from a matrix `A`.

    Parameters
    ----------
    A : Matrix

    Returns
    -------
    `np.numpy.ndarray`
        Matrix in a orthogonal basis

    """
    m = A.shape[0]

    for i in range(m-1):
        v = [A[i, :]]
        v /= np.linalg.norm(v)
        A[i, :] = v

        sA = A[i+1:, :]
        u = np.matmul(sA, np.transpose(v))
        sA -= np.matmul(u, np.conjugate(v))
        A[i+1:, :] = sA
        u = np.matmul(sA, np.transpose(v))

    A[-1,:] /= np.linalg.norm(A[-1,:])
    return A

class EarlyStopping:
    """ Variation of `EarlyStopping` class from :class:tensorflow.

    Attributes
    ----------
    monitor : str
        Name of metric to be monitored.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    patience : int
        Number of epochs for tracking the metric, if no improvement after training is stopped.
    mode: str
        Two options are valid: `min` - minimization, `max` - maximization of objective function
    """
    def __init__(self, monitor, min_delta, patience, mode):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
    
    def on_begin_train(self, history):
        if self.monitor not in history.keys():
            raise ValueError(f'This metric {self.monitor} is not monitored. Change metric for EarlyStopping.monitor')
        if self.mode not in ['min', 'max']:
            raise ValueError(f'EarlyStopping mode can be either "min" or "max".')

        self.memory = dict()
        self.memory['best'] = np.Inf if self.mode == 'min' else -np.Inf
        self.memory['best_epoch'] = 0 # track on each epoch
        if self.mode == 'min':
            self.min_delta = self.min_delta*(-1)
            self.operator = np.less
        else:
            self.min_delta = self.min_delta*1
            self.operator = np.greater
        self.memory['wait'] = 0
    
    def on_end_epoch(self, loss_current, epoch):

        if self.memory['wait'] == 0 and epoch == 0:
            self.memory['best'] = loss_current
            self.memory['best_model'] = self
            self.memory['best_epoch'] = epoch
            #memory['wait'] += 1
        if epoch > 0: self.memory['wait'] += 1
        if self.operator(loss_current - self.min_delta, self.memory['best']):
            self.memory['best'] = loss_current
            self.memory['best_model'] = self
            self.memory['best_epoch'] = epoch
            self.memory['wait'] = 0
        if self.memory['wait'] >= self.patience and epoch > 0:
            best_epoch = self.memory['best_epoch']
            print(f'Training stopped by EarlyStopping on epoch: {best_epoch}', flush=True)
            self = self.memory['best_model']
            return 1
        if self.memory['wait'] > 0: 
            print('Waiting for ' + str(self.memory['wait']) + ' epochs.', flush=True)
        
        return 0


class ReduceLROnPlateau:
    """ Variation of `ReduceLROnPlateau` class from :class:tensorflow.

    Attributes
    ----------
    monitor : str
        Name of metric to be monitored.
    factor: float
        factor by which the learning rate will be reduced. new_lr = lr * factor.
    patience : int
        Number of epochs with no improvement after which learning rate is reduced.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    mode: str
        Two options are valid: `min` - minimization, `max` - maximization of objective function
    min_lr: float
        lower bound on the learning rate.

    """
    def __init__(self, monitor, factor, patience, min_delta, mode, min_lr):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_lr = min_lr


class ExponentialDecay:
    """ Variation of `ExponentialDecay` class from :class:tensorflow. Once
    the exponential decay has started, the learning rate at each step is computed
    as: initial_learning_rate * decay_rate ^ (step / decay_steps) .

    Attributes
    ----------
    initial_learning_rate : float
        Initial learning rate.
    decay_steps : int
        Number of decay_steps
    decay_rate : float
        Decay rate of the algorithm.
    start_decay : int
        The step in which the exponential decay starts.
    """
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        start_decay: int = 0
        ):

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.start_decay = start_decay

    def __call__(self, step):
        return self.initial_learning_rate * (self.decay_rate ** (step / self.decay_steps))
    

class ExponentialGrowth:
    """ Exponential growth of learning rate. Once
    the exponential growth has started, the learning rate at each step is computed
    as: initial_learning_rate * ((1 + growth_rate)^(step / decay_steps)) .

    Attributes
    ----------
    initial_learning_rate : float
        Initial learning rate.
    growth_steps : int
        Number of time steps
    growth_rate : float
        Growth rate of the algorithm.
    start_step : int
        The step in which the exponential growth starts.
    """
    def __init__(
        self,
        initial_learning_rate: float,
        growth_steps: int,
        growth_rate: float,
        start_step: int = 0
        ):

        self.initial_learning_rate = initial_learning_rate
        self.growth_steps = growth_steps
        self.growth_rate = growth_rate
        self.start_step = start_step

    def __call__(self, step):
        return self.initial_learning_rate * ((1 + self.growth_rate) ** (step / self.growth_steps))
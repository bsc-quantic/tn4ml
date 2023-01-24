import re
from typing import NamedTuple

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
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
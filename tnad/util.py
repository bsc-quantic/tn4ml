import re
from typing import NamedTuple

def return_digits(array):
    """
    Helper function to convert array of string numbers to integers.
    """
    digits=[]
    for text in array:
        split_text = re.split(r'(\d+)', text)
        for t in split_text:
            if t.isdigit(): digits.append(int(t))
            else: continue
    return digits

class EarlyStopping(NamedTuple):
    """ Variation of EarlyStopping class from Tensorflow """
    monitor: str # name of metric to be monitored
    min_delta: float # minimum change in the monitored quantity to qualify as an improvement
    patience: int # number of epochs for tracking the metric, if no improvement after training is stopped
    mode: str # 'min' - minimization, 'max' - maximization of objective function
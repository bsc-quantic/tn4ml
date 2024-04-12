import re
import jax.numpy as jnp
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

def normalize(v, p=2, atol=1e-9):
    """
    Normalize a vector based on its p-norm, with a check to avoid division by a very small norm.
    
    Parameters 
    ----------
    v : jax.numpy.ndarray
        The vector to be normalized.
    p : Real, optional
        The norm type (default is 2).
    atol : Real, optional
        The tolerance level for considering the norm as zero (default is 1e-9).
    
    Returns
    -------
    The normalized vector, or the original vector if its norm is below the tolerance.
    """
    norm = jnp.linalg.norm(v, ord=p)
    if norm > atol:
        return v / norm
    else:
        # Handle the case where the vector is near-zero or the algorithm encounters linear dependence.
        return None # Indicate that the vector should be skipped

def gramschmidt_row(A, atol=1e-10):
    """
    Performs the Modified Gram-Schmidt process on matrix A, skipping near-zero norm vectors.
    By row.

    Parameters
    ----------
    A : jax.numpy.ndarray
        The input matrix to be orthogonalized.
    p : Real, optional
        The norm type (default is 2).
    atol : Real, optional
        The tolerance level for considering vectors as zero (default is 1e-9).
    
    Returns
    -------
    Orthonormal matrix A.
    """
    m, n = A.shape
    Q = []
    for i in range(m):
        q = A[i, :]
        for j in range(0, i):
            rij = jnp.tensordot(jnp.conj(Q[j]), q, axes=1)
            q = q - rij * Q[j]
        norm_q = jnp.linalg.norm(q)
        if norm_q > atol:
            Q.append(q / jnp.linalg.norm(q))
        else:
            print(f"Vector at row {i} is zero or near-zero norm, cannot normalize.")
            Q.append(jnp.zeros_like(A[i, :]))
    Q = jnp.stack(tuple(Q), axis=0)
    return Q

def gramschmidt_col(A, atol=1e-10):
    # TODO - fix, not sure if it works
    """
    Performs the Modified Gram-Schmidt process on matrix A, skipping near-zero norm vectors.
    By column.

    Parameters
    ----------
    A : jax.numpy.ndarray
        The input matrix to be orthogonalized.
    p : Real, optional
        The norm type (default is 2).
    atol : Real, optional
        The tolerance level for considering vectors as zero (default is 1e-9).
    
    Returns
    -------
    Orthonormal matrix A.
    """
    m, n = A.shape
    Q = []
    for j in range(n):
        q = A[:, j]
        for i in range(0, j):
            rij = jnp.tensordot(jnp.conj(Q[i]), q, axes=1)
            q = q - rij * Q[i]
        norm_q = jnp.linalg.norm(q)
        if norm_q > atol:
            Q.append(q / jnp.linalg.norm(q))
        else:
            print(f"Vector at col {j} is zero or near-zero norm, cannot normalize.")
            Q.append(jnp.zeros_like(A[:, j]))
    Q = jnp.stack(tuple(Q), axis=1)
    return Q

def gradient_clip(grads, threshold=1.0):
    """ Clip gradients to a maximum threshold value. 
    
    Parameters
    ----------
    grads : list
        List of gradients.
    threshold : float, optional
        Maximum value of the gradient norm (default is 1.0).

    Returns
    -------
    List of clipped gradients.s
    """
    assert threshold > 0, "Threshold must be positive."
    assert len(grads) > 0, "No gradients to clip."
    assert all([len(g) > 0 for g in grads]), "No gradients to clip."
    
    new_grads = []
    for gradients in grads:
        grad_norm = jnp.linalg.norm(gradients)
        scale_factor = min(1., threshold / (grad_norm + 1e-6))
        scaled_gradients = [g * scale_factor for g in gradients]
        new_grads.append(scaled_gradients)
    return new_grads

def zigzag_order(images):
    """ Rearrange pixels in zig-zag order (from https://arxiv.org/pdf/1605.05775.pdf).
    
    Parameters
    ----------
    images : list
        List of images to be rearranged.
    
    Returns
    -------
    List of images with pixels in zig-zag order.
    """
    data_zigzag = []
    for x in images:
        image = []
        for i in x:
            image.extend(i)
        data_zigzag.append(image)
    return np.asarray(data_zigzag)

def integer_to_one_hot(labels, num_classes=None):
    """ Convert integer labels to one-hot encoded labels.
    
    Parameters
    ----------
    labels : list
        List of integer labels.
    num_classes : int, optional
        Number of classes (default is None).
    
    Returns
    -------
    One-hot encoded labels.
    """
    # If num_classes is not explicitly given, infer from the labels
    if num_classes is None:
        num_classes = np.max(labels) + 1

    # Create an array of zeros with shape (number of labels, number of classes)
    one_hot_encoded = np.zeros((len(labels), num_classes))

    # Use np.arange to generate indices and labels to specify where the 1s should go
    one_hot_encoded[np.arange(len(labels)), labels] = 1

    return one_hot_encoded

class EarlyStopping:
    # not used
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
    # not used
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
    # not used
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
        print(self.initial_learning_rate * (self.decay_rate ** (step / self.decay_steps)))
        return self.initial_learning_rate * (self.decay_rate ** (step / self.decay_steps))
    

class ExponentialGrowth:
    # not used
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
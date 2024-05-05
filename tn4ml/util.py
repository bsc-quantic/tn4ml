""" Helper functions """

import re
import numpy as np
import torch
import jax.numpy as jnp


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

def return_digits(array):
    """Convert array of string numbers to integers.
    
    Parameters
    ----------
    array : np.array

    Returns
    -------
    np.array
    """
    digits=[]
    for text in array:
        split_text = re.split(r'(\d+)', text)
        for t in split_text:
            if t.isdigit(): digits.append(int(t))
            else: continue
    return digits

def squeeze_image(image, k=3, device=torch.device('cpu')):
    """
    Squeeze over H,W,D dimensions, but enlarge feature dimension to k**(S), where S=dimensionality of image.

    Parameters
    ----------
    image : torch.tensor
        3D image
    k : int = 3
        Kernel stride and size (if k=3 then kernel has shape (3,3,3) and its moving by stride 3)

    Returns
    -------
    torch.tensor
    """
    S = len(image.shape) - 1 # dimensionality of image
    if S < 2:
        ValueError('Image should be 2D or 3D!')

    new_dims = []
    for dim in list(image.shape)[:-1]:
        new_dims.append(dim // k)
    new_dims.append(list(image.shape)[-1] * k**S)
    reshaped_image = torch.zeros(tuple(new_dims), dtype=torch.float64).to(device=device)

    feature = 0
    for x in range(k):
        for y in range(k):
            if S == 3:
                # 3D image
                for z in range(k):
                    kernel = np.zeros((k,k,k))
                    kernel[x,y,z] = 1.0
                    kernel = np.expand_dims(kernel, axis=-1)
                    kernel = torch.tensor(kernel, dtype=torch.float64).to(device=device)
                    
                    tensor = torch.zeros(tuple(new_dims[:-1]), dtype=torch.float64).to(device=device)
                    for i in range(0, image.shape[0], k):
                        for j in range(0, image.shape[1], k):
                            for l in range(0, image.shape[2], k):
                                patch = torch.sum(image[i:i+k, j:j+k, l:l+k, :] * kernel).to(device=device)
                                
                                tensor[i//k, j//k, l//k] = patch
                    reshaped_image[:,:,:,feature] = tensor
                    feature += 1
            else:
                # 2D image
                kernel = np.zeros((k,k))
                kernel[x,y] = 1.0
                #kernel = np.expand_dims(kernel, axis=-1)
                kernel = torch.tensor(kernel, dtype=torch.float64).to(device=device)
                
                tensor = torch.zeros(tuple(new_dims[:2]), dtype=torch.float64).to(device=device)
                for i in range(0, image.shape[0], k):
                    for j in range(0, image.shape[1], k):
                            patch = torch.sum(torch.tensordot(image[i:i+k, j:j+k, :], kernel)).to(device=device)
                            new_row = i//k
                            new_col = j//k

                            # Create a mask tensor indicating the position to assign the patch - VMAP
                            mask = torch.zeros_like(tensor, dtype=torch.bool).to(device=device)
                            mask[new_row, new_col] = True
                            tensor = tensor + patch * mask

                #reshaped_image[:,:,feature] = tensor
                # Create a mask tensor indicating the position to assign the patch - VMAP
                mask = torch.zeros_like(reshaped_image, dtype=torch.bool).to(device=device)
                mask[:,:,feature] = True
                reshaped_image = reshaped_image + tensor.unsqueeze(-1) * mask
                feature += 1
    return reshaped_image

def unsqueeze_image_pooling(image, S=3, device=torch.device('cpu')):
    """
    Unsqueeze over H,W,(D) dimensions, but average over feature dimension.

    Parameters
    ----------
    image : np.array
        2D or 3D image
    S : int = 3
        Dimensionality of input image. S = 3 --> 3D image

    Returns
    -------
    np.array
    """
    n_mps, n_features = image.shape
    new_dims = []
    for _ in range(S):
        new_dims.append(round(n_mps**(1/S)))
    new_dims.append(n_features)

    reshaped_image = image.reshape(tuple(new_dims))
    averaged_image = torch.mean(reshaped_image, -1).to(device=device)
    averaged_image = (averaged_image - torch.min(averaged_image))/(torch.max(averaged_image) - torch.min(averaged_image)).to(device=device)
    averaged_image = torch.unsqueeze(averaged_image, -1).to(device=device)
    return averaged_image

def rearange_image(image, S=3, device=torch.device('cpu')):
    """
    Reshape image back to H, W, (D)
    TODO - implement for image with channels
    
    Parameters
    ----------
    image : np.array
        2D or 3D image
    S : int = 3
        Dimensionality of input image. S = 3 --> 3D image

    Returns
    -------
    np.array
    """
    (N_i,) = image.shape

    new_dims = []
    for _ in range(S):
        new_dims.append(round(N_i**(1/S)))
    new_dims.append(1)

    reshaped_image = image.reshape(tuple(new_dims)).to(device)
    return reshaped_image

def squeeze_dimensions(input_dims, k=3):
    """ Find dimensions after performing squeezing function on image with dimension = input_dims.

     Parameters
    ----------
    input_dims : tuple [int, int, ...]

    k : int
        Kernel size, in all D dimensions of image, and value of stride.
    
    Returns
    -------
    tuple, int
    """
    if len(input_dims) < 3:
        ValueError('Input data must have at least 2 dimensions and one channel')
    S = len(input_dims) - 1

    new_dims = []
    for dim in input_dims[:-1]:
        new_dims.append(dim // k)
    
    feature_dim = input_dims[-1] * k**S # C = input_dims[-1]
    return tuple(new_dims), feature_dim

def rearanged_dimensions(input_dims, S=3):
    """ Find dimensions after performing after returning image with dimension = input_dims to image format HxWx(D)xC.
    For now C = 1 (TODO implement for channels C > 1).

     Parameters
    ----------
    input_dims : tuple [int, int, ...]

    S : int
        Dimensionality of input image. S = 3 --> 3D image
    
    Returns
    -------
    tuple
    """

    (N_i,) = input_dims

    new_dims = []
    for _ in range(S):
        new_dims.append(round(N_i**(1/S)))
    new_dims.append(1) # averaged by feature dimension or just added C as channel value

    return tuple(new_dims)

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
        print(self.initial_learning_rate * (self.decay_rate ** (step / self.decay_steps)))
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
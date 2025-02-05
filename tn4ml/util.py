import re
import jax
import jax.numpy as jnp
import numpy as np
from typing import List

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

def zigzag_order(data):
    """ Rearrange pixels in zig-zag order (from https://arxiv.org/pdf/1605.05775.pdf).

    Parameters
    ----------
    images : list
        List of images to be rearranged.

    Returns
    -------
    List of images with pixels in zig-zag order.
    """
    data = np.squeeze(data)
    # Reshape the array to (N, -1) where N is the number of images, and flatten each image
    data_zigzag = data.reshape(data.shape[0], -1)
    return data_zigzag

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

def pad_image_alternately(image: np.ndarray, k: int) -> np.ndarray:
    """
    Pad the image alternately from the right and left sides in a single step.

    Parameters
    ----------
    image: :class:`numpy.ndarray`
        A 2D array of pixel intensities.
    k: int
        The patch size.

    Returns
    -------
    :class:`numpy.ndarray`
        A padded 2D array of pixel intensities.
    """
    H, W = image.shape

    # Determine padding required
    pad = (k - H % k) % k

    # Alternate padding by adding to left or right as necessary
    top_pad, bottom_pad = (pad//2, pad//2) if pad % 2 == 0 else (pad//2 + 1, pad//2)
    left_pad, right_pad = (pad//2, pad//2) if pad % 2 == 0 else (pad//2 + 1, pad//2)

    # Apply padding in a single operation
    padded_image = jnp.pad(
        image,
        ((top_pad, bottom_pad), (left_pad, right_pad)),
        mode='constant',
        constant_values=0
    )

    return padded_image

def divide_into_patches(image: np.ndarray, k: int) -> np.ndarray:
    """
    Divide the image into patches of size kxk.

    Parameters
    ----------
    image: :class:`numpy.ndarray
        A 2D array of pixel intensities.
    k: int
        The patch size.

    Returns
    -------
    :class:`numpy.ndarray`
        A list of 2D arrays, each of size kxk.
    """
    # Pad the image to ensure it's divisible by k
    padded_image = pad_image_alternately(image, k) if image.shape[0] % k != 0 else image

    H, W = padded_image.shape

    # Reshape and move axes to create kxk patches
    patches = padded_image.reshape(H // k, k, W // k, k).swapaxes(1, 2).reshape(-1, k, k)

    return jnp.array(patches)

def from_dense_to_mps(statevector: jnp.ndarray, n_qubits: int, max_bond: int = None) -> List[jnp.ndarray]:
    """
    Convert a dense statevector to a Matrix Product State (MPS) representation in JAX.

    Parameters
    ----------
    statevector: jnp.ndarray
        A dense statevector as a 1D array.
    n_qubits: int
        The number of qubits (log2 of the statevector length).
    max_bond: int, optional
        The maximum bond dimension for truncating the MPS tensors.

    Returns
    -------
    list[:class:`jax.numpy.ndarray`]
        A list of MPS tensors in JAX with shapes for Left-Right-Physical (LRP).
    """
    # Step 1: Reshape the statevector to a tensor of shape (2, 2, ..., 2)
    psi = statevector.reshape((2,) * n_qubits)

    # Initialize a list to store MPS tensors
    mps = []
    current_tensor = psi
    left_bond = 1  # Start with a left bond dimension of 1
    physical_dim = 2  # Physical dimension for qubit systems

    for _ in range(n_qubits - 1):
        # Reshape the current tensor for SVD: (left_bond * physical_dim, -1)
        d1 = left_bond * physical_dim
        current_tensor = current_tensor.reshape(d1, -1)

        # Step 2: Perform SVD
        u, s, vh = jnp.linalg.svd(current_tensor, full_matrices=False)

        # Truncate based on max bond dimension
        bond_dim = s.shape[0] if max_bond is None else min(s.shape[0], max_bond)
        u = u[:, :bond_dim]
        s = s[:bond_dim]
        vh = vh[:bond_dim, :]

        # Reshape `u` to the `LRP` format
        right_bond = bond_dim
        mps_tensor = u.reshape(left_bond, physical_dim, right_bond).transpose([0, 2, 1]) # (l, p, r) -> (l, r, p)
        mps.append(mps_tensor)

        # Move the singular values into the next tensor
        current_tensor = jnp.diag(s) @ vh
        left_bond = right_bond  # Update left_bond for the next tensor

        # Reshape the next tensor
        remaining_elements = current_tensor.size
        if remaining_elements == bond_dim * physical_dim:
            current_tensor = current_tensor.reshape(bond_dim, physical_dim, 1)
        elif remaining_elements >= bond_dim * physical_dim:
            current_tensor = current_tensor.reshape(bond_dim, physical_dim, -1)

    # Final tensor for the last site with shape (left_bond, 1, physical_dim=2)
    mps.append(current_tensor.reshape(left_bond, physical_dim, 1).transpose([0, 2, 1]))

    return mps


def from_mps_to_dense(mps: List[jnp.ndarray], n_qubits: int) -> jnp.ndarray:
    """
    Convert a Matrix Product State (MPS) representation back to a dense statevector.

    Parameters
    ----------
    mps: List[jnp.ndarray]
        A list of MPS tensors where each tensor has the shape "lrp".
    n_qubits: int
        The number of qubits.

    Returns
    -------
    :class:`jax.numpy.ndarray`
        The dense statevector as a 1D array of length 2^n_qubits.

    """
    # Start with the first MPS tensor.
    statevector = mps[0].transpose([0, 2, 1])  # (l, r, p) -> (l, p, r)

    # Iterate over the remaining MPS tensors and contract them.
    for i in range(1, n_qubits):
        next_tensor = mps[i].transpose([0, 2, 1]) # (l, r, p) -> (l, p, r)

        statevector = jnp.tensordot(statevector, next_tensor, axes=[-1, 0])  # (l, p, p, p, bond) (bond, p, r) -> (l, p, p, p, p, r)

    # Flatten the resulting tensor to form the dense statevector.
    return statevector.flatten()


class EarlyStopping:
    """ Variation of `EarlyStopping` class from :class:`tensorflow`.

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
        """
        Set up the memory for the early stopping algorithm.

        Parameters
        ----------
        history : dict
            The dictionary containing the training history.

        Raises
        ------
        ValueError
            If the metric to be monitored is not in the history.
        ValueError
            If the mode is not `min` or `max`.
        """
        if self.monitor not in history.keys():
            raise ValueError(f'This metric {self.monitor} is not monitored. Change metric for EarlyStopping.monitor')
        if self.mode not in ['min', 'max']:
            raise ValueError(f'EarlyStopping mode can be either "min" or "max".')

        self.memory = dict()
        self.memory['best'] = np.inf if self.mode == 'min' else -np.inf
        self.memory['best_epoch'] = 0 # track on each epoch
        if self.mode == 'min':
            self.min_delta = self.min_delta*(-1)
            self.operator = np.less
        else:
            self.min_delta = self.min_delta*1
            self.operator = np.greater
        self.memory['wait'] = 0

    def on_end_epoch(self, loss_current, epoch):
        """
        Check if the training should be stopped based on the monitored metric.

        Parameters
        ----------
        loss_current : float
            The value of the monitored metric.
        epoch : int
            The current epoch number.

        Returns
        -------
        int
            A flag to indicate if the training should be stopped.
        """

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
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
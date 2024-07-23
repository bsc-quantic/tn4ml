""" Test initializer functions """

import pytest
from tn4ml.util import *
import tn4ml
import jax, jax,numpy as jnp

def check_orthonormal_vectors(Q, type='rows', atol=1e-6):
    """
    Check if the rows of matrix Q form an orthonormal set.
    
    Parameters:
    Q : numpy.ndarray
        The matrix with vectors as its rows.
    atol : float, optional
        The absolute tolerance for numerical comparisons.
    
    Returns:
    bool
        True if the rows of Q are orthonormal, False otherwise.
    """
    num_rows, num_cols = Q.shape

    if type == 'rows':
        # Check normality (unit norm for each row vector)
        for row in range(num_rows):
            if not jnp.isclose(jnp.linalg.norm(Q[row, :]), 1.0, atol=atol):
                return False
        print('Normality okay')
        # Check orthogonality between each pair of distinct row vectors
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                print(jnp.dot(Q[i, :], Q[j, :]))
                if not jnp.isclose(jnp.dot(Q[i, :], Q[j, :]), 0, atol=atol):
                    return False
        
        return True
    else:
        # Check normality (unit norm for each col vector)
        for col in range(num_cols):
            print(jnp.linalg.norm(Q[:, col]))
            if not jnp.isclose(jnp.linalg.norm(Q[:, col]), 1.0, atol=atol):
                return False
        print('Normality okay')
        # Check orthogonality between each pair of distinct col vectors
        for i in range(num_cols):
            for j in range(i + 1, num_cols):
                print(jnp.dot(Q[:, i], Q[:, j]))
                if not jnp.isclose(jnp.dot(Q[:, i], Q[:, j]), 0, atol=atol):
                    return False
        
        return True

@pytest.mark.parametrize("type,std,shape", [
        ('bond', 1e-6, (1,2,2)),
        ('bond', 1e-6, (5,5,2)),
        ('bond', 1e-6, (1,5,5)),
        ('copy', 1e-7, (1,2,3)),
        ('copy', 1e-7, (2,2,4)),
        ('copy', 1e-7, (5,2,5)),
        ('bond', 1e-7, (1,5,5)),
        ('bond', 1e-8, (5,1,5)),
        ('copy', 1e-6, (5,5,2,3)),
        ('bond', 1e-6, (5,5,2,3)),
        ('copy', None, (5,5,2)),
        ('bond', None, (5,5,2))
        ])
def test_identity_init(type, std, shape):
    initializer = tn4ml.initializers.identity_init(type, std)
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    assert Q != None


@pytest.mark.parametrize("dist,scale,shape", [
    ('uniform', 1e-2, (2, 3)),
    ('uniform', 1e-2, (5, 5)),
    ('uniform', 1e-2, (1, 5)),
    ('normal', 1e-2, (2, 3)),
    ('normal', 1e-2, (5, 5)),
    ('normal', 1e-2, (1, 5)),
    ('uniform', 1e-3, (5, 5, 2)),
    ('normal', 1e-3, (5, 5, 2)),
    ('uniform', 1e-3, (5, 5, 2, 3)),
    ('normal', 1e-3, (5, 5, 2, 3))
])
def test_gramschmidt_init(dist, scale, shape):
    initializer = tn4ml.initializers.gramschmidt_init(dist, scale)
    Q = initializer(jax.random.key(42), shape, jnp.float32)
    matrix_shape = shape[0], np.prod(shape[1:])

    Q = Q.reshape(matrix_shape)
    assert check_orthonormal_vectors(Q, 'rows')
    assert Q != None
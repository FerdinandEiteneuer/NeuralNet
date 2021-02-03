import numpy as np
from functools import wraps
#blablabla

def sigmoid(z, derivative=False):
    sigma = 1/(1+np.exp(-z))
    if not derivative:
        return sigma
    else:
        return sigma * (1 - sigma)

def relu(z, derivative=False):
    if not derivative:
        return z * (z > 0)
    else:
        return 1 * (z > 0)

def lrelu(alpha=0.1):
    assert 0 <= alpha <= 1, f'alpha must be between 0 and 1, but received: {alpha=}'
    def lrelu_(z, derivative=False):
        if not derivative:
            return z * (z > 0) + alpha * z * (z <= 0)
        else:
            return 1 * (z > 0) + alpha * (z <= 0)

    return lrelu_

def softmax(z, derivative=False):
    nb_examples = z.shape[-1]
    nb_classes = z.shape[0]

    exp_z = np.exp(z - np.max(z))  # subtract max for numerical stability
    norm = np.sum(exp_z, axis=0, keepdims=True)

    exp_norm = exp_z / norm

    if not derivative:
        return exp_norm
    else:
        jacobian = np.einsum('ik,jk->ijk', exp_norm, exp_norm)  # 1/N**2 * exp(z)_{j, m} * exp(z)_{k, m}
        assert jacobian.shape == (nb_classes, nb_classes, nb_examples)

        diagonal = np.zeros(jacobian.shape)
        idx = np.arange(nb_classes)
        diagonal[idx, idx, :] = exp_norm  # 1/N * exp(z)_{j, m} * I_{j,k}  (indices=j,k,m)

        ####diagonal_alternative = np.einsum('ij,jk->ijk', np.eye(nb_classes, nb_classes), exp_norm)  # this also works but is slower by factor 2-3

        derivative_softmax = diagonal - jacobian

        return derivative_softmax

def binary_crossentropy(z, derivative=False):
    if not derivative:
        pass
    else:
        pass

def tanh(z, derivative=False):
    if not derivative:
        return np.tanh(z)
    else:
        return 1 - np.tanh(z) ** 2

def linear(z, derivative=False):
    if not derivative:
        return z
    else:
        return np.ones(z.shape)

def prelu(z):
    raise NotImplementedError

def selu(z):
    raise NotImplementedError

if __name__ == '__main__':

    np.random.seed(1)
    np.set_printoptions(precision=2, linewidth=200)

    z = np.random.random((10, 1000))
    norm = np.sum(z, axis=0, keepdims=True)
    n = z/norm
    print(f'{z.shape=}\n{norm.shape=}')

    jac = np.einsum('ik,jk->ijk', n, n)
    print(f'{jac.shape=}')

    smd = softmax(z, derivative=True)

    shape = (2, 2, 7)
    x = np.zeros(shape)
    data = np.arange(2*7).reshape((2,7))
    i = np.arange(2)


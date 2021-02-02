import numpy as np
from functools import wraps
from layer import Layer
from kernel_initializer import kernels

def check_nan(func):
    check = ['w', 'b']
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func
    return wrapper

class Dense(Layer):
    def __init__(
            self,
            input_dim,
            output_dim,
            activation,
            kernel_init,
            kernel_regularizer=None,
            bias_regularizer=None,
            verbose=False):

        super().__init__()

        self.verbose = verbose
        self.g = activation
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.shape = output_dim, input_dim

        self.w = kernels[kernel_init](self.shape)
        self.b = np.zeros((output_dim, 1))

        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)


    def __str__(self):
        representation = f'layer {self.layer_id} (fully connected) '\
                         f'w.shape={self.w.shape}, b.shape={self.b.shape}'

        return representation

    def forward(self, a):

        self.z = np.dot(self.w, a) + self.b
        self.a = self.g(self.z)
        return self.a

    def backward_step(self, a_next, w_prev, error_prev):

        batch_size = a_next.shape[-1]

        derivative = self.g(self.z, derivative=True)
        error = np.dot(w_prev.T, error_prev) * derivative

        self.dw = 1 / batch_size * np.dot(error, a_next.T)
        self.db = 1 / batch_size * np.sum(error, axis=1, keepdims=True)
        return error

    def get_grads(self, a_prev):

        N = self.error.shape[1]
        if self.verbose: print(f'{N=}, inside get_grads, error.shape={self.error.shape}, a_prev.T.shape={a_prev.T.shape}')
        self.dw = np.dot(self.error, a_prev.T) / N
        self.db = np.sum(self.error, axis=1, keepdims=True) / N

        try:
            if np.any(np.isnan(self.dw)):
                print('ERROR: NAN IN self.dw')
            if np.any(np.isnan(self.db)):
                print('ERROR: NAN IN self.db')
        except TypeError as e:
            print('bad type?')

        return self.dw, self.db

    def get_error(self, back_err, verbose=True):
        derivative = self.g(self.z, derivative=True)
        if verbose: print(f'in get_error {self.layer_id}, {back_err.shape}, {derivative.shape}, {self.z.shape}, {self.g}')
        if verbose: print(f'{derivative=}')
        if verbose: print(f'received {back_err=}')
        if verbose: print(self)
        self.error = back_err * derivative
        if verbose: print(f'{self.error=}')
        #raise ValueError('hat es einen einfluss, was self.error, derivative sind?. Probier auch mal aus die get_error, backward funktionen zu vereinigen. baue einfach das delta und gut ist')
        return self.error

    def backward(self):
        self.backerr = np.dot(self.w.T, self.error)
        return self.backerr

    def update_weights(self, lr):
        if self.verbose: print(f'{self.layer_id}: {self.w.shape=}, {self.dw.shape=}\n   {self.b.shape=}, {self.db.shape=}')
        self.w -= lr * self.dw
        self.b -= lr * self.db


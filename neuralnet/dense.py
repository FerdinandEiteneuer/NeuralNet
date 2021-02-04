import numpy as np
from functools import wraps

from .layer import Layer
from .kernel_initializer import kernels

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

import numpy as np
from functools import wraps

from .layer import Layer
from . import kernel_initializers

class Dense(Layer):
    def __init__(
            self,
            input_dim,
            output_dim,
            activation,
            kernel_initializer=kernel_initializers.normal,
            kernel_regularizer=None,
            bias_regularizer=None,
            verbose=False):

        super().__init__()

        self.verbose = verbose
        self.g = activation

        shape = output_dim, input_dim
        self.w = kernel_initializers.create(kernel_initializer, shape)
        self.dw = np.zeros(self.w.shape)

        self.b = np.zeros((output_dim, 1))
        self.db = np.zeros(self.b.shape)

        self.kernel_regularizer = kernel_regularizer(self, 'w') if kernel_regularizer else None
        self.bias_regularizer = bias_regularizer(self, 'b') if bias_regularizer else None

    def __str__(self):
        description = f'layer {self.layer_id} (fully connected) '\
                         f'w.shape={self.w.shape}, b.shape={self.b.shape}'

        return description

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

        if self.kernel_regularizer:
            self.dw += self.kernel_regularizer.derivative()

        if self.bias_regularizer:
            self.db += self.bias_regularizer.derivative()

        return error

    def loss_from_regularizers(self):

        loss = 0
        if self.kernel_regularizer:
            loss += self.kernel_regularizer.loss()
        if self.bias_regularizer:
            loss += self.bias_regularizer.loss()
        return loss

import numpy as np
from functools import wraps
from layer import Layer
from kernel_initializer import kernels


class Dense(Layer):
    def __init__(self, input_dim, output_dim, activation, kernel_init, verbose=False):

        self.verbose = verbose
        self.g = activation
        self.shape = output_dim, input_dim

        self.w = kernels[kernel_init](self.shape)
        self.b = np.zeros((output_dim, 1))

        self.layer_id = next(self.__class__._ids)  # layer number automatically generated.
        if self.verbose: print('AFTER INIT LAYER SHAPE:', self.w.shape)

    #def __call__(self, a):
    #    return self.forward(a, gradient_check = False, grad_check_info=0)

    def __str__(self):
        representation = f'Layer number {self.layer_id}. '\
                         f'w.shape={self.w.shape}, b.shape={self.b.shape}'

        return representation

    def forward(self, a, gradient_check = False, grad_check_info=0):
        if gradient_check:
            w_temp = np.copy(self.w)
            j, k, _, _, eps = grad_check_info
            self.w[j,k] += eps

        self.z = np.dot(self.w, a) + self.b
        self.a = self.g(self.z)

        if gradient_check:
            self.w = w_temp

        return self.a

    def get_grads(self, a_prev):

        N = self.error.shape[1]
        if self.verbose: print(f'{N=}, inside get_grads, err.shape={self.error.shape}, a_prev.T.shape={a_prev.T.shape}')
        self.dw = np.dot(self.error, a_prev.T) / N
        self.db = np.sum(self.error, axis=1, keepdims=True) / N
        return self.dw, self.db

    def get_error(self, back_err):
        derivative = self.g(self.z, derivative=True)
        if self.verbose: print(f'in get_error {self.layer_id}, {back_err.shape}, {derivative.shape}, {self.z.shape}, {self.g}')
        if self.verbose: print(self)
        self.error = back_err * derivative
        return self.error

    def backward(self):
        self.backerr = np.dot(self.w.T, self.error)
        return self.backerr

    def update_weights(self, lr):
        if self.verbose: print(f'{self.layer_id}: {self.w.shape=}, {self.dw.shape=}\n   {self.b.shape=}, {self.db.shape=}')
        self.w -= lr * self.dw
        self.b -= lr * self.db

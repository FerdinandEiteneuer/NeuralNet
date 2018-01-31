from kernel_initializer import kernels
import numpy as np


class Dense:
    def __init__(self, input_dim, output_dim, activation, kernel_init):
        self.g = activation
        self.initialize_params(input_dim, output_dim, kernel_init)    
        self.params = {'w': self.w, 'b': self.b}

    def initialize_params(self, input_dim, output_dim, kernel_init):
        shape = output_dim, input_dim
        self.w = kernels[kernel_init](shape)
        self.b = np.zeros((output_dim, 1))

    def forward(self, a):
        #print self.w.shape, a.shape, self.b.shape
        self.z = np.dot(self.w, a) + self.b
        self.a = self.g(self.z)
        self.params.update({'z': self.z, 'a': self.a})
        
        return self.a

    def grads(self, a_prev, N):
        self.dw = np.dot(self.error, a_prev.T) / N
        self.db = np.sum(self.error, axis=1, keepdims=True) / N

    def get_error(self, back_err):
        self.error = back_err * self.g(self.z, derivative=True)
        return self.error
    
    def backward(self):
        self.backerr = np.dot(self.w.T, self.error)
        return self.backerr

    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

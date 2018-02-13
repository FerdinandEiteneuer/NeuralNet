import numpy as np
from activations import linear

class Flatten:
    def __init__(self):
        self.g = linear

    def forward(self, a):
        self.n_H, self.n_W, self.c, self.m = a.shape
        self.flattened_shape = (self.n_H * self.n_W * self.c, self.m)
        self.z = self.a = np.reshape(a, self.flattened_shape)
        return self.a

    def grads(self, a_prev, N):
        pass

    def get_error(self, back_err):
        #just reshape the error delta^l_{jn} into ~ delta^l_{i,j,c,n)
        #print '\tinside flattened.get_error: back_err shape:', back_err.shape
        self.error = back_err #activation function is linear, hence derivative of it is just one, omit it.
        return self.error
    
    def backward(self):
        #dont need to do modify self.error here
        self.backerr = np.reshape(self.error , (self.n_H, self.n_W, self.c, self.m))
        return self.backerr

    def update(self, lr):
        pass

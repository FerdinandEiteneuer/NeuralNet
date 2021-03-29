import numpy as np
from functools import wraps

from .layer import Layer
from . import kernel_initializers
from . import activations

class Dense(Layer):
    def __init__(
            self,
            output_dim,
            activation,
            kernel_initializer=kernel_initializers.normal,
            kernel_regularizer=None,
            bias_regularizer=None,
            input_shape=None,
            p_dropout=1,
            verbose=False):

        super().__init__()

        self.verbose = verbose
        self.g = activation

        assert isinstance(output_dim, int)
        self.output_dim = (output_dim, )

        self.input_shape = input_shape

        self.kernel_initializer = kernel_initializer

        self.kernel_regularizer = kernel_regularizer(self, 'w') if kernel_regularizer else None
        self.bias_regularizer = bias_regularizer(self, 'b') if bias_regularizer else None

        assert 0 < p_dropout <= 1, f'{p_dropout=} is not in (0,1]. Note: p_dropout is the probabilty to keep a neuron'
        self.p_dropout = p_dropout

    def prepare_params(self, input_shape=None):
        if input_shape:
            self.shape = self.output_dim + input_shape
        else:
            self.shape = self.output_dim + (self.input_shape, )

        self.w = kernel_initializers.create(self.kernel_initializer, self.shape)
        self.dw = np.zeros(self.w.shape)

        self.b = np.zeros(self.output_dim + (1,) )
        self.db = np.zeros(self.b.shape)
        return self.output_dim


    def forward(self, x, mode='test'):

        self.batch_size = x.shape[-1]

        self.x = x
        self.z = np.dot(self.w, x) + self.b

        self.a = self.g(self.z)

        if mode == 'train':
            p = self.p_dropout
            self.dropout_mask = (np.random.rand(*self.a.shape) < p) / p
            self.a *= self.dropout_mask
        elif mode == 'gradient':
            self.a *= self.dropout_mask  # need to reuse the original dropout mask used for backprop, do not create a new one!
        elif mode == 'test':
            pass

        return self.a


    def backward(self, dout):

        #print(f'in backward {self.name}:', np.all(self.dropout_mask == 1))
        dout = self.dropout_mask * dout

        dlayer = self.g(self.z, derivative=True)

        if self.g is activations.softmax:
            dout = np.einsum('in,jin->jn', dout, dlayer)
        else:
            dout = dout * dlayer

        self.dw = np.dot(dout, self.x.T)
        self.db = np.sum(dout, axis=1, keepdims=True)


        if self.kernel_regularizer:
            self.dw += self.kernel_regularizer.derivative()
            assert np.all(self.kernel_regularizer.param == self.w)


        if self.bias_regularizer:
            self.db += self.bias_regularizer.derivative()

        dx = np.dot(self.w.T, dout)  # creating new upstream derivative
        return dx

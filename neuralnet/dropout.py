import numpy as np

from .layer import Layer
from . import activations



class Dropout(Layer):
    def __init__(self, p_dropout):

        super().__init__()

        assert 0 < p_dropout <= 1, f'{p_dropout=} is not in (0,1]. Note: p_dropout is the probabilty to keep a neuron'
        self.p_dropout = p_dropout


    def prepare_params(self, input_shape):
        self.output_dim = input_shape
        return self.output_dim


    def forward(self, x, mode='test'):
        self.x = x
        if mode == 'train':
            p = self.p_dropout
            self.dropout_mask = (np.random.rand(*self.x.shape) < p) / p
            a = self.x * self.dropout_mask
        elif mode == 'gradient':
            try:
                a = self.x * self.dropout_mask
            except AttributeError as e:
                print('ERROR: forwardmode \'gradient\' was used before forward mode \'train\'. '
                      'We need to reuse the dropoutmask used during the training pass.')
                raise e
        elif mode == 'test':
            a = self.x
        return a


    def backward(self, dout):
        dout = self.dropout_mask * dout
        return dout

"""
Convolution and its correspoding utility (MaxPooling2D and Flatten) Layers.
"""

import itertools
import sys

import numpy as np

from .layer import Layer
from . import kernel_initializers


def product(*args):
    """
    For iterating over multiple indices.

    >>> for x,y,z in product(3,4,5):
    >>>     pass

    is equivalent to

    >>> for x in range(3):
    >>>      for y in range(4):
    >>>         for z in range(5):
    >>>             pass
    """
    ranges = map(range, args)
    return itertools.product(*ranges)


class MaxPooling2D(Layer):
    """
    MaxPooling2D Layer.
    Note: currently the backward implementation is really slow.
    """

    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride


    def forward(self, a, mode='test'):

        self.x = a

        Hout, Wout, C = self.output_dim
        N = a.shape[-1]

        out = np.zeros((Hout, Wout, C, N))

        for h, w in product(Hout, Wout):
            hh = slice(h * self.stride, h * self.stride + self.pool_size)
            hh = slice(w * self.stride, w * self.stride + self.pool_size)

            x = a[hh, ww, ...]
            out[h, w, ...] = np.max(x, axis=(0,1))

        return out


    def prepare_params(self, input_shape):

        H, W, C = input_shape

        Hout = int(1 + (H - self.pool_size)/self.stride)
        Wout = int(1 + (W - self.pool_size)/self.stride)

        self.output_dim = (Hout, Wout, C)
        return self.output_dim


    def backward(self, dout):

        H, W, C, H = self.x.shape
        Hout, Wout, _, _ = dout.shape

        dx = np.zeros_like(self.x)

        for hout, wout, c, n in product(Hout, Wout, C, N):
            h = slice(self.stride * hout, self.stride * hout + self.pool_size)
            w = slice(self.stride * wout, self.stride * hout + self.pool_size)

            where = np.argwhere(self.x[h,w,c,n] == np.max(self.x[h,w,c,n]))
            xidx, yidx = where[0]

            h = self.stride * hout + xidx
            w = self.stride * wout + yidx

            dx[h, w, c, n] = dout[hout, wout, c, n]

        return dx


class Flatten(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, a, mode='test'):
        self.x = a
        nb_examples = a.shape[-1]
        self.a = a.reshape(-1, nb_examples)
        return self.a

    def prepare_params(self, input_shape):
        self.input_shape = input_shape
        self.output_dim = (np.product(input_shape), )
        return self.output_dim

    def backward(self, dout):
        return dout.reshape(*self.input_shape, -1)


class Conv2D(Layer):

    def __init__(
            self,
            filters,
            kernel_size,
            stride,
            padding,
            activation,
            input_shape=None,
            kernel_initializer=kernel_initializers.glorot_uniform,
            kernel_regularizer=None,
            bias_regularizer=None):

        super().__init__()
        """     model = Sequential([
                  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
                  MaxPooling2D(pool_size=pool_size),
                  Flatten(),
                  Dense(10, activation='softmax')])"""
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.g = activation
        self.input_shape = input_shape

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        if self.padding == 'valid':
            self.p = 0
        elif self.padding == 'same':
            self.p = int( (kernel_size - 1) / 2)
        else:
            raise ValueError(f'invalid padding: {padding}, must be one of "same" or "valid".')

        #pad only height and width (and not channels, trainexamples)
        self.pads = ((self.p, self.p), (self.p, self.p), (0,0), (0,0))

        if self.kernel_size % 2 == 0:
            #TODO  if this error is removed, introduce the floor function to self.p
            raise ValueError('invalid kernel size: {kernel_size}. must be an odd number')

    def prepare_params(self, input_shape=None):

        f, k = self.filters, self.kernel_size

        if not input_shape:
            input_shape = self.input_shape

        self.prev_width = input_shape[0]
        self.prev_height = input_shape[1]
        self.prev_f = input_shape[2]

        shape = (k, k, self.prev_f, f)

        self.w = kernel_initializers.create(self.kernel_initializer, shape)
        self.dw = np.zeros(self.w.shape)

        #self.b = np.zeros((f, 1))
        self.b = np.zeros((1,1,f,1))
        self.db = np.zeros(self.b.shape)

        output_height = int( (self.prev_height + 2*self.p - self.kernel_size)/self.stride + 1 )
        output_width = int( (self.prev_width + 2*self.p - self.kernel_size)/self.stride + 1 )

        if self.padding == 'same':
            assert output_height == self.prev_height
            assert output_width == self.prev_width

        self.output_dim = (output_height, output_width, f)
        return self.output_dim

    def forward(self, x, mode='test'):

        #print(f'forward {self.name}')
        #print(f'{x.shape=}')
        self.x = x

        if self.padding == 'same':
            a = np.pad(x, pad_width=self.pads, mode='constant', constant_values=0)
        else:
            a = x
        nb_examples = a.shape[-1]


        height, width, _ = self.output_dim

        self.z = np.zeros((height, width, self.filters, nb_examples))

        #a = a[:,:,:,np.newaxis,:] #(height, width, c_prev, m) -> (height, width, c_prev, AXIS, m)
        #W = self.w[...,np.newaxis] #(f, f, c_prev, c) -> (f, f, c_prev, c, AXIS)

        k = self.kernel_size

        for h, w in product(height, width):

            #print(h,w)
            image_part = a[h: h + k, w: w + k, ...]
            conv = np.einsum('ijkm,ijkn', self.w, image_part)
            self.z[h, w, ...] = conv# + self.b

        self.z += self.b
        self.a = self.g(self.z)

        #print(f'forward {self.name} done')
        return self.a


    def backward(self, dout):
        dlayer = self.g(self.z, derivative=True)

        dout = dout * dlayer

        self.dw = self.get_dw(dout)
        self.db = np.sum(dout, axis=(0,1,3), keepdims=True)

        if self.layer_id > 1:
            dx = self.get_dx(dout)
        else:
            dx = None
        return dx


    def get_dx(self, dout):

        # pad dout
        dout_p = 1
        dout_pads = [(dout_p, dout_p), (dout_p, dout_p), (0, 0), (0, 0)]
        dout_pad = np.pad(dout, dout_pads, mode='constant', constant_values=0)
        dout_pad = dout_pad[:, :,  np.newaxis, :, :]      # (H, W, 1, F, N)

        # w shape = (HH, WW, C, F)
        flipped_w = self.w[::-1, ::-1, :, :, np.newaxis]  # (HH, WW, C, F, 1)

        H, W = self.x.shape[:2]
        HH, WW = self.w.shape[:2]

        dx = np.zeros(self.x.shape)

        for h, w in product(H, W):

            dout_part = dout_pad[h:h+HH, w:w+WW, ...]
            conv = flipped_w * dout_part
            dx[h, w] = np.sum(conv, axis=(0, 1, 3))  # remaining axes: N, C

        return dx


    def get_dw(self, dout):
        dw = np.zeros(self.w.shape)

        H, W = self.x.shape[:2]
        HH, WW = self.w.shape[:2]

        x_pad = np.pad(self.x, self.pads, mode='constant', constant_values=0)

        for h, w in product(HH, WW):
            x_part = x_pad[h:h+H, w:w+W, ...]

            #print('dout   ', dout.shape)
            #print('x_part ', x_part.shape)
            conv = np.einsum('ijuw,ijvw', x_part, dout)

            #print(conv.shape)
            dw[h,w, ...] = conv
        return dw



if __name__ == '__main__':

    pass
    np.random.seed(231)
    N = 4
    C = 3
    H, W = 5, 5
    F = 2
    WW, HH = 3, 3

    x = np.random.randn(C, H, W, N)
    w = np.random.randn(F, C, WW, HH)
    b = np.random.randn(F, 1)
    dout = np.random.randn(F, H, W, N)  # (N, F, H, W)
    conv_param = {'stride': 1, 'pad': 1}

"""
Batchnormalization layer
"""

import itertools
import sys
import time

import numpy as np

from .layer import Layer
from . import kernel_initializers


class BatchNormalization(Layer):

    def __init__(self, epsilon=1e-3, momentum=0.99):
        super().__init__()

        self.ε = epsilon
        self.μ = momentum

        self.running_mean = 0
        self.running_var = 0

    @property
    def name(self):
        return f'batchnorm_{self.class_layer_id}'  # e.g: dense_1

    def prepare_params(self, input_shape):

        self.input_shape = input_shape
        self.output_dim = input_shape

        shape = self.output_dim + (1, )
        self.γ = np.ones(shape)   # scaling
        self.β = np.zeros(shape)  # shift

        self.running_mean = np.zeros(shape)
        self.running_var= np.zeros(shape)

        self.trainable_parameters = ['γ', 'β']
        self.nontrainable_parameters = ['running_mean', 'running_var']
        return self.output_dim


    def forward(self, a, mode='test'):
        self.x = a
        nb_examples = a.shape[-1]

        if mode == 'test':

            self.a = self.γ * (a - self.running_mean) / np.sqrt(self.running_var + self.ε) + self.β

        elif mode == 'train' or mode == 'gradient':

            self.sample_mean = a.mean(axis=-1, keepdims=True)
            self.sample_var = a.var(axis=-1, keepdims=True)

            self.running_mean = self.μ * self.running_mean + (1 - self.μ) * self.sample_mean
            self.running_var = self.μ * self.running_var + (1 - self.μ) * self.sample_var

            self.xhat = (a - self.sample_mean) / np.sqrt(self.sample_var + self.ε)
            self.a = self.γ * self.xhat + self.β

        return self.a

    def backward(self, dout):

        dx = np.zeros(self.x.shape)

        # convenience, make names shorter
        x = self.x
        N = self.x.shape[-1]
        sample_μ = self.sample_mean

        # calculate intermediate variables
        σ_norm = np.sqrt(self.sample_var + self.ε)
        sum_dout = np.sum(dout, axis=-1, keepdims=True)
        diff = x - sample_μ

        self.dx = self.γ / σ_norm * (
                + dout
                - 1/N * sum_dout
                - 1/(N * σ_norm**2) * diff * np.sum(dout * diff, axis=-1, keepdims=True)
                )


        self.dγ = np.sum(dout * self.xhat, axis=-1, keepdims=True)
        self.dβ = sum_dout

        return self.dx


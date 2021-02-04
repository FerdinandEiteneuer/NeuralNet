'''
Optimizers for gradient descent
'''

import numpy as np

class SGD:
    '''
    Stochastic Gradient Descent
    '''
    def __init__(self, learning_rate=0.01, momentum=0, bias_correction=True):
        self.lr = learning_rate
        self.beta_1 = momentum
        self.bias_correction = bias_correction

        self.decay = 0.04
        self.updates = 0

        # momentum
        self.mom_w = {}
        self.mom_b = {}

        self.network = None


    def prepare(self, network):
        self.network = network

        for layer in self.network:

            self.mom_w[layer.layer_id] = np.zeros(layer.w.shape)
            self.mom_b[layer.layer_id] = np.zeros(layer.b.shape)


    def __str__(self):
        s = f'Optimizer: SGD(lr={self.lr}, momentum={self.beta_1}, bias_correction={self.bias_correction})'
        return s


    def update_weights(self):
        μ = self.beta_1  # readability

        for layer in self.network:

            l = layer.layer_id

            self.mom_w[l] = μ * self.mom_w[l] + (1 - μ) * layer.dw
            self.mom_b[l] = μ * self.mom_b[l] + (1 - μ) * layer.db

            if self.bias_correction:
                correction = 1 - μ ** (1 + self.updates)
            else:
                correction = 1

            layer.w -= self.lr * self.mom_w[l] / correction
            layer.b -= self.lr * self.mom_b[l] / correction

        self.updates += 1


class Nadam:
    '''
    Nadam Optimizer. Combines nesterov, momentum, RMS prop step.
    Algorithm from https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    '''
    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999, eps=10**(-8), bias_correction=True):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.bias_correction = bias_correction

        self.updates = 1
        self.network = None

        #momenta for w and b
        self.mom_w = {}
        self.mom_b = {}

        #rms prop momentum
        self.rms_w = {}
        self.rms_b = {}

    def prepare(self, network):
        '''
        Initializes the momentum parameters.
        '''
        self.network = network

        for layer in self.network:
            l = layer.layer_id

            self.mom_w[l] = np.zeros(layer.w.shape)
            self.mom_b[l] = np.zeros(layer.b.shape)


            self.rms_w[l] = np.zeros(layer.w.shape)
            self.rms_b[l] = np.zeros(layer.b.shape)

    def __str__(self):
        s = f'Optimizer: Nadam(lr={self.lr}, beta_1={self.beta_1}, beta_2={self.beta_2}, eps={self.eps})'
        return s


    def update_weights(self):
        # set vars for readability
        μ = self.beta_1
        ν = self.beta_2
        t = self.updates
        ε = self.eps

        for layer in self.network:

            l = layer.layer_id
            #momentum
            self.mom_w[l] = μ * self.mom_w[l] + (1 - μ) * layer.dw
            self.mom_b[l] = μ * self.mom_b[l] + (1 - μ) * layer.db

            #RMS prop.
            self.rms_w[l] = ν * self.rms_w[l] + (1 - ν) * layer.dw**2
            self.rms_b[l] = ν * self.rms_b[l] + (1 - ν) * layer.db**2

            #Nesterov
            m_w = μ * self.mom_w[l] / (1 - μ**(t + 1)) \
                  + (1 - μ) * layer.dw / (1 - μ**t)

            m_b = μ * self.mom_b[l] / (1 - μ**(t + 1)) \
                  + (1 - μ) * layer.db / (1 - μ**t)


            n_w = ν * self.rms_w[l] / (1 - ν**t)
            n_b = ν * self.rms_b[l] / (1 - ν**t)

            #update parameters

            layer.w -= self.lr * m_w / np.sqrt(n_w + ε)
            layer.b -= self.lr * m_b / np.sqrt(n_b + ε)

        self.updates += 1

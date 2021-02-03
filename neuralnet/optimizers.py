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

    def __str__(self):
        s = f'Optimizer: SGD(lr={self.lr}, momentum={self.beta_1}, bias_correction={self.bias_correction})'
        return s

    def prepare(self, network):
        '''
        Initializes momentum terms.
        '''
        self.network = network
        self.updates = 0

        self.mom_w = {}
        self.mom_b = {}

        for layer in self.network:

            self.mom_w[layer.layer_id] = np.zeros(layer.w.shape)
            self.mom_b[layer.layer_id] = np.zeros(layer.b.shape)

    def update_weights(self):

        b1 = self.beta_1  # readability

        for layer in self.network:

            l = layer.layer_id

            self.mom_w[l] = b1 * self.mom_w[l] + (1 - b1) * layer.dw
            self.mom_b[l] = b1 * self.mom_b[l] + (1 - b1) * layer.db

            if self.bias_correction:
                correction = 1 - b1 ** (1 + self.updates)
            else:
                correction = 1

            layer.w -= self.lr * self.mom_w[l] / correction
            layer.b -= self.lr * self.mom_b[l] / correction

        self.updates += 1

class Nadam:
    '''Nadam Optimizer. Combines nesterov, momentum, RMS prop step'''
    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999, eps=10**(-8), network=None, bias_correction=True):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.bias_correction = bias_correction

        #momenta for w and bias
        self.mom_w = {}
        self.mom_b = {}

        #rms prop momentum
        self.rms_w = {}
        self.rms_b = {}
        #they get initialized with zeros
        for l in network.layers:
            self.mom_w[l] = np.zeros(network.w[l].shape)
            self.mom_b[l] = np.zeros(network.b[l].shape)

            self.rms_w[l] = np.zeros(network.w[l].shape)
            self.rms_b[l] = np.zeros(network.b[l].shape)

    def update(self, fc, epoch):
        for l in fc.layers:
            #momentum
            self.mom_w[l] = self.beta_1 * self.mom_w[l] + (1 - self.beta_1) * fc.dw[l]
            self.mom_b[l] = self.beta_1 * self.mom_b[l] + (1 - self.beta_1) * fc.db[l]

            #RMS prop.
            self.rms_w[l] = self.beta_2 * self.rms_w[l] + (1 - self.beta_2) * fc.dw[l]**2
            self.rms_b[l] = self.beta_2 * self.rms_b[l] + (1 - self.beta_2) * fc.db[l]**2

            #bias correction
            if self.bias_correction:
                epoch += 1
                self.rms_w[l] /= (1 - self.beta_2 ** epoch)
                self.rms_b[l] /= (1 - self.beta_2 ** epoch)

            #Nesterov + bias correcton, both for momentum

            if self.bias_correction:
                m_hat_w = self.beta_1 * self.mom_w[l] / (1 - self.beta_1**(epoch+1)) + (1 - self.beta_1) * fc.dw[l] / (1-self.beta_1 ** epoch)
                m_hat_b = self.beta_1 * self.mom_b[l] / (1 - self.beta_1**(epoch+1)) + (1 - self.beta_1) * fc.db[l] / (1-self.beta_1 ** epoch)
            else:
                 m_hat_w = self.beta_1 * self.mom_w[l] + (1 - self.beta_1) * fc.dw[l]
                 m_hat_b = self.beta_1 * self.mom_b[l] + (1 - self.beta_1) * fc.db[l]

            #update parameters
            fc.w[l] -= self.lr * m_hat_w/(np.sqrt(self.rms_w[l]) + self.eps)
            fc.b[l] -= self.lr * m_hat_b/(np.sqrt(self.rms_b[l]) + self.eps)


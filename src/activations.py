import numpy as np

'''
fill the dictionary "act" with the functions,
i.e act['relu'](np.array([-1,1])) works
'''
act = {}

def sigmoid(z, derivative=False):
    if not derivative:
        return 1/(1+np.exp(-z))
    else:
        sigma = sigmoid(z)
        return sigma * (1 - sigma)
act['sigmoid'] = sigmoid

def relu(z, derivative=False):
    if not derivative:
        z[z<0] = 0
        return z
    else:
        z[z<0] = 0
        z[z>0] = 1
        return z
act['relu'] = relu


def softmax(z, derivative=False):
    if not derivative:
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    else:
        exp_z = np.exp(z)
        N = np.sum(exp_z, axis=0, keepdims=True)
        return exp_z / N - exp_z ** 2 / N ** 2
        #return np.exp(z)/N - np.exp(2*z) / N ** 2
act['softmax'] = softmax

def binary_crossentropy(z, derivative=False):
    if not derivative:
        pass
    else:
        pass
act['binary_crossentropy'] = binary_crossentropy


def tanh(z, derivative=False):
    if not derivative:
        return np.tanh(z)
    else:
        return 1 - np.tanh(z) ** 2
act['tanh'] = tanh

def lrelu(z, alpha=0.1, derivative=False):
    if not derivative:
        z[z<0] = alpha * z
        return z
    else:
        z[z<0] = alpha
        z[z>0] = 1
        return z
act['lrelu'] = lrelu

def linear(z, derivative=False):
    if not derivative:
        return z
    else:
        return np.ones(z.shape)
act['linear'] = linear

def prelu(z):
    pass

def selu(z):
    pass

import numpy as np
from functools import wraps
import sys

def check_dims(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ypred_dim = args[0].shape[1]
        ytrue_dim = args[1].shape[1]
        assert ypred_dim == ytrue_dim
        return func(*args, **kwargs)
    return wrapper
'''
input lossfunction:
    dim ypred=(output_dim, number trainingexamples)
    dim ytrue=(output_dim, number trainingexamples)
output lossfunction:
    dim 


input derivative_loss function:
    dim ypred=(
'''

@check_dims
def _mse(ypred, ytrue, verbose=False):
    N = ypred.shape[1]
    diff = ypred - ytrue
    #print('in _mse', N, diff.shape)
    loss = np.sum(diff**2) / N
    if np.isnan(loss):
        print('LOSS IS NAN', loss, diff, ypred)
        sys.exit()
    return float(np.squeeze(loss))

def _derivative_mse(ypred, ytrue):
    diff = ypred - ytrue
    derivative = 2 * diff
    assert(derivative.shape == ypred.shape)
    return derivative
mse = {'function': _mse, 'derivative': _derivative_mse}


def mae(ypred, ytrue):
    assert(ypred.shape == ytrue.shape)
    N = ypred.shape[1]
    ratio = np.abs((ypred - ytrue) / ytrue)
    return ratio[0], float(np.squeeze(np.mean(ratio, axis=1, keepdims=True)))
mean_average_error = mae

def _cross_entropy(ypred, ytrue):
    assert(ypred.shape == ytrue.shape)
    N = ytrue.shape[1]
    return - np.sum(ytrue*np.log(ypred), keepdims=True)
def _derivative_cross_entropy(ypred, ytrue):
    derivative = - np.sum(ytrue / ypred)
    return derivative
cross_entropy = {'function': _cross_entropy, 'derivative': _derivative_cross_entropy}
    


def derivative(ypred, ytrue, func='mse', arg=''):
    func = 'derivative_' + func
    if arg == '':
        return eval(func)(ypred, ytrue)
    else:
        return eval(func)(ypred, ytrue, arg)

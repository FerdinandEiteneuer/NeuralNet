import numpy as np

'''
input lossfunction:
    dim ypred=(output_dim, number trainingexamples)
    dim ytrue=(output_dim, number trainingexamples)
output lossfunction:
    dim 


input derivative_loss function:
    dim ypred=(
'''

def _mse(ypred, ytrue, verbose=False):
    assert(ypred.shape == ytrue.shape)
    N = ypred.shape[1]
    diff = ypred - ytrue
    loss = np.sum(diff**2, axis=1, keepdims=True) / N
    return float(np.squeeze(loss))

def _derivative_mse(ypred, ytrue):
    diff = ypred - ytrue
    derivative = 2 * diff
    assert(derivative.shape == ypred.shape)
    return derivative

mse = (_mse, _derivative_mse)
mean_square_error = {'function': _mse, 'derivative': _derivative_mse}


def mae(ypred, ytrue):
    assert(ypred.shape == ytrue.shape)
    N = ypred.shape[1]
    ratio = np.abs((ypred - ytrue) / ytrue)
    return ratio[0], float(np.squeeze(np.mean(ratio, axis=1, keepdims=True)))
mean_average_error = mae

def _cross_entropy(ypred, ytrue):
    assert(ypred.shape == ytrue.shape)
    N = ytrue.shape[1]
    return -np.sum( ytrue*np.log(ypred))
def _derivative_cross_entropy(ypred, ytrue):
    diff = - ytrue / ypred

    


def derivative(ypred, ytrue, func='mse', arg=''):
    func = 'derivative_' + func
    if arg == '':
        return eval(func)(ypred, ytrue)
    else:
        return eval(func)(ypred, ytrue, arg)

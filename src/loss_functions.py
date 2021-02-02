import numpy as np
from functools import wraps
import sys

class Lossfunction():
    def __call__(self, ypred, ytrue, average_examples=None, derivative=False):
        assert ypred.shape == ytrue.shape, f'function inputs are not equal: {ypred.shape=}, {ytrue.shape=}'
        if derivative:
            if average_examples is None:
                average_examples = False
            return self.derivative(ypred, ytrue, average_examples)
        else:
            if average_examples is None:
                average_examples = True
            return self.function(ypred, ytrue, average_examples)

class _crossentropy(Lossfunction):

    @staticmethod
    def function(ypred, ytrue, average_examples=True):
        if average_examples:
            N = ytrue.shape[1]
            loss = - 1 / N * np.sum(ytrue*np.log(ypred))
            return loss
        else:
            loss_per_example = -np.sum(ytrue * np.log(ypred), axis=0, keepdims=True)
            return loss_per_example

    @staticmethod
    def derivative(ypred, ytrue, average_examples=False):
        if average_examples:
            N = ypred.shape[1]  # num datapoints
            der_loss = -1 / N * np.sum(ytrue / ypred)
            return der_loss
        else:
            der_loss = - ytrue / ypred
            return der_loss

def crossentropy(ypred, ytrue, average_examples=True, derivative=False):

    if derivative:
        return _crossentropy.derivative(ypred, ytrue, average_examples=average_examples)
    else:
        return _crossentropy.function(ypred, ytrue, average_examples=average_examples)


class mse_(Lossfunction):

    @staticmethod
    def function(ypred, ytrue, average_examples=True):
        diff = ypred - ytrue
        loss_per_example = np.sum(diff**2, axis=0, keepdims=True)
        nb_examples = ytrue.shape[1]
        if average_examples:
            total_loss = np.sum(loss_per_example)/nb_examples
            assert total_loss.shape == ()
            return total_loss
        else:
            assert loss_per_example.shape == (1, nb_examples)
            return loss_per_example

    @staticmethod
    def derivative(ypred, ytrue, average_examples=False):
        if not average_examples:
            derivative = 2 * (ypred - ytrue)
            return derivative
        else:
            raise NotImplementedError

def check_dims(lossfct):
    """
    Checks the dimensions of the input and outputs to the lossfunction.
    """
    @wraps(lossfct)
    def wrapper(ypred, ytrue, derivative=False, average_examples=True):
        loss = lossfct(ypred, ytrue, average_examples, derivative)
        if average_examples:
            assert loss.shape == (), f'dimensions of loss ({loss.shape} is false. should be (), i.e just a number'
        else:
            assert loss.shape == (1, ytrue_dim), f'dimensions of loss is false, should be (1, {ytrue_dim}), i.e the loss per example'
        return loss
    return wrapper


def mean_squared_error(ypred, ytrue, derivative=False, average_examples=True):
    '''
    Mean Squared Error.
    If derivative is False, the function optionally computes the complete average loss.
    If derivative is True, the function computes the matrix with idx (class, train_example).
    '''
    ypred_dim = ypred.shape[1]
    ytrue_dim = ytrue.shape[1]
    assert ypred_dim == ytrue_dim, f'dimensions of ytrue ({ytrue.shape}) and ypred ({ypred.shape}) don\'t match'

    diff = ypred - ytrue

    if not derivative:
        loss_per_example = np.sum(diff**2, axis=0, keepdims=True)

        if average_examples:
            nb_examples = ypred.shape[1]
            total_loss = np.sum(loss_per_example) / nb_examples
            assert total_loss.shape == (), f'dimensions of loss ({loss.shape} is false. should be (), i.e just a number'
            return total_loss
        else:
            assert loss_per_example.shape == (1, ytrue_dim), f'dimensions of loss is false, should be (1, {ytrue_dim}), i.e the loss per example'
            return loss_per_example

    else:
        loss_per_example_and_class = 2 * diff
        if not average_examples:
            return loss_per_example_and_class
        else:
            raise NotImplementedError

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

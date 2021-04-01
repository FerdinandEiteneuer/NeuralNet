import numpy as np
from functools import wraps
import sys

class crossentropy:

    @staticmethod
    def function(ypred, ytrue, average_examples=True, ε=1e-12):

        ypred = np.clip(ypred, ε, 1)

        if average_examples:
            N = ytrue.shape[1]
            loss = - 1 / N * np.sum(ytrue*np.log(ypred))
            return loss
        else:
            loss_per_example = -np.sum(ytrue * np.log(ypred), axis=0, keepdims=True)
            return loss_per_example

    @staticmethod
    def derivative(ypred, ytrue, average_examples=False, ε=1e-12):

        ypred = np.clip(ypred, ε, 1)

        if average_examples:
            N = ypred.shape[1]  # num datapoints
            der_loss = - 1 / N * np.sum(ytrue / ypred)
            return der_loss
        else:
            der_loss = - ytrue / ypred
            return der_loss

class mse:

    @staticmethod
    def function(ypred, ytrue, average_examples=True):
        diff = ypred - ytrue
        loss_per_example = np.sum(diff**2, axis=0, keepdims=True)
        nb_examples = ytrue.shape[1]
        if average_examples:
            total_loss = np.sum(loss_per_example) / nb_examples
            assert total_loss.shape == ()
            return total_loss
        else:
            assert loss_per_example.shape == (1, nb_examples)
            return loss_per_example

    @staticmethod
    def derivative(ypred, ytrue, average_examples=False):
        derivative_per_example = 2 * (ypred - ytrue)
        if not average_examples:
            return derivative_per_example
        else:
            N = ypred.shape[-1]
            derivative_total = 1 / N * np.sum(derivative_per_example)
            return derivative_total



if __name__ == '__main__':

    classes = 4
    examples = 10

    x = np.random.randn(classes, examples)
    y_flat = np.random.randint(0, classes, size=examples)
    y = np.eye(classes)[y_flat].T


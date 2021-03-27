'''
Credit dataset to google
download url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
'''

import numpy as np
import sys
import os

DTYPE = 'float64'


def reduce_data(data, factor=0.2):

    assert 0 < factor <= 1

    size = data.shape[-1]
    new_size = int(factor * size)

    return data[..., :new_size]

def _reduce_data(xtrain, xtest, ytrain, ytest, factor=0.2):

    size_train = xtrain.shape[-1]
    nb_train = int(factor * size_train)

    size_test = xtest.shape[-1]
    nb_test = int(factor * size_test)

    xtrain = xtrain[...,:nb_train]
    ytrain = ytrain[...,:nb_train]

    xtest = xtest[...,:nb_test]
    ytest = ytest[...,:nb_test]

    return xtrain, xtest, ytrain, ytest

def one_hot(y, dim=10):
    matrix = np.eye(dim, dtype=DTYPE)[y]
    matrix = matrix.T
    return matrix

def load(fraction_of_data=1):

    datadir = os.path.dirname(__file__)
    path = os.path.join(datadir, 'mnist.npz')
    print(f'Loading data from {path=}')

    with np.load(path) as data:

        # load from disk
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

        # transform into correct format
        y_train = one_hot(y_train)
        y_test = one_hot(y_test)

        # put examples last
        x_train = np.transpose(x_train, (1, 2, 0))
        x_test = np.transpose(x_test, (1, 2, 0))

        # introduce 1 channel (i.e x_train.shape = (28, 28, 1, 60000)
        x_train = x_train[:,:,np.newaxis,:]
        x_test = x_test[:,:,np.newaxis,:]

        x_train = x_train.astype(DTYPE)
        x_test = x_test.astype(DTYPE)

        # preprocessing
        x_train /= 255.0
        x_test /= 255.0

        if fraction_of_data != 1:
            x_train = reduce_data(x_train, factor=fraction_of_data)
            x_test = reduce_data(x_test, factor=fraction_of_data)

            y_train = reduce_data(y_train, factor=fraction_of_data)
            y_test = reduce_data(y_test, factor=fraction_of_data)

        return x_train, x_test, y_train, y_test

if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load()


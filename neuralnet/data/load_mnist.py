'''
Credit dataset to google
download url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
'''

import numpy as np
import sys
import os

DTYPE = 'float32'


def reduce_data(data, factor=0.2):

    assert 0 < factor <= 1

    size = data.shape[-1]
    new_size = int(factor * size)

    return data[..., :new_size]

def one_hot(y, dim=10):
    matrix = np.eye(dim, dtype=DTYPE)[y]
    matrix = matrix.T
    return matrix

def load(fraction_of_data=1, nb_points=None, preprocess=True):

    if fraction_of_data < 1 and nb_points is not None:
        raise ValueError('can only use \'nb_points\' or \'fraction_of_data\', not both')

    datadir = os.path.dirname(__file__)
    path = os.path.join(datadir, 'mnist.npz')
    print(f'Loading data from {path=}')

    with np.load(path) as data:

        # load from disk
        xtrain = data['x_train']
        ytrain = data['y_train']
        xtest = data['x_test']
        ytest = data['y_test']

        # transform into correct format
        ytrain = one_hot(ytrain)
        ytest = one_hot(ytest)

        # put examples last
        xtrain = np.transpose(xtrain, (1, 2, 0))
        xtest = np.transpose(xtest, (1, 2, 0))

        # introduce 1 channel (i.e xtrain.shape = (28, 28, 1, 60000)
        xtrain = xtrain[:,:,np.newaxis,:]
        xtest = xtest[:,:,np.newaxis,:]

        xtrain = xtrain.astype(DTYPE)
        xtest = xtest.astype(DTYPE)

        if preprocess:
            xtrain = (xtrain - xtrain.mean()) / xtrain.std()
            xtest = (xtest - xtest.mean()) / xtest.std()

        if nb_points is not None:
            total_nb = xtrain.shape[-1]
            fraction_of_data = nb_points / total_nb

        if fraction_of_data != 1:
            xtrain = reduce_data(xtrain, factor=fraction_of_data)
            xtest = reduce_data(xtest, factor=fraction_of_data)

            ytrain = reduce_data(ytrain, factor=fraction_of_data)
            ytest = reduce_data(ytest, factor=fraction_of_data)

        return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':

    xtrain, xtest, ytrain, ytest = load()


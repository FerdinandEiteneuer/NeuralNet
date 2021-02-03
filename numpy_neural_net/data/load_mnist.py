'''
Credit dataset to google
download url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
'''

import numpy as np
import sys
import os

DTYPE = 'float64'


def one_hot(y):
    dim = 10
    matrix = np.eye(dim, dtype=DTYPE)[y]
    matrix = matrix.T
    return matrix

def load():

    datadir = os.path.dirname(__file__)
    path = os.path.join(datadir, 'mnist.npz')
    print(f'{path=}  --- file')

    with np.load(path) as data:
        # load from disk
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

        # transform into correct format
        y_train = one_hot(y_train)
        y_test = one_hot(y_test)

        x_train = np.transpose(x_train, (1, 2, 0))
        x_test = np.transpose(x_test, (1, 2, 0))

        x_train = x_train.astype(DTYPE)
        x_test = x_test.astype(DTYPE)

        x_train /= 255.0
        x_test /= 255.0

        #xtrain, xtest, ytrain, ytest

        print(f'{y_train.shape=}')

        return x_train, x_test, y_train, y_test

if __name__ == '__main__':

    x_train, x_test, y_train, y_test = load()


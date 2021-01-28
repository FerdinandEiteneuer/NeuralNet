import numpy as np
import os
import sys

def load_mnist(data_dir):

    try:
        xtrain = np.load(os.path.join(data_dir, 'x_train.npy'))
        xtest = np.load(os.path.join(data_dir, 'x_test.npy'))

        ytrain = np.load(os.path.join(data_dir, 'y_train.npy'))
        ytest = np.load(os.path.join(data_dir, 'y_test.npy'))
    except FileNotFoundError:
        print('could not find mnist dataset')
        sys.exit(-1)

    return xtrain, xtest, ytrain, ytest

"""
import numpy as np
import os
import sys

import data
print(f'{dir(data)=}')

def load(data_dir):

    try:
        xtrain = np.load(os.path.join(data_dir, 'x_train.npy'))
        xtest = np.load(os.path.join(data_dir, 'x_test.npy'))

        ytrain = np.load(os.path.join(data_dir, 'y_train.npy'))
        ytest = np.load(os.path.join(data_dir, 'y_test.npy'))
    except FileNotFoundError:
        pass

    return xtrain, xtest, ytrain, ytest
"""

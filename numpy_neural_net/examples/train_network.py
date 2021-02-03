'''
Trains a neural network with mnist data
'''

import numpy as np
import sys
import os
#np.warnings.filterwarnings('ignore')

import numpy_neural_net as nnn

from numpy_neural_net.network import Network
from numpy_neural_net.dense import Dense
from numpy_neural_net.activations import relu, sigmoid, linear, tanh, softmax, lrelu
from numpy_neural_net.loss_functions import mse, crossentropy
from numpy_neural_net.data import load_mnist


if __name__ == '__main__':

    ########
    # DATA #
    ########
    xtrain, xtest, ytrain, ytest = load_mnist.load()

    xtrain = np.resize(xtrain, (28**2, 60000))
    xtest = np.resize(xtest, (28**2, 10000))

    ###########
    # NETWORK #
    ###########

    input_dim = xtrain.shape[0]
    output_dim = ytrain.shape[0]

    kernel_init= 'normal'
    depth = 200

    model = Network(verbose=False)

    model.add(Dense(input_dim, depth, relu, kernel_init))
    model.add(Dense(depth, depth, relu, kernel_init))
    model.add(Dense(depth, output_dim, softmax, kernel_init))

    model.compile(loss = crossentropy, lr = 1*10**(-1))

    loss = model.get_loss(xtrain, ytrain, average_examples=True)

    model.fit(
        x=xtrain,
        y=ytrain,
        epochs=500,
        batch_size=1000,
        validation_data=(xtest, ytest),
        gradients_to_check_each_epoch=5,
        verbose=True
    )

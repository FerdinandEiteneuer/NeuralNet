'''
Trains a neural network with mnist data
'''

import numpy as np
import sys
import os

from neuralnet.network import Network
from neuralnet.dense import Dense
from neuralnet.activations import relu, sigmoid, linear, tanh, softmax, lrelu
from neuralnet.loss_functions import mse, crossentropy
from neuralnet.optimizers import SGD

from neuralnet.data import load_mnist

if __name__ == '__main__':

    #np.random.seed(123)  # reproducibility
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
    momentum = 0.9

    model = Network(verbose=False)

    model.add(Dense(input_dim, depth, relu, kernel_init))
    model.add(Dense(depth, depth, relu, kernel_init))
    model.add(Dense(depth, output_dim, softmax, kernel_init))


    opt = SGD(learning_rate=10**(-1), bias_correction=True, momentum=momentum)
    model.compile(loss = crossentropy, optimizer=opt)

    print(model)
    loss = model.get_loss(xtrain, ytrain, average_examples=True)
    print(f'Sanity check: Initial{loss=:.5f}')

    model.fit(
        x=xtrain,
        y=ytrain,
        epochs=12,
        batch_size=32,
        validation_data=(xtest, ytest),
        gradients_to_check_each_epoch=5,
        verbose=True
    )

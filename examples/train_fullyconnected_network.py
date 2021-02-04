'''
Trains the neural numpy net with mnist data.
'''

import numpy as np
import sys
import os

from neuralnet.network import Sequential
from neuralnet.dense import Dense
from neuralnet.activations import relu, sigmoid, linear, tanh, softmax, lrelu
from neuralnet.loss_functions import mse, crossentropy
from neuralnet.optimizers import SGD, Nadam
from neuralnet.regularizers import L1, L2, L1_L2

from neuralnet.data import load_mnist

np.random.seed(123)  # reproducibility

if __name__ == '__main__':

    xtrain, xtest, ytrain, ytest = load_mnist.load()

    xtrain = np.resize(xtrain, (28**2, 60000))
    xtest = np.resize(xtest, (28**2, 10000))

    input_dim = xtrain.shape[0]
    output_dim = ytrain.shape[0]

    kernel_init= 'normal'
    depth = 100

    model = Sequential()

    model.add(Dense(input_dim, depth, relu, kernel_init, kernel_regularizer=L2(1e-4)))
    model.add(Dense(depth, depth, relu, kernel_init, bias_regularizer=L1_L2(1e-3, 1e-3)))
    model.add(Dense(depth, output_dim, softmax, kernel_init))


    sgd = SGD(learning_rate=2*10**(-1), bias_correction=True, momentum=0.9)
    nadam = Nadam(learning_rate=0.7*10**(-3), beta_1=0.9, beta_2=0.999, eps=10**(-8))

    model.compile(loss = crossentropy, optimizer=nadam)
    print(model)

    # initial sanity check. print out loss + regularization loss
    model.get_loss(xtrain, ytrain, average_examples=True, verbose=True)


    model.fit(
        x=xtrain,
        y=ytrain,
        epochs=120,
        batch_size=500,
        validation_data=(xtest, ytest),
        gradients_to_check_each_epoch=3,
        verbose=True
    )

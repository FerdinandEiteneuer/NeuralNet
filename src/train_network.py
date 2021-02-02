'''
Trains a neural network with mnist data
'''

import numpy as np
import sys
import os
#np.warnings.filterwarnings('ignore')

from network import Network
from dense import Dense
from activations import relu, sigmoid, linear, tanh, softmax, lrelu
from loss_functions import mse, crossentropy
import loss_functions
import misc
import mnist

def prepare_data(datapath):

    xtrain, xtest, ytrain, ytest = mnist.load_mnist(datapath)
    xtrain /= 255.0
    xtest /= 255.0

    xtrain.resize((28**2, 60000))
    xtest.resize((28**2, 10000))

    return xtrain, xtest, ytrain, ytest

if __name__ == '__main__':

    ########
    # DATA #
    ########
    datapath = os.environ['HOME'] + '/python/neuralnet/data'
    xtrain, xtest, ytrain, ytest = prepare_data(datapath)

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

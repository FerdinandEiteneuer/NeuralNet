'''
Trains a neural network with mnist data
'''

import numpy as np
import sys
import os
np.warnings.filterwarnings('ignore')
np.random.seed(1)

from network import Network
from dense import Dense
from conv2d import Conv2D
from flatten import Flatten
from activations import relu, sigmoid, linear, tanh, softmax, lrelu
from loss_functions import mse, cross_entropy
import loss_functions
import misc
import mnist

if __name__ == '__main__':

    try:
        epochs = int(sys.argv[1])
    except IndexError:
        epochs = 2

    xtrain, xtest, ytrain, ytest = mnist.load_mnist(os.environ['HOME'] + '/python/neuralnet/data')

    xtrain.resize((28**2, 60000))
    xtest.resize((28**2, 10000))

    model = Network(verbose=False)

    input_dim = xtrain.shape[0]
    output_dim = ytrain.shape[0]


    kernel_init = 'glorot_uniform'
    kernel_init= 'normal'
    act = relu

    depth = 30
    model.add(Dense(input_dim, depth, relu, kernel_init))
    #model.add(Dense(depth, depth, relu, kernel_init))
    #model.add(Dense(depth, depth, relu, kernel_init))
    model.add(Dense(depth, output_dim, softmax, kernel_init))
    model.compile(loss = mse, lr = 0.001)

    '''

    #filter1 = (5,5,1,3)
    #filter2 = (2,2,3,5)

    filter1 = (4, 4, 1, 6)
    filter2 = (3, 3, 6, 12)
    flat_depth = filter2[-1] * 36

    model.add(Conv2D(filter1,2,0, act, kernel_init))
    model.add(Conv2D(filter2,2,0, act, kernel_init))
    model.add(Flatten())
    model.add(Dense(flat_depth, output_dim, softmax, kernel_init))
    model.compile(loss = cross_entropy, lr = 0.0005)
    '''
    l = model.layers

    printeach = 1
    for epoch in range(1, epochs):
        model.lr *= 0.995
        minibatches = misc.minibatches(xtrain, ytrain, batch_size=16)
        for m, minibatch in enumerate(minibatches):
            xmini, ymini = minibatch

            model.train_on_batch(xmini, ymini)

            loss = model.get_loss(xmini, ymini)
            print('epoch', epoch, 'minibatch', m, 'loss',loss)

            #ratio, grad_manual, grad_backprop = model.gradient_check(xmini, ymini, 1, (0,0,0,0))

            #print('gradient checking:', ratio, 'manual calculation', grad_manual, 'backprop', grad_backprop)

        model.epoch += 1


    pred = model.predict(xtrain)
    print(pred)
    print(ytrain)

    ratio, meanerr = loss_functions.mae(pred, ytrain)
    print('mae', meanerr)
    print('ratios', ratio)

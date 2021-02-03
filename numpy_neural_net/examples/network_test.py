'''
Trains a neural network with mnist data
'''
from tqdm import tqdm

import numpy as np
import sys
import os
#np.warnings.filterwarnings('ignore')
np.random.seed(2)
np.set_printoptions(precision=3, linewidth=180)
np.seterr(all='raise')

from network import Network
from dense import Dense
from conv2d import Conv2D
from flatten import Flatten
from activations import relu, sigmoid, linear, tanh, softmax, lrelu
from loss_functions import mse, crossentropy
import loss_functions
import misc
import mnist

if __name__ == '__main__':

    try:
        epochs = int(sys.argv[1])
    except IndexError:
        epochs = 2

    datapath = os.environ['HOME'] + '/python/neuralnet/data'
    xtrain, xtest, ytrain, ytest = mnist.load_mnist(datapath)

    ytrain_labels = np.argmax(ytrain, axis=0)
    ytest_labels = np.argmax(ytest, axis=0)


    xtrain /= 255.0
    xtest /= 255.0

    xtrain.resize((28**2, 60000))
    xtest.resize((28**2, 10000))

    Ntrain=60000
    Ntest=10000


    model = Network(verbose=False)

    input_dim = xtrain.shape[0]
    output_dim = ytrain.shape[0]


    kernel_init = 'glorot_uniform'
    kernel_init= 'normal'

    depth = 10
    #model.add(Dense(input_dim, output_dim, relu, kernel_init))
    model.add(Dense(input_dim, depth, relu, kernel_init))
    #model.add(Dense(depth, depth, relu, kernel_init))
    #model.add(Dense(depth, depth, relu, kernel_init))
    model.add(Dense(depth, output_dim, softmax, kernel_init))

    model.compile(loss = crossentropy, lr = 1*10**(-1))

    loss = model.get_loss(xtrain, ytrain, average_examples=True)
    print(f'basic loss test: {loss}')

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

    layer = 1
    idx = (2,3)
    goodness = None

    printeach = 1
    for epoch in range(1, epochs):
        losses = []
        model.lr *= 0.99
        minibatches = misc.minibatches(xtrain, ytrain, batch_size=1000)
        for m, minibatch in enumerate(minibatches):
            x, y = minibatch
            #print(y)

            model.train_on_batch(x, y)

            loss = model.get_loss(x, y)
            losses.append(loss)
            #print(f'{epoch=}, {m=}, {loss=:4f}')
            #print(f'{model[layer].w[idx]=:.3f}')
            if m == 1:
                model.complete_gradient_check(x, y, eps=10**(-6))
                goodness = model.gradient_checks(x=x, ytrue=y, eps=10**(-6), checks=15)
                sys.exit()
                pass

            model.update_weights()
            #model.complete_gradient_check(x, y, eps=10**(-6))
            #sys.exit()

        loss = np.mean(loss)

        a_train = model(xtrain)
        a_test = model(xtest)

        ytrain_pred = np.argmax(a_train, axis=0)
        ytest_pred = np.argmax(a_test, axis=0)

        train_correct = np.sum(ytrain_pred == ytrain_labels)
        test_correct = np.sum(ytest_pred == ytest_labels)

        val_loss = model.get_loss(xtest, ytest)
        print(f'{epoch=}, {loss=:.3f}, {val_loss=:.3f}, train: {train_correct}/{Ntrain}, test: {test_correct}/{Ntest}, gradcheck: {goodness}')
        model.epoch += 1



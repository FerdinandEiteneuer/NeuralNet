'''
Trains the neural numpy net with mnist data.
'''

import numpy as np
import sys
import os


from neuralnet.network import Sequential
from neuralnet.dropout import Dropout
from neuralnet.dense import Dense
from neuralnet.conv2d import Conv2D, Flatten
from neuralnet.activations import relu, sigmoid, linear, tanh, softmax, lrelu
from neuralnet.loss_functions import mse, crossentropy
from neuralnet.optimizers import SGD, Nadam
from neuralnet.regularizers import L1, L2, L1_L2
from neuralnet.kernel_initializers import normal, glorot_uniform


from neuralnet.data import load_mnist

#np.random.seed(123)  # reproducibility
np.set_printoptions(precision=4, linewidth=120)

if __name__ == '__main__':

    xtrain, xtest, ytrain, ytest = load_mnist.load(fraction_of_data=1)

    input_dim = xtrain.shape[0]
    output_dim = ytrain.shape[0]

    kernel_init = glorot_uniform

    model = Sequential()

    model.add(Conv2D(
        kernel_size=3,
        filters=15,
        stride=1,
        padding='same',
        input_shape=(28, 28, 1),
        activation=relu,
        )
    )

    model.add(Flatten())
    model.add(Dense(64, relu, kernel_initializer=kernel_init))
    #model.add(Dropout(1))
    model.add(Dense(output_dim, softmax, kernel_initializer=kernel_init))


    sgd = SGD(learning_rate=2*10**(-1), bias_correction=True, momentum=0.9)
    nadam = Nadam(learning_rate=0.3*10**(-3), beta_1=0.9, beta_2=0.999, eps=10**(-8))

    model.compile(loss = crossentropy, optimizer=nadam)
    print(model)

    #c =  model[1]
    #x = xtrain[..., :250]
    #out=c(x)

    #sys.exit()
    # initial sanity check. print out loss + regularization loss
    # model.loss(xtrain, ytrain, verbose=True)  # calculation may not fit into memory

    model.fit(
        x=xtrain,
        y=ytrain,
        epochs=10,
        batch_size=100,
        validation_data=(xtest, ytest),
        #gradients_to_check_each_epoch=10,
        verbose=True
    )

from network import Network
from dense import Dense
from conv2d import Conv2D
from activations import relu, sigmoid
from loss_functions import mean_square_error as mse
import loss_functions
import misc

import numpy as np, sys
epochs = int(sys.argv[1])


x, y = misc.generate_test_data(10000)
#x, y = misc.load_california_housing()
xtrain, ytrain, xtest, ytest = misc.split(x, y, split_portion=.8)
minibatches = misc.minibatches(xtrain, ytrain, size=1000)


model = Network(loss = mse)

depth = 15
kernel_init = 'glorot_uniform'

model.add(Dense(1, depth, relu, kernel_init))
model.add(Dense(depth, depth, relu, kernel_init))
model.add(Dense(depth, 1, relu, kernel_init))
model.compile(lr = 0.05)


printeach = 10
for epoch in range(1, epochs):
    for m, minibatch in enumerate(minibatches):
        #print 'minibatch', m
        model.train_step(minibatch)

    if epoch % printeach == 0:
        xmini, ymini = minibatch
        print epoch, model.get_loss(xmini, ymini)

pred = model.predict(xtrain) 
print pred
print ytrain

ratio, meanerr = loss_functions.mae(pred, ytrain)
print 'mae', meanerr
print 'ratios', ratio
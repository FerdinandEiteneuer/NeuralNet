from network import Network
from dense import Dense
from conv2d import Conv2D
from flatten import Flatten
from activations import relu, sigmoid, linear, tanh, softmax, lrelu
from loss_functions import mse, cross_entropy
import loss_functions
import misc

import numpy as np, sys
np.warnings.filterwarnings('ignore')
np.random.seed(1)
epochs = int(sys.argv[1])


#x, y = misc.generate_test_data(10000)
#x, y = misc.load_california_housing()
#xtrain, ytrain, xtest, ytest = misc.split(x, y, split_portion=.8)

xtrain, ytrain = misc.load_mnist()
#xtrain, ytrain = misc.generate_conv_test_data()
    
model = Network(verbose=False)

input_dim = xtrain.shape[0]
output_dim = ytrain.shape[0]


kernel_init = 'glorot_uniform'
kernel_init= 'normal'
act = relu

'''
depth = 10
model.add(Dense(input_dim, depth, relu, kernel_init))
#model.add(Dense(depth, depth, relu, kernel_init))
#model.add(Dense(depth, depth, relu, kernel_init))
model.add(Dense(depth, output_dim, kernel_init))
model.compile(loss = mae, lr = 0.00001)
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
l = model.layers

printeach = 1
for epoch in range(1, epochs):
    model.lr *= 0.995
    minibatches = misc.minibatches(xtrain, ytrain, size=10)
    for m, minibatch in enumerate(minibatches):
        model.train_step(minibatch)
        xmini, ymini = minibatch
        loss = model.get_loss(xmini, ymini)
        print 'epoch',epoch, 'minibatch',m, 'loss',loss[0,0]
        ratio, grad_manual, grad_backprop = model.gradient_check(xmini, ymini, 1, (0,0,0,0))
        print 'gradient checking:', ratio, 'manual calculation', grad_manual, 'backprop', grad_backprop
        sys.exit()
pred = model.predict(xtrain) 
print pred
print ytrain

ratio, meanerr = loss_functions.mae(pred, ytrain)
print 'mae', meanerr
print 'ratios', ratio


import numpy as np
import sys
import os

from neuralnet.network import Sequential
from neuralnet.batchnorm import BatchNormalization

from neuralnet.dense import Dense
from neuralnet.activations import relu, sigmoid, linear, tanh, softmax, lrelu
from neuralnet.loss_functions import mse, crossentropy
from neuralnet.optimizers import SGD, Nadam
from neuralnet.regularizers import L1, L2, L1_L2
from neuralnet.kernel_initializers import normal, glorot_uniform
from neuralnet.conv2d import Flatten, Conv2D
from neuralnet.data import load_mnist

def print_mean_std(arr, axis=1):
    print('   means: ', arr.mean(axis=axis))
    print('   stds:  ', arr.std(axis=axis))

np.set_printoptions(linewidth=120, precision=4, suppress=True)



######################
# TEST 1             #
# train forward test #
######################

np.random.seed(231)
N, D1, D2, D3 = 200, 50, 60, 3
X = np.random.randn(N, D1)
W1 = np.random.randn(D1, D2)
W2 = np.random.randn(D2, D3)

a = np.maximum(0, X.dot(W1)).dot(W2)

print('Before batch normalization:')
print_mean_std(a, axis=0)

gamma = np.ones((D3, ))
beta= np.zeros((D3, ))



bn = BatchNormalization(epsilon=1e-5, momentum=0.9)
bn.prepare_params(input_shape=3)

out = bn(a.T, mode='train').T

print('After batch normalization (gamma=1, beta=0)')
print_mean_std(out, axis=0)


bn.γ = np.array([1,2,3]).reshape(3,1)
bn.β = np.array([11,12,13]).reshape(3,1)


out = bn(a.T, mode='train').T
print('After batch normalization (gamma=[1,2,3], β=[11,12,13]')
print_mean_std(out, axis=0)




####################
# TEST 2           #
# testtime forward #
####################
bn.prepare_params(input_shape=3)
np.random.seed(231)
W1 = np.random.randn(D1, D2)
W2 = np.random.randn(D2, D3)

for t in range(50):
    X = np.random.randn(N, D1)
    a = np.maximum(0, X.dot(W1)).dot(W2)
    out = bn(a.T, mode='train').T


X = np.random.randn(N, D1)
a = np.maximum(0, X.dot(W1)).dot(W2)
out = bn(a.T, mode='test').T

print('After batch normalization (test-time):')
print_mean_std(out, axis=0)

print('''Correct result test-time:
   means:  [-0.03927354 -0.04349152 -0.10452688]
   stds:   [1.01531428 1.01238373 0.97819988]
''')



#################
# TEST 3        # 
# backward pass #
#################

np.random.seed(231)
N, D = 4, 5
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)

bn = BatchNormalization(epsilon=1e-5, momentum=0.9)
bn.prepare_params(input_shape=D)
bn.γ = gamma.reshape(bn.γ.shape)
bn.β = beta.reshape(bn.β.shape)


out = bn(x.T, mode='train').T
dx = bn.backward(dout.T)
dγ = bn.dγ
dβ = bn.dβ


print('''Backprop correct:
 dx:
     [[-0.00310319  0.00305468 -0.00156246  0.17251307  0.01388029]
     [ 0.01147762 -0.10800884 -0.01112564 -0.02021632 -0.02098085]
     [-0.01682492 -0.01106847 -0.00384286  0.13581055 -0.04108612]
     [ 0.00845049  0.11602263  0.01653096 -0.2881073   0.04818669]]
dgamma:
     [2.29048278 1.39248907 2.93350569 0.98234546 2.08326113]
dbeta:
     [ 0.08461601  0.59073617  1.2668311  -1.75428014 -0.80317214]
''')

print('Backprop calculated')
print('dx:\n', dx.T)
print('dγ:\n', dγ.T)
print('dβ:\n', dβ.T)


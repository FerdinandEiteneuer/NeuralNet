import numpy as np

def generate_test_data(N=10000):
    #y=np.random.uniform(0,1,(1,N))
    y=np.random.random((1,N))
    x=0.5-y**2
    return x, y
def load_california_housing(path='/home/gefett/NeuralNet/testdata/california_housing'):
    x=np.load(path+'/cal_housing_x.npy')
    y=np.load(path+'/cal_housing_y.npy')
    y = 0.5*(2* (y - y.min()) / (y.max() - y.min()) - 1)
    for i in range(x.shape[0]):
        mean = x[i].mean()
        std = x[i].std()
        x[i] = (x[i] - mean) / std
    return x, y

def sub_man_div_std(data):
    pass

def split(x, y, split_portion=0.8):
    N = x.shape[1]
    cut = int(split_portion * N)
    xtrain = x[:,:cut]
    ytrain = y[:,:cut]
    xtest = x[:,cut:]
    ytest = y[:,cut:]
    return xtrain, ytrain, xtest, ytest

def minibatches(x, y, size=32):
    '''generates minibatches of given size
        does not work on the fly, generates all at once'''
    N = x.shape[1]
    assert(N == y.shape[1])

    N, size = int(N), int(size)
    N_batches = N / size

    if N % size != 0:
        N_batches += 1

    batches = []
    for i in range(N_batches):
        start = i * size
        end = (i+1) * size
        mini_x = x[:,start:end]
        mini_y = y[:,start:end]
        batches.append((mini_x, mini_y))

    return batches

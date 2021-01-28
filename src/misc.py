import numpy as np

def one_hot(ydata, classes):
    y = np.zeros((classes, ydata.shape[0]))
    for i in np.arange(ydata.shape[0]):
        y[ydata[i],i] = 1
    return y

def load_mnist(path = '/home/gefett/python/data/mnist/', load_test_val_data = (0,0)):
    load_test, load_val = load_test_val_data
    trainx = np.load(path + 'trainx.npy')
    trainy = np.load(path + 'trainy.npy')
    if load_test:
        testx = np.load(path + 'testx.npy')
        testy = np.load(path + 'testy.npy')
    else:
        return trainx, trainy
    if loadvaldata:
        valx = np.load(path + 'valx.npy')
        valy = np.load(path + 'valy.npy') 
        return trainx, trainy, testx, testy, valx, valy 
    else:
        return trainx, trainy, testx, testy

def generate_conv_test_data(N=20, dim=17, classes=3):
    shape=(dim,dim,1,N)
    trainx = np.random.randint(0,100,shape)
    #trainx = np.ones(shape)
    cls = np.random.randint(0, classes, N)
    trainy = one_hot(cls, classes)
    return trainx, trainy

def generate_test_data(N=10000):
    #y=np.random.uniform(0,1,(1,N))
    y=np.random.random((1,N))
    x=0.5-y**2
    return x, y

def load_california_housing(path='/home/gefett/python/data/california_housing'):
    x=np.load(path+'/cal_housing_x.npy')
    y=np.load(path+'/cal_housing_y.npy')
    #y = (y - y.min()) / (y.max() - y.min()) + 0.1 #0..1
    #y = 2*y - 1
    y = (y - y.mean()) / y.std()
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

    N = x.shape[-1]
    assert(N == y.shape[-1])
    
    #random_choice = np.random.permutation(np.arange(N))
    #x = x[:,random_choice]
    #y = y[:,random_choice]

    N, size = int(N), int(size)
    N_batches = N / size

    if N % size != 0:
        N_batches += 1

    batches = []
    for i in range(N_batches):
        start = i * size
        end = (i+1) * size
        mini_x = x[...,start:end]
        mini_y = y[:,start:end]
        batches.append((mini_x, mini_y))

    return batches

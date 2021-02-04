import numpy as np

def split(x, y, split_portion=0.8):
    N = x.shape[1]
    cut = int(split_portion * N)
    xtrain = x[:,:cut]
    ytrain = y[:,:cut]
    xtest = x[:,cut:]
    ytest = y[:,cut:]
    return xtrain, ytrain, xtest, ytest

def minibatches(x, y, batch_size=32):
    '''
    Generates minibatches of given size.
    '''

    assert x.shape[-1] == y.shape[-1], f'shapes dont match: {x.shape}, {y.shape}'

    nb_trainpoints = x.shape[-1]
    nb_batches = int(nb_trainpoints // batch_size)

    random_choice = np.random.permutation(np.arange(nb_trainpoints))

    for i in range(nb_batches):

        choice = random_choice[ i*batch_size : (i+1)*batch_size ]

        minix = x[...,choice]
        miniy = y[...,choice]

        minibatch = minix, miniy

        yield minibatch

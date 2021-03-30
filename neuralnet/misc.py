"""
Utility functions.
"""
import sys
import time

import numpy as np

def rel_error(x, y):
    """
    Returns relative error between x and y.
    """
    x = np.array(x)
    y = np.array(y)
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def split(x, y, split_portion=0.8):
    """
    Splits two datasets.
    """
    N = x.shape[1]
    cut = int(split_portion * N)
    xtrain = x[:,:cut]
    ytrain = y[:,:cut]
    xtest = x[:,cut:]
    ytest = y[:,cut:]
    return xtrain, ytrain, xtest, ytest

def minibatches(x, y, batch_size=32):
    """
    Generates minibatches of given size.
    """

    assert x.shape[-1] == y.shape[-1], f'shapes dont match: {x.shape}, {y.shape}'

    nb_trainpoints = x.shape[-1]
    nb_batches = max(1, nb_trainpoints // batch_size)

    random_choice = np.random.permutation(np.arange(nb_trainpoints))

    for i in range(nb_batches):

        choice = random_choice[ i*batch_size : (i+1)*batch_size ]

        minix = x[...,choice]
        miniy = y[...,choice]

        minibatch = minix, miniy

        yield minibatch

class ProgressBar():
    """
    [==========>...................]
    """

    def __init__(self):
        self.epochs = 0
        self.steps_per_epoch = 0

        self.epoch = 0
        self.step = 0
        self.loss = 0
        self.val_loss = None

        self.train_accuracy = 0
        self.val_accuracy = None

        self.gradient_check = None

        # config
        self.width_arrow = 20
        self.verbose=True

    def setup(self, epochs, steps_per_epoch, verbose=True):
        self.epochs += epochs
        self.steps_per_epoch = steps_per_epoch
        self.step = 0
        self.verbose=True

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, steps):
        self._step = steps

        if self._step > self.steps_per_epoch:
            self._step = self._step % self.steps_per_epoch


    def clear_line(self):
        print(200*'\b', end='', flush=True)


    def arrow(self):
        prog = self.step / self.steps_per_epoch
        x = int(prog * (self.width_arrow - 2))
        x = min(self.width_arrow - 3, x)
        y = self.width_arrow - x - 3
        out = '[' + x*"=" + ">" + y*"." + "]"
        return out


    def write(self):
        if self.verbose:
            self.clear_line()
            epoch = f'Epoch {self.epoch}/{self.epochs}'
            whitespace = " " * (len(str(self.steps_per_epoch)) - len(str(self.step)))
            batch = f'Batch {whitespace}{self.step}/{self.steps_per_epoch}'
            arr = self.arrow()

            out = f'{epoch} {batch} {arr} loss: {self.loss:.4f}, train_acc: {self.train_accuracy:.2%}'
            print(out, end='', flush=True)

    def append(self, val_loss, val_acc):
        if self.verbose:
            out = f', val_los: {val_loss:.4f}, val_acc: {val_acc:.2%}'
            if self.gradient_check:
                out += f', grad_check: {self.gradient_check:.3e}'

            print(out, end='\n', flush=True)





if __name__ == '__main__':
    p = ProgressBar()
    p.setup(epochs=10, steps_per_epoch=33, use_validation=False)

    for i in range(60):
        p.step += 1
        print(p.step, end=' ')
        p.arrow()




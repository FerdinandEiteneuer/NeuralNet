"""
Utility functions.
"""
import sys
import time
from collections import defaultdict

import numpy as np


def accuracy(pred, true):
    """
    Returns accuracy of predicted values.
    """
    batch_size = pred.shape[-1]
    pred_labels = np.argmax(pred, axis=0)

    if len(true.shape) == 2:
        true = np.argmax(true, axis=0)  # collapse one-hot representation to 1 dim
    elif len(true.shape) == 1:
        pass
    else:
        raise ValueError(f'Dont know how to handle true array of shape {true.shape}')
    correctly_labeled = np.sum(pred_labels == true)
    return correctly_labeled / batch_size

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

        # config
        self.width_arrow = 20
        self.verbose=True

    def setup(self, steps_per_epoch, history, epochs, verbose=True):
        self.steps_per_epoch = steps_per_epoch
        self.history = history
        self.verbose = verbose
        self.epochs = epochs

    def clear_line(self):
        print(200*'\b', end='', flush=True)

    @property
    def step(self):
        return self.history.step

    @property
    def epoch(self):
        return self.history.epoch

    def arrow(self):
        prog = self.step / self.steps_per_epoch
        x = int(prog * (self.width_arrow - 2))
        x = min(self.width_arrow - 3, x)
        y = self.width_arrow - x - 3
        out = '[' + x*"=" + ">" + y*"." + "]"
        return out

    def write(self):
        if self.verbose:

            loss = self.history.train_loss.value
            train_acc = self.history.train_acc.value

            self.clear_line()
            epoch = f'Epoch {self.epoch}/{self.epochs}'
            whitespace = " " * (len(str(self.steps_per_epoch)) - len(str(self.step)))
            batch = f'Batch {whitespace}{self.step}/{self.steps_per_epoch}'
            arr = self.arrow()

            out = f'{epoch} {batch} {arr} loss: {loss:.4f}, train_acc: {train_acc:.2%}'
            print(out, end='', flush=True)


    def write_end_of_episode(self):
        val_loss, val_acc = self.history.get_last_val_data()

        if self.verbose:
            out = f', val_los: {val_loss:.4f}, val_acc: {val_acc:.2%}'
            if goodness := self.history.get_last_gradient_check():
                out += f', grad_check: {goodness:.3e}'

            print(out, end='\n', flush=True)


class History:

    def __init__(self, exponential_moving_average_constant=0.8):

        self.β = exponential_moving_average_constant

        self.train_losses = defaultdict(list)
        self.train_accs = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.val_accs = defaultdict(list)
        self.gradient_checks = defaultdict(list)

        self.train_loss = MovingExponentialAverage(β=self.β)
        self.train_acc = MovingExponentialAverage(β=self.β)

        self.epoch = None

    def update_minibatch(self, train_loss, train_acc):

        self.train_loss.update(train_loss)
        self.train_acc.update(train_acc)

        self.train_losses[self.epoch].append(train_loss)
        self.train_accs[self.epoch].append(train_acc)

    def update_val_data(self, val_loss, val_acc):
        self.val_losses[self.epoch].append(val_loss)
        self.val_accs[self.epoch].append(val_acc)

    def update_gradients(self, goodness):
        self.gradient_checks[self.epoch].append(goodness)

    def get_last_val_data(self):
        loss = self.val_losses[self.epoch][-1]
        acc = self.val_accs[self.epoch][-1]
        return loss, acc

    def get_last_gradient_check(self):
        goodness = self.gradient_checks[self.epoch]
        if goodness == []:
            return None
        else:
            return goodness[-1]

    def update_current_state(self, epoch, minibatch):
        self.epoch = epoch
        self.step = minibatch


class MovingExponentialAverage:
    def __init__(self, β=0.9):
        self.t = 0
        self._value = 0
        self.β = β


    @property
    def value(self):
        return self._value / (1-self.β**self.t)


    def update(self, newval):
        self.t += 1
        self._value = self.β * self._value + (1 - self.β) * newval
        #return self.value / (1 - β**self.t)
        return self.value


    def __str__(self):
        return str(self.value)


if __name__ == '__main__':
    """
    p = ProgressBar()
    p.setup(epochs=10, steps_per_epoch=33)

    for i in range(60):
        p.step += 1
        print(p.step, end=' ')
        p.arrow()
    """

    loss = MovingExponentialAverage(β=0.5)

    loss.update(1)
    print(loss)
    loss.update(1)
    print(loss)


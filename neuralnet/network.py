'''
Implementation of the numpynets
'''

from functools import partial
import itertools
import warnings

import numpy as np

from .layer import Layer
from .activations import softmax
from . import misc

__all__ = ['Sequential']


class BaseNetwork():

    def __init__(self, verbose=True):
        self._layers = [Layer()]
        self.epoch = 0
        self.verbose=verbose

    def __call__(self, a):
        return self.forward_step(a)

    def __len__(self):
        return len(self._layers) - 1

    def __getitem__(self, layerid):
        return self._layers[layerid]

    def __iter__(self):
        for layer in self._layers[1:]:
            yield layer

    def add(self, layer):
        self._layers.append(layer)

    def predict(self, x):
        return self(x)


class Sequential(BaseNetwork):

    def __str__(self):

        s = 18*'*' + '\nmodel information:\n'

        number_params = 0
        params = ['w','b','beta','gamma']
        for layer in self:
            s += str(layer) + '\n'
            for param in params:
                if hasattr(layer,param):
                    N = np.multiply.accumulate(getattr(layer,param).shape)[-1]
                    number_params += N

        s += f'total number of parameters: {number_params}\n'
        if hasattr(self, 'optimizer'):
            s += str(self.optimizer) + '\n'

        return s

    def compile(self, loss, optimizer):

        self.loss_fct = loss().function
        self.derivative_loss_fct = loss().derivative

        optimizer.prepare(self)
        self.optimizer = optimizer


    def forward_step(self, a):
        self[0].a = a  # layer #0 is just there for saving the training data. it gets accessed in the last iteration in the loop in backpropagation during a_next = self[l - 1].a
        for layer in self:
            a = layer(a)
        return a


    def train_on_batch(self, x, y):
        self.forward_step(x)
        self.backpropagation(x, y)


    def backpropagation(self, x, y, verbose=True):

        error_prev = self._backprop_last_layer(x, y)

        for l in range(len(self)-1, 0, -1):

            a_next = self[l - 1].a
            w_prev = self[l + 1].w

            error_prev = self[l].backward_step(a_next, w_prev, error_prev)


    def _backprop_last_layer(self, x, y):
        '''
        calculates the error for the last layer.
        It is a little bit special as it involves
        the cost function, so do it in its own function.
        '''

        derivative_loss = self.derivative_loss_fct(
            ypred=self[-1].a,
            ytrue=y,
            average_examples=False
        )

        derivative_layer = self[-1].g(
            z=self[-1].z,
            derivative=True
        )

        if self[-1].g is softmax:
            deltaL = np.einsum('in,jin->jn', derivative_loss, derivative_layer)
        else:
            deltaL = derivative_layer * derivative_loss

        batch_size = x.shape[-1]
        self[-1].dw = 1 / batch_size * np.dot(deltaL, self[-2].a.T)
        self[-1].db = 1 / batch_size * np.sum(deltaL, axis=1, keepdims=True)
        return deltaL


    def get_loss(self, x, ytrue, average_examples=True, verbose=False):

        ypred = self(x)
        loss = self.loss_fct(ypred, ytrue, average_examples=average_examples)

        regularizer_loss = sum(layer.loss_from_regularizers() for layer in self)

        if verbose:
            print(f'{loss=:.4f}, {regularizer_loss=:.4f}')

        return loss + regularizer_loss


    def fit(self, x, y, epochs=1, batch_size=128, validation_data=None, gradients_to_check_each_epoch=None, verbose=True):

        ytrain_labels = np.argmax(y, axis=0)
        Ntrain = x.shape[-1]

        if validation_data:
            xtest, ytest = validation_data
            assert xtest.shape[-1] == ytest.shape[-1]
            ytest_labels = np.argmax(ytest, axis=0)
            Ntest = xtest.shape[-1]
        else:
            val_printout = ''

        if not gradients_to_check_each_epoch:
            grad_printout = ''

        for epoch in range(1, epochs + 1):

            losses = []
            self.optimizer.lr *= 0.993

            minibatches = misc.minibatches(x, y, batch_size=batch_size)
            for m, minibatch in enumerate(minibatches):

                self.train_on_batch(*minibatch)

                losses.append(self.get_loss(*minibatch))

                # important: do gradient checking before weights are changed!
                if gradients_to_check_each_epoch and m == 1:
                    goodness = self.gradient_checks(*minibatch, eps=10**(-6), checks=3)
                    if goodness:
                        grad_printout = f'gradcheck: {goodness:.3e}'
                    else:
                        grad_printout = 'gradcheck n/a, all grads are zero'

                self.optimizer.update_weights()


            a_train = self(x)
            ytrain_pred = np.argmax(a_train, axis=0)
            train_correct = np.sum(ytrain_pred == ytrain_labels)
            loss = np.mean(losses)

            if validation_data:
                a_test = self(xtest)
                ytest_pred = np.argmax(a_test, axis=0)
                test_correct = np.sum(ytest_pred == ytest_labels)
                val_loss = self.get_loss(xtest, ytest)

                val_printout = f'{val_loss=:.3f}, test: {test_correct}/{Ntest}'

            print(f'{epoch=}, {loss=:.3f}, train: {train_correct}/{Ntrain}, {val_printout}, {grad_printout}')

            self.epoch += 1


    def complete_gradient_check(self, x, y, eps=10**(-6)):
        self.grads_ = []
        for layer in self:

            gradient_manual = np.zeros(layer.w.shape)

            ranges = [range(dim) for dim in layer.w.shape]
            for idx in itertools.product(*ranges):

                gradient = self.gradient_check(
                    x=x,
                    ytrue=y,
                    eps=eps,
                    layer_id=layer.layer_id,
                    weight_idx=idx
                )

                gradient_manual[idx] = gradient

            numerator = np.linalg.norm(gradient_manual - layer.dw)
            denominator = np.linalg.norm(gradient_manual) +  np.linalg.norm(layer.dw)

            goodness = numerator / denominator
            self.n= numerator
            self.d=denominator
            self.grads_.append(gradient_manual)
            print(f'backprop err layer {layer.layer_id}: {goodness=}')


    def gradient_checks(self, x, ytrue, checks=15, eps=10**(-6)):
        '''
        Carries out several gradient checks in random places at once
        '''

        grads = np.zeros(checks)
        grads_backprop = np.zeros(checks)

        for check in range(checks):

            layer_id = np.random.randint(1, len(self))
            shape = self[layer_id].w.shape
            weight_idx = tuple(np.random.choice(dim) for dim in shape)
            gradient = self.gradient_check(
                x=x,
                ytrue=ytrue,
                eps=eps,
                layer_id=layer_id,
                weight_idx=weight_idx
            )

            grads[check] = gradient
            grads_backprop[check] = self[layer_id].dw[weight_idx]

        n = np.linalg.norm
        normed_sum = n(grads) + n(grads_backprop)
        if normed_sum == 0:
            goodness = None
        else:
            goodness = n(grads - grads_backprop) / normed_sum

        return goodness


    def gradient_check(self, x, ytrue, eps, layer_id, weight_idx):
        ''' to test the backprop algorithm, we also manually check the gradient
            for one randomly chosen weight/bias
            do this by using df(x)/dx = (f(x+eps) - f(x-eps)/2/eps'''

        cost = 0
        w_original = self[layer_id].w[weight_idx]

        for sign in [+1, -1]:

            self[layer_id].w[weight_idx] = w_original + sign * eps # change weight
            cost += sign * self.get_loss(x, ytrue, average_examples=True)

        self[layer_id].w[weight_idx] = w_original  # restore weight


        gradient_manual = cost / (2*eps)
        return gradient_manual

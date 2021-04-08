'''
Implementation of the numpynets
'''

from functools import partial
from collections import defaultdict
import itertools
import warnings

import numpy as np

from .layer import Layer
from .dense import Dense
from .dropout import Dropout
from .conv2d import Conv2D, Flatten, MaxPooling2D
from .activations import softmax
from .batchnorm import BatchNormalization
from . import misc

__all__ = ['Sequential']

_layer_types = [Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization]


class BaseNetwork():


    def __init__(self, *args, layers=None, **kwargs):

        self.layer_numbers = {ltype: 0 for ltype in _layer_types}

        self._layers = [Layer(layer_id=0)]
        self.w = None
        self.b = None

        if layers:
            for layer in layers:
                self.add(layer)

        self.epoch = 0

    def summary(self):
        return self.__str__()

    def __str__(self):

        width = 69
        s  = '\n' + width * '_' + '\n'
        s += 'Layer (type)                      Output Shape              Param #   \n'
        s += width * '=' + '\n'
        params = ['w','b','β','γ']

        for layer in self:
            s += str(layer) + '\n'

        trainable_params = sum(layer.n_parameters for layer in self)
        nontrainable_params = sum(layer.n_nontrainable_parameters for layer in self)
        total = trainable_params + nontrainable_params


        s += f'{width * "="}\n' \
            f'Total params: {total}\n' \
            f'Trainable params: {trainable_params}\n' \
            f'Non-trainable params: {nontrainable_params}\n' \
            f'{width * "_"}\n'


        if hasattr(self, 'optimizer'):
            s += '\n' + str(self.optimizer) + '\n'

        return s

    def __call__(self, a, mode='test'):
        return self.forward(a, mode=mode)

    def __len__(self):
        return len(self._layers) - 1

    def __getitem__(self, layerid):
        try:
            return self._layers[layerid]
        except TypeError:
            for layer in self._layers[1:]:
                if layer.name == layerid:
                    return layer
            else:
               raise KeyError(f'could not find layer {layerid} in model with names {[l.name for l in self]}')

    def __iter__(self):
        for layer in self._layers[1:]:
            yield layer

    def add(self, layer):

        self._layers.append(layer)

        layer.layer_id = len(self._layers) - 1  # access layer by using model[layer_id]
        layer.class_layer_id = self.layer_numbers[type(layer)]  # enumerates the number of layers of that type
        self.layer_numbers[type(layer)] += 1

        if len(self._layers) >= 3:

            prev_layer = self._layers[-2]

            # error handling
            if isinstance(layer, Dense) and isinstance(prev_layer, Conv2D):
                raise TypeError('need a \'Flatten\' Layer inbetween Conv2D layers and Dense Layers.')
            if isinstance(layer, Conv2D) and isinstance(prev_layer, Conv2D):
                layer.previous_filters = prev_layer.filters


    def trainable_layers(self):
        for layer in self:
            #if len(layer.trainable_parameters) > 0:
            #    yield layer
            if hasattr(layer, 'w'):
                yield layer

    def get_size(self, mode='G'):
        size = 0
        for layer in self:
            size += layer.get_size(mode)


    @property
    def dropout_used(self):
        d_used = False
        for layer in self:
            try:
                if layer.p_dropout < 1:
                    d_used = True
            except:
                pass
        return d_used

    def predict(self, x):
        return self(x, mode='test')


class Sequential(BaseNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progbar = misc.ProgressBar()
        self.epoch = 1


    def compile(self, loss, optimizer):
        self.loss_fct = loss().function
        self.derivative_loss_fct = loss().derivative

        first_layer = self[1]

        assert not isinstance(first_layer, Flatten), 'Flatten as first layer is not supported'

        next_input_shape = first_layer.prepare_params()

        if not next_input_shape:
            raise ValueError('The first layer must have an input_dim.'
                             'It can not be deduced by previous layers.')

        for layer in self[2:]:
            #print(f'preparing {layer.name} with shape {next_input_shape}')
            next_input_shape = layer.prepare_params(next_input_shape)

        self.optimizer = optimizer.prepare_params(self)


    def forward(self, a, mode='test'):
        for layer in self:
            a = layer(a, mode)
        return a


    def train_on_batch(self, x, y):
        self.forward(x, mode='train')
        self.backward(x, y)


    def backward(self, x, y, verbose=0):

        final_layer = self[-1]

        dout = self.derivative_loss_fct(
            ypred=final_layer.a,
            ytrue=y,
            average_examples=False
        )

        batch_size = y.shape[-1]
        dout /= batch_size


        if verbose: print('err prev from last layer backprop', dout.shape)

        for l in range(len(self), 0, -1):

            dout = self[l].backward(dout)
            if verbose: print(f'got error_prev for {self[l]}, {dout.shape=}')


    def loss(self, x, y, mode='test', verbose=False):
        ypred = self(x, mode=mode)
        return self.evaluate_costfunction(ypred, y, verbose=verbose)


    def evaluate_costfunction(self, ypred, ytrue, verbose=False, fast=False):

        loss = self.loss_fct(ypred, ytrue, average_examples=True)
        regularizer_loss = sum(layer.loss_from_regularizers() for layer in self)
        total_loss = loss + regularizer_loss

        if verbose:
            print(f'total loss: {total_loss:.4e}, ({loss=:.4e}, {regularizer_loss=:.4e})\n')

        return total_loss


    def fix_dropout_seed(self):
        seed = np.random.randint(2**32 - 1)
        for layer in self:
            layer.dropout_seed = seed


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

        history = misc.History(
            current_epoch=self.epoch,
            end_epoch=self.epoch + epochs - 1,
            steps_per_epoch=max(1, Ntrain // batch_size),
            exponential_moving_average_constant=0.9
        )

        self.progbar.setup(
            history=history,
            verbose=verbose,
        )

        for epoch in range(self.epoch, self.epoch + epochs):

            self.optimizer.lr *= 1

            minibatches = misc.minibatches(x, y, batch_size=batch_size)

            for m, minibatch in enumerate(minibatches):

                self.train_on_batch(*minibatch)

                train_loss = self.evaluate_costfunction(
                    ypred=self[-1].a,
                    ytrue=minibatch[1]
                )

                train_acc = misc.accuracy(self[-1].a, minibatch[1])

                history.update_minibatch(train_loss, train_acc)

                # important: do gradient checking before weights are changed!
                if gradients_to_check_each_epoch and m == 0:
                    if gradients_to_check_each_epoch == 'all':
                        self.complete_gradient_check(*minibatch)
                    else:
                        goodness = self.gradient_checks(*minibatch, eps=10**(-6), checks=gradients_to_check_each_epoch, verbose=0)
                        history.update_gradients(goodness)
                #self.complete_gradient_check(*minibatch)


                self.optimizer.update_weights()
                self.progbar.write()


            if validation_data:
                ytest_pred = self(xtest)

                val_loss = self.evaluate_costfunction(ytest_pred, ytest)
                val_acc = misc.accuracy(ytest_pred, ytest_labels)

                history.update_val_data(val_loss, val_acc)

                self.progbar.write_end_of_episode()

        self.epoch += epochs
        return history


    def gradient_check(self, x, ytrue, eps, layer_id, weight_idx):
        ''' To test the backprop algorithm, we manually check the gradient
            for one chosen weight/bias. do this by using

               df(x)/dx = (f(x+eps) - f(x-eps)/2/eps
        '''

        gradient_manual = 0
        w_original = self[layer_id].w[weight_idx]

        for sign in [+1, -1]:

            self[layer_id].w[weight_idx] = w_original + sign * eps # change weight

            ypred = self.forward(x, mode='gradient')
            gradient_manual += sign * self.evaluate_costfunction(ypred, ytrue)

        self[layer_id].w[weight_idx] = w_original  # restore weight

        gradient_manual /= (2*eps)
        return gradient_manual


    def gradient_checks(self, x, ytrue, checks=15, eps=10**(-6), verbose=False):
        '''
        Carries out several gradient checks in random places.
        '''

        self._grads = defaultdict(list)
        self._grads_backprop = defaultdict(list)

        check = 0
        while check < checks:

            layer_id = np.random.randint(1, len(self) + 1)
            if self[layer_id] not in self.trainable_layers():
                continue

            shape = self[layer_id].w.shape
            weight_idx = tuple(np.random.choice(dim) for dim in shape)
            gradient = self.gradient_check(
                x=x,
                ytrue=ytrue,
                eps=eps,
                layer_id=layer_id,
                weight_idx=weight_idx
            )

            self._grads[layer_id].append(gradient)
            self._grads_backprop[layer_id].append(self[layer_id].dw[weight_idx])

            check += 1

        goodness = []

        if verbose: print('')
        for l in self._grads.keys():
            goodness_layer = misc.rel_error(self._grads[l], self._grads_backprop[l])
            goodness.append(goodness_layer)

            if verbose:
                print(f'goodness {self[l].name}: {goodness_layer:.3e}')
                if len(self._grads[l]) < 8 and goodness_layer > 1e-4:
                    print(f'grads manual {self[l].name} {self._grads[l]}')
                    print(f'grads backprp {self[l].name} {self._grads_backprop[l]}')
        if verbose: print('')


        return max(goodness)


    def complete_gradient_check(self, x, y, eps=10**(-6)):
        """ Checks the gradient for every weight in the network.
            For computational reasons, can only be used in very small networks."""
        self._gradchecks = []
        worst_check = -np.inf

        for layer in self.trainable_layers():

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

            #print(gradient_manual)
            print(f'grad backprop:\n{layer.dw}\ngrad manual\n{gradient_manual}')

            self._gradchecks.append(gradient_manual)
            goodness = misc.rel_error(gradient_manual, layer.dw)
            print(f'backprop err layer {layer.name}: {goodness=}')

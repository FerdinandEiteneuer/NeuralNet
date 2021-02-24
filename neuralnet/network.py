'''
Implementation of the numpynets
'''

from functools import partial
import itertools
import warnings

import numpy as np

from .layer import Layer
from .dense import Dense
from .conv2d import Conv2D, Flatten
from .activations import softmax
from . import misc

__all__ = ['Sequential']


class BaseNetwork():

    def __init__(self, layers=None):

        self._dense_layers = 0
        self._conv_layers = 0
        self._flatten_layers = 0

        self._layers = [Layer(layer_id=0)]

        if layers:
            for layer in layers:
                self.add(layer)

        self.epoch = 0

    def summary(self):
        return self.__str__()

    def __str__(self):

        width = 65
        s  = '\n' + width * '_' + '\n'
        #s += 'Layer (type)                 Output Shape              Param #\n'
        s += 'Layer (type)                 Output Shape              Param #   \n'
        s += width * '=' + '\n'
        number_params = 0
        params = ['w','b','beta','gamma']
        for layer in self:
            s += str(layer) + '\n'
            for param in params:
                if hasattr(layer, param):
                    par = getattr(layer, param)
                    N = np.product(par.shape)
                    number_params += N

        s += f'{width * "="}\n' \
            f'Total params: {number_params}\n' \
            f'Trainable params: {number_params}\n' \
            f'Non-trainable params: 0\n' \
            f'{width * "_"}\n'


        if hasattr(self, 'optimizer'):
            s += '\n' + str(self.optimizer) + '\n'

        return s

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

        # setting layer id, which is the index of the layer in model, e.g model[2]
        # returns the layer with layer_id 2
        layer.layer_id = len(self._layers) - 1

        # setting class layer ids (used ONLY in __str__)
        if isinstance(layer, Dense):
            layer.class_layer_id = self._dense_layers
            self._dense_layers += 1

        elif isinstance(layer, Conv2D):
            layer.class_layer_id = self._conv_layers
            self._conv_layers += 1

        elif isintance(layer, Flatten):
            layer._class_layer_id = self._flatten_layers
            self._flatten_layers += 1

        else:
            raise TypeError(f'Do not know how to handle {layer=}')

        if len(self._layers) >= 3:

            prev_layer = self._layers[-2]

            # error handling
            if isinstance(layer, Dense) and isinstance(prev_layer, Conv2D):
                raise TypeError('need a \'Flatten\' Layer inbetween Conv2D layers and Dense Layers.')
            if isinstance(layer, Conv2D) and isinstance(prev_layer, Conv2D):
                layer.previous_filters = prev_layer.filters

    def predict(self, x):
        return self(x)


class Sequential(BaseNetwork):

    def compile(self, loss, optimizer):

        self.loss_fct = loss().function
        self.derivative_loss_fct = loss().derivative

        first_layer = self[1]

        assert isinstance(first_layer) != Flatten, 'Flatten as first layer is not supported'

        next_input_dim = first_layer.prepare_params()

        if not next_input_dim:
            raise ValueError('The first layer must have an input_dim.'
                             'It can not be deduced by previous layers.')

        for layer in self[2:]:
            next_input_dim = layer.prepare_params(next_input_dim)

        self.optimizer = optimizer.prepare_params(self)


    def forward_step(self, a):
        self[0].a = a  # layer #0 is just there for saving the training data. it gets accessed in the last iteration in the loop in backpropagation during a_next = self[l - 1].a
        for layer in self:
            a = layer(a)
        return a


    def train_on_batch(self, x, y):
        self.forward_step(x)
        self.backpropagation(x, y)


    def backpropagation(self, x, y, verbose=True):

        error_prev = self.backprop_last_layer(x, y)

        for l in range(len(self)-1, 0, -1):

            a_next = self[l - 1].a
            w_prev = self[l + 1].w

            error_prev = self[l].backward_step(a_next, w_prev, error_prev)


    def backprop_last_layer(self, x, y):
        '''
        calculates the error for the last layer.
        It is a little bit special as it involves
        the cost function, so do it in its own function.
        '''

        layer = self[-1]

        derivative_loss = self.derivative_loss_fct(
            ypred=layer.a,
            ytrue=y,
            average_examples=False
        )

        derivative_layer = layer.g(
            z=layer.z,
            derivative=True
        )

        if layer.g is softmax:
            deltaL = np.einsum('in,jin->jn', derivative_loss, derivative_layer)
        else:
            deltaL = derivative_layer * derivative_loss

        batch_size = x.shape[-1]

        layer.dw = 1 / batch_size * np.dot(deltaL, self[-2].a.T)
        layer.db = 1 / batch_size * np.sum(deltaL, axis=1, keepdims=True)

        if layer.kernel_regularizer:
            layer.dw += 1 / batch_size * self.kernel_regularizer.derivative()
        if layer.bias_regularizer:
            layer.db += 1 / batch_size * self.bias_regularizer.derivative()

        return deltaL


    def get_loss(self, x, ytrue, average_examples=True, verbose=False):

        ypred = self(x)
        loss = self.loss_fct(ypred, ytrue, average_examples=average_examples)

        batch_size = ytrue.shape[-1]
        regularizer_loss = sum(layer.loss_from_regularizers(batch_size) for layer in self)

        if verbose:
            print(f'{loss=:.4e}, {regularizer_loss=:.4e}')

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
        return losses


    def gradient_check(self, x, ytrue, eps, layer_id, weight_idx):
        ''' To test the backprop algorithm, we manually check the gradient
            for one chosen weight/bias. do this by using

               df(x)/dx = (f(x+eps) - f(x-eps)/2/eps
        '''

        cost = 0
        w_original = self[layer_id].w[weight_idx]

        for sign in [+1, -1]:

            self[layer_id].w[weight_idx] = w_original + sign * eps # change weight
            cost += sign * self.get_loss(x, ytrue, average_examples=True)

        self[layer_id].w[weight_idx] = w_original  # restore weight


        gradient_manual = cost / (2*eps)
        return gradient_manual


    def gradient_checks(self, x, ytrue, checks=15, eps=10**(-6)):
        '''
        Carries out several gradient checks in random places.
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


    def complete_gradient_check(self, x, y, eps=10**(-6)):
        ''' Checks the gradient for every weight in the network.'''
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




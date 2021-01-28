import layer
from flatten import Flatten
import numpy as np

class Network(list):

    def __init__(self, verbose=True):

        self.layers = {}
        self.epoch = 0
        self.verbose=verbose

        self.append(layer.Layer())  # starting layer, for holding input values to the network

    def __str__(self):
        number_params = 0
        params = ['w','b','beta','gamma']
        for layer in self:
            for param in params:
                if hasattr(layer,param):
                    N = np.multiply.accumulate(getattr(layer,param).shape)[-1]
                    number_params += N

        s = 'model information:\n'
        s += '  layers: %i\n' % len(self)
        s += '  number of parameters: %i' % number_params
        print(s)

    def __call__(self, a):
        return self.forward_step(a)

    def add(self, layer):
        self.append(layer)

    def compile(self, loss, lr):
        self.batch_norm = False
        for l in self.layers:
            try:
                if l.info == 'batch':
                    self.batch_norm = True
            except AttributeError:
                pass

        self.lr = lr
        self.lossfunction = loss['function']
        self.derivative_lossfunction = loss['derivative']

    def forward_step(self, a):

        self[0].a = a
        if self.verbose: print('START FORWARD STEP')
        for layer in self:
            a = layer(a)
            if self.verbose: print(f'{a.shape=}')
            if self.verbose: print(layer)
        return a

    def train_on_batch(self, x, y):
        self.forward_step(x)
        self.backpropagation(x, y)

    def backpropagation(self, x, y):
        if self.verbose: print('START BACKPROP')

        #first do last layer, then the rest
        y_pred = self[-1].a
        back_err = self.derivative_lossfunction(y_pred, y)

        # the rest of the network
        for l in range(len(self)-1, 0, -1):

            if self.verbose: print('\n' + 30*'*')
            if self.verbose: print(f'in backprop layer {l=}, using {back_err.shape=}')

            prev_layer = self[l-1]
            layer = self[l]

            prev_a = prev_layer.a
            a = layer.a

            if self.verbose: print('\nget_error')
            layer.get_error(back_err)

            if self.verbose: print('\nget_grads')
            layer.get_grads(prev_a)

            if self.verbose: print('\nupdate_weights')
            layer.update_weights(self.lr)

            back_err = layer.backward()
            #print '\terror shape', layer.error.shape

        #self.optimizer.update()

    def get_loss(self, x, ytrue):
        ypred = self.predict(x)
        loss = self.lossfunction(ypred, ytrue)
        return loss

    def predict(self, x):
        return self.forward_step(x)

    def gradient_check(self, x, ytrue, check_layer, grad_check_info, eps=10**(-7), random_weight = True):
        ''' to test the backprop algorithm, we also manually check the gradient
            for one randomly chosen weight/bias
            do this by using df(x)/dx = (f(x+eps) - f(x-eps)/2/eps'''
        take_weights = True
        grad_manual = 0
        check_k, check_j, check_c_prev, check_c = grad_check_info
        #a = layer.forward(a)
        for multiplier in [+1, -1]:
            tinychange = multiplier * eps
            a = x
            for l, layer in list(self.layers.items())[1:]:
                if l == check_layer:
                    a = layer.forward(a, gradient_check=True, grad_check_info = grad_check_info + (tinychange,))
                else:
                    a = layer.forward(a)

            cost = multiplier * self.get_loss(x, ytrue)
            grad_manual += cost

        grad_manual /= (2*eps)

        if take_weights:
            grad_backprop = self.layers[check_layer].dw[check_k, check_j, check_c_prev, check_c]

        #if self.verbose: print('in gradcheck:', grad_manual, grad_backprop)
        if grad_manual == 0:
            if grad_backprop == 0:
                ratio = 1
            else:
                ratio = np.inf
        else:
            ratio =  grad_backprop/grad_manual
            if abs(ratio-1) > 0.001:
                if take_weights:
                    parameter = 'w[%i][%i,%i,%i,%i]' % (check_layer, check_k, check_j, check_c_prev, check_c)
                if self.verbose: print('ratio backprop/manual=%.5f. cause: %s' % (ratio, parameter))
        return ratio, grad_manual, grad_backprop


    def save_model(self, name):
        #h5 file
        pass

from flatten import Flatten
import numpy as np

class Network:
    
    def __init__(self, verbose=True):
        self.layers = {}
        self.epoch = 0
        self.verbose=verbose

    def info(self):
        '''print model information'''
        number_params = 0
        params = ['w','b','beta','gamma']
        for l, layer in self.layers.items():
            for param in params:
                if hasattr(layer,param):
                    N = np.multiply.accumulate(getattr(layer,param).shape)[-1]
                    number_params += N
                
        s = 'model information:\n'
        s += '  layers: %i\n' % self.L - 1 
        s += '  number of parameters: %i' % number_params
        print s

    def add(self, layer):
        l = len(self.layers) + 1
        layer.l = l
        self.layers[l] = layer
        #self.set_attributes(layer.params, l) 
        if isinstance(layer, Flatten):
           layer.w = self.layers[l-1].w
  
    def compile(self, loss, lr):
        self.batch_norm = False
        for l in self.layers:
            try:
                if l.info == 'batch':
                    self.batch_norm = True
            except:
                pass
        self.L = len(self.layers)
        class placeholder: pass
        self.layers[0] = placeholder()
        self.lr = lr
        self.lossfunction = loss['function']
        self.derivative_lossfunction = loss['derivative']
 
    def set_attributes(self, params, l):
        for parameter_name, parameter in params.items():
            if not hasattr(self, parameter_name):
                setattr(self, parameter_name, {})
            
            #add for example self.w[l] = weight_matrix_layer_l
            getattr(self, parameter_name)[l] = parameter 

    def forward_step(self, a):
        for l, layer in self.layers.items()[1:]:
            a = layer.forward(a)
            if self.verbose: print 'layer %s' % l, 'activation shape', a.shape
        return a
    
    def train_step(self, train_minibatch):
        #do forward step
        x, y = train_minibatch
        self.layers[0].a = x

        N = y.shape[1]
        self.epoch += 1
        a = self.forward_step(x)

        #backprop. first do last layer, then the rest
        back_err = self.derivative_lossfunction(a, y)
        print 'loss_derivative, starting of backprop', back_err
        if self.verbose: print 'in backprop layer %i\n' % self.L, '\tback_err shape', back_err.shape

        for l in reversed(range(1, self.L+1)):
            if self.verbose: print '\nin backprop layer %i, using back_err_%i with shape %s\n' % (l, l+1, back_err.shape)
            layer = self.layers[l]
            a_prev = self.layers[l-1].a

            layer.get_error(back_err)
            layer.grads(a_prev, N)
            layer.update(self.lr) 
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
            for l, layer in self.layers.items()[1:]:
                if l == check_layer:
                    a = layer.forward(a, gradient_check=True, grad_check_info = grad_check_info + (tinychange,))
                else:
                    a = layer.forward(a)
                     
            cost = multiplier * self.get_loss(x, ytrue)
            grad_manual += cost

        grad_manual /= (2*eps)

        if take_weights:
            grad_backprop = self.layers[check_layer].dw[check_k, check_j, check_c_prev, check_c]
        
        #print('in gradcheck:', grad_manual, grad_backprop)
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
                print 'ratio backprop/manual=%.5f. cause: %s' % (ratio, parameter)
        return ratio, grad_manual, grad_backprop
        

    def save_model(self, name):
        #h5 file
        pass

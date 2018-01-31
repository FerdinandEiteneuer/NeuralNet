class Network:
    
    def __init__(self, loss):
        self.layers = {}
        self.lossfunction = loss['function']
        self.derivative_lossfunction = loss['derivative']
        self.epoch = 0

    def __string__(self):
        pass

    def add(self, layer):
        l = len(self.layers) + 1
        self.layers[l] = layer
        #self.set_attributes(layer.params, l)
 
    def compile(self, lr=0.001):
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

    def set_attributes(self, params, l):
        for parameter_name, parameter in params.items():
            if not hasattr(self, parameter_name):
                setattr(self, parameter_name, {})
            
            #add for example self.w[l] = weight_matrix_layer_l
            getattr(self, parameter_name)[l] = parameter 

    def forward_step(self, a):
        for l, layer in self.layers.items():
            if l == 0: continue
            a = layer.forward(a)
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
        #last_act_derivative = self.layers[-1].g(a, derivative=True)
        #error = loss_err * last_act_derivative

        for l in reversed(range(1, self.L+1)):
            layer = self.layers[l]
            a_prev = self.layers[l-1].a

            error = layer.get_error(back_err)
            #same as 
            #error= back_err * layer.g(layer.z, derivative=True)
            #N = a_prev.shape[-1] ?????
            layer.grads(a_prev, N)
            layer.update(self.lr) 

            back_err = layer.backward()
    
        #self.optimizer.update()

    def get_loss(self, x, ytrue):
        ypred = self.predict(x)
        loss = self.lossfunction(ypred, ytrue)
        return loss

    def predict(self, x):
        return self.forward_step(x)

    def save_model(self, name):
        #h5 file
        pass

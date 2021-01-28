import numpy as np
class SGD:
    def __init__(self, learning_rate=0.01, beta_1=None, model=None):
        self.lr = learning_rate
        self.classical_momentum = False
        #classical momentum
        if beta_1 != None:
            self.classical_momentum = True
            self.model = model
            self.initialize_momentum(beta_1)
    
    
    def update(self, fc, epoch):
        #update for each layer
        if self.classical_momentum:
            for l in fc.layers:
                #for each w and b, first update momentum vector and then w, b
                self.mom_w[l] = self.beta_1 * self.mom_w[l] + (1 - self.beta_1) * fc.dw[l]
                #self.mom_w[l] /= (1 - self.beta_1 ** (epoch+1)) #bias correction
                fc.w[l] -= self.lr * self.mom_w[l]

                self.mom_b[l] = self.beta_1 * self.mom_b[l] + (1 - self.beta_1) * fc.db[l]
                #self.mom_b[l] /= (1 - self.beta_1 ** (epoch+1)) #bias correction
                fc.b[l] -= self.lr * self.mom_b[l]
        else:        
            for l in fc.layers:
                fc.w[l] -= self.lr * fc.dw[l]
                fc.b[l] -= self.lr * fc.db[l]
        
    def initialize_momentum(self, beta_1):
        self.beta_1 = beta_1
        self.decay = 0.04
        #momenta for w and bias
        self.mom_w = {}
        self.mom_b = {}
        #they get initialized with zeros
        for l in self.model.layers:
            self.mom_w[l] = np.zeros(self.model.w[l].shape)
            self.mom_b[l] = np.zeros(self.model.b[l].shape)


class Nadam:
    '''Nadam Optimizer. Combines nesterov, momentum, RMS prop step'''
    def __init__(self, learning_rate=0.01, beta_1=0.9, beta_2=0.999, eps=10**(-8), model=None, bias_correction=True):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.bias_correction = bias_correction

        #momenta for w and bias
        self.mom_w = {}
        self.mom_b = {}
        
        #rms prop momentum
        self.rms_w = {}
        self.rms_b = {}
        #they get initialized with zeros
        for l in model.layers:
            self.mom_w[l] = np.zeros(model.w[l].shape)
            self.mom_b[l] = np.zeros(model.b[l].shape)
            
            self.rms_w[l] = np.zeros(model.w[l].shape)
            self.rms_b[l] = np.zeros(model.b[l].shape)
    
    def update(self, fc, epoch):
        for l in fc.layers:
            #momentum
            self.mom_w[l] = self.beta_1 * self.mom_w[l] + (1 - self.beta_1) * fc.dw[l]
            self.mom_b[l] = self.beta_1 * self.mom_b[l] + (1 - self.beta_1) * fc.db[l]

            #RMS prop.
            self.rms_w[l] = self.beta_2 * self.rms_w[l] + (1 - self.beta_2) * fc.dw[l]**2
            self.rms_b[l] = self.beta_2 * self.rms_b[l] + (1 - self.beta_2) * fc.db[l]**2

            #bias correction
            if self.bias_correction:
                epoch += 1
                self.rms_w[l] /= (1 - self.beta_2 ** epoch)
                self.rms_b[l] /= (1 - self.beta_2 ** epoch)

            #Nesterov + bias correcton, both for momentum
            
            if self.bias_correction:        
                m_hat_w = self.beta_1 * self.mom_w[l] / (1 - self.beta_1**(epoch+1)) + (1 - self.beta_1) * fc.dw[l] / (1-self.beta_1 ** epoch)
                m_hat_b = self.beta_1 * self.mom_b[l] / (1 - self.beta_1**(epoch+1)) + (1 - self.beta_1) * fc.db[l] / (1-self.beta_1 ** epoch)
            else:
                 m_hat_w = self.beta_1 * self.mom_w[l] + (1 - self.beta_1) * fc.dw[l]
                 m_hat_b = self.beta_1 * self.mom_b[l] + (1 - self.beta_1) * fc.db[l]

            #update parameters
            fc.w[l] -= self.lr * m_hat_w/(np.sqrt(self.rms_w[l]) + self.eps)
            fc.b[l] -= self.lr * m_hat_b/(np.sqrt(self.rms_b[l]) + self.eps)


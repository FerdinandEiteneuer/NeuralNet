from kernel_initializer import kernels
import numpy as np

class Conv2D:
    def __init__(self, filtersize, stride, padding, activation, kernel_initializer):
        self.f = filtersize[0]
        self.s = stride
        self.p = padding

        #pad only height and width (and not channels, trainexamples)
        self.pads = ((self.p, self.p),(self.p, self.p),(0,0), (0,0))

        self.c_prev = filtersize[2] #input channels
        self.c = filtersize[3] #amount of new channels
        self.initialize_params(kernel_initializer)

    def initialize_params(self, kernel_initializer)
        shape = (self.f, self.f, self.c_prev, self.c)
        self.w = kernels[kernel_initializer](shape)
        self.b = np.zeros(shape)

    def forward(self, a):
        #zero padding
        a = np.pad(a, pad_width=self.pads, mode='constant', constant_values=0)
        height_prev, width_prev, channels_prev, m = a.shape
        assert(channels_prev = self.c_prev)

        #create output volume
        height = (height_prev + 2*self.p - self.f)/self.s + 1
        width = (weight_prev + 2*self.p - self.f)/self.s + 1
        new = np.zeros((height, width, self.c, m))

        #prepare for convolution (corrext axis so that shapes fit for * operation)
        a = a[:,:,:,np.newaxis,:] #(height, width, c_prev, m) -> (height, width, c_prev, AXIS, m)
        W = self.w[...,np.newaxis] #(f, f, c_prev, c) -> (f, f, c_prev, c, AXIS)
        #this way, when summing over the first three axis( height/f, width/f, c_prev) a tensor with indices c,m (in this order) is created
        #compare to the shape of the tensor 'new'

        #convolution
        for h in height:
            for w in width:
                start = h*self.s
                end = h*self.s + f
                image_patch = a[start:end, start:end,...]
                new[h,w,:,:] = np.sum(image_patch * W, axis = (0,1,2)) + float(self.b)
        return new


    def grads(self, a_prev, N):
        pass

    def get_error(self, back_err):
        pass

    def backward(self):
        pass

    def update(self, lr)
        self.w -= lr * self.dw
        self.b -= lr * self.db  

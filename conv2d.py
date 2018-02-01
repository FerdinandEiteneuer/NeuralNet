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

    def initialize_params(self, kernel_initializer):
        shape = (self.f, self.f, self.c_prev, self.c)
        self.w = kernels[kernel_initializer](shape)
        self.b = np.zeros((1,1,self.c,1))

    def forward(self, a):
        #zero padding
        a = np.pad(a, pad_width=self.pads, mode='constant', constant_values=0)
        height_prev, width_prev, channels_prev, m = a.shape
        assert(channels_prev == self.c_prev)

        #create output volume
        height = (height_prev + 2*self.p - self.f)/self.s + 1
        width = (width_prev + 2*self.p - self.f)/self.s + 1
        new = np.zeros((height, width, self.c, m))

        #prepare for convolution (corrext axis so that shapes fit for * operation)
        a = a[:,:,:,np.newaxis,:] #(height, width, c_prev, m) -> (height, width, c_prev, AXIS, m)
        W = self.w[...,np.newaxis] #(f, f, c_prev, c) -> (f, f, c_prev, c, AXIS)
        #this way, when summing over the first three axis( height/f, width/f, c_prev) a tensor with indices c,m (in this order) is created
        #compare to the shape of the tensor 'new'

        #convolution
        for h in range(height):
            for w in range(width):
                start_h = h*self.s
                end_h = h*self.s + self.f

                start_w = w*self.s
                end_w = w*self.s + self.f

                image_patch = a[start_h:end_h, start_w:end_w,:,:]
                #print 'PIXEL h:%i, w:%i' % (h, w)
                new[h,w,:,:] = np.sum(image_patch * W, axis = (0,1,2)) + self.b
                print 'shape',new[h,w,:,:].shape, self.b.shape
        return new


    def grads(self, a_prev, N):
        pass

    def get_error(self, back_err):
        pass

    def backward(self):
        pass

    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


if __name__ == '__main__':
    from kernel_initializer import integers
    from activations import relu

    x = np.random.randint(1,5,(3,3,1,1))

    conv = Conv2D((2,2,1,2), 1, 0, relu, 'integers')
    a = conv.forward(x)

    w = conv.w
    f1 = w[:,:,0,0]
    f2 = w[:,:,0,1]
    img = x[:,:,0,0]
    a1 = a[:,:,0,0]
    a2 = a[:,:,1,0]

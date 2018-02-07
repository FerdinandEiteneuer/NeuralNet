from kernel_initializer import kernels
import numpy as np

class Conv2D:
    def __init__(self, filtersize, stride, padding, activation, kernel_initializer):
        self.g = activation

        self.f = filtersize[0]
        self.s = stride

        self.get_padding(padding) #gets self.p / self.pads

        self.c_prev = filtersize[2] #input channels
        self.c = filtersize[3] #amount of new channels
        self.initialize_params(kernel_initializer)

    def get_padding(self, padding):
        self.pad_anything = True
        if type(padding) == int:
            self.p = padding
            if self.p == 0:
                self.pad_anything = False
        elif padding == 'valid':
            self.p = None #what about self.pad_anything? here and same
        elif padding == 'same':
            self.p = None
        #pad only height and width (and not channels, trainexamples)
        self.pads = ((self.p, self.p),(self.p, self.p),(0,0), (0,0))

    def initialize_params(self, kernel_initializer):
        shape = (self.f, self.f, self.c_prev, self.c)
        self.w = kernels[kernel_initializer](shape)
        self.b = np.zeros((1,1,self.c,1))

    def forward(self, a):
        #zero padding
        if self.pad_anything:
            a = np.pad(a, pad_width=self.pads, mode='constant', constant_values=0)
        height_prev, width_prev, channels_prev, m = a.shape
        assert(channels_prev == self.c_prev)

        #create output volume
        height = (height_prev + 2*self.p - self.f)/self.s + 1
        width = (width_prev + 2*self.p - self.f)/self.s + 1
        self.z = np.zeros((height, width, self.c, m))

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
                self.z[h,w,:,:] = np.sum(image_patch * W, axis = (0,1,2)) + self.b
                #print 'shape',new[h,w,:,:].shape, self.b.shape
   
        self.a = self.g(self.z) 
        return self.z, self.a


    def grads(self, a_prev, N):
        I, J, C, N = self.error.shape
        
        self.dw =  None
        self.db = None

    def get_error(self, back_err):
        self.error = back_err * self.g(self.z, derivative=True)
        return self.error

    def backward(self):
        height, width, channels, m = self.z.shape
        self.backerr = np.zeros(self.z.shape)
    
        indices = lower_upper_summation_indices(height, self.f, self.s)
    
        for h in range(height):
            for w in range(width):
                low, up = indices[h], indices[w]
                error_part = self.error[low:up:,:,:]
                error_part = error_part[:,:,:,np.newaxis,:]

                axis1 = h - self.s * low
                axis2 = w - self.s * up
                W_part = self.w[np.ix_(axis1, axis2)]
                W_part = np.transpose(W_part, (0,1,3,2))
                W_part = W_part[...,np.newaxis]

                back_part = np.sum(error_part, W_part, axis=(0,1,2))
                self.backerr[h,w,:,:] = back_part
        return self.backerr 
        

    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def lower_upper_summation_indices(self, h, f, s):
        #lower
        mins = []
        for i in range(0, h):
           mins.append(int(max(0, round((i-f+1)/float(s)))))

        #upper is just shifted
        delta = abs(f-s)
        h_next = int((h - f)/float(s)) + 1
        stop_at = h_next - 1
        maxs = mins[delta:] + delta*[stop_at]

        mins = np.array(mins)
        maxs = np.array(maxs) + 1 #+1 because range(a,b) not inclusive for upper index
        return mins, maxs


if __name__ == '__main__':
    from kernel_initializer import integers
    from activations import relu

    x = np.random.randint(1,5,(3,3,1,1))

    conv = Conv2D((2,2,1,2), 1, 0, relu, 'integers')
    z, a = conv.forward(x)

    w = conv.w
    f1 = w[:,:,0,0]
    f2 = w[:,:,0,1]
    img = x[:,:,0,0]
    a1 = a[:,:,0,0]
    a2 = a[:,:,1,0]

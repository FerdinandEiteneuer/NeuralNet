from kernel_initializer import kernels
import numpy as np, sys

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

    def forward(self, a, gradient_check = False, grad_check_info=0):
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
        #compare to the shape of the tensor 'self.z'
        
        #gradient checking
        if gradient_check:
            W_temp = np.copy(W)
            j_change, k_change, c_prev_change, c_change, tinychange = grad_check_info
            W[j_change, k_change, c_prev_change, c_change] += tinychange

        #convolution
        for h in range(height):
            for w in range(width):
                start_h = h*self.s
                end_h = h*self.s + self.f

                start_w = w*self.s
                end_w = w*self.s + self.f

                image_patch = a[start_h:end_h, start_w:end_w,:,:]
                #print 'PIXEL h:%i, w:%i' % (h, w)
                #print image_patch.shape, a.shape, start_w, end_w
                self.z[h,w,:,:] = np.sum(image_patch * W, axis = (0,1,2)) + self.b
                #print 'shape',new[h,w,:,:].shape, self.b.shape
   
        self.a = self.g(self.z)
        #sys.exit()

        if gradient_check:
            W = W_temp

        return self.a


    def grads(self, a_prev, N):
        I, J, C, N = self.error.shape     
        self.dw = np.zeros(self.w.shape)
        self.db = np.zeros(self.b.shape)
        
        error = self.error[:,:,np.newaxis,:,:]

        for k in range(self.f):
            for l in range(self.f):
                axis1 = self.s * np.arange(I) + k
                axis2 = self.s * np.arange(J) + l
                a_prev_part = a_prev[np.ix_(axis1, axis2)]
                a_prev_part = a_prev_part[:,:,:,np.newaxis,:]
                
                self.dw[k,l,:,:] = 1./N * np.sum(error * a_prev_part, axis = (0,1,4))
        
        self.db = 1./N * np.sum(self.error, axis=(0,1,3), keepdims=True)
                
    
    def get_error(self, back_err):
        self.error = back_err * self.g(self.z, derivative=True)
        return self.error

    def backward(self, verbose=True, mode=2):
        if mode == 1:
            return self.backward1(verbose)
        elif mode == 2:
            return self.backward2(verbose)

    def backward1(self, verbose=False):
        height = width = self.s*(self.z.shape[0] - 1) - 2*self.p + self.f #this is the shape of previous activation
        channels = self.c_prev
        n = self.z.shape[3]
        self.backerr = np.zeros((height, width, channels, n))
         
        mins, maxs = self.lower_upper_summation_indices(height, self.f, self.s)
        if verbose: print '\tbackward: using the following summations\n\t\tmins', mins, '\n\t\tmaxs', maxs, '\n\t\terror shape used:', self.error.shape, 'backerr shape', self.backerr.shape
        for h in range(height):
            for w in range(width):
                low = np.arange(mins[h], maxs[h])
                up = np.arange(mins[w], maxs[w])
                if verbose: print '\t\th=%i w=%i' % (h, w), 'low', low, 'up', up
                if verbose: print 'error part'
                if verbose: print self.error.shape
                error_part = self.error[np.ix_(low,up)]#self.error[low,up,:,:]
                if verbose: print error_part.shape
                error_part = error_part[:,:,np.newaxis,:,:]
                if verbose: print error_part.shape

                axis1 = h - self.s * low
                axis2 = w - self.s * up
                if verbose: print 'W part'
                if verbose: print self.w.shape
                W_part = self.w[np.ix_(axis1, axis2)]
                if verbose: print W_part.shape
                W_part = W_part[...,np.newaxis]
                if verbose: print W_part.shape
                
                if verbose: print 'error_part', error_part.shape,'*', 'W_part', W_part.shape
                back_part = np.sum(error_part * W_part, axis=(0,1,3))
                if verbose: print 'back_part = error_part*W_part shape ',back_part.shape
                self.backerr[h,w,:,:] = back_part
        if verbose: print '\tcalculated back_err_%s for next shallow layer. shape:' % (self.l),  self.backerr.shape
        return self.backerr 
        
    def backward2(self, verbose=False):
        height = width = self.s*(self.z.shape[0] - 1) - 2*self.p + self.f #this is the shape of previous activation
        channels = self.c_prev
        n = self.z.shape[3]
        self.backerr = np.zeros((height, width, channels, n))
        
        I_max, J_max, C_max = self.z.shape[:3]
        
        for h in range(height):
            for w in range(width):
                I = np.arange(I_max)
                J = np.arange(J_max)
                if verbose: print 'h:%i, w:%s' % ( h,w)
                idx_I = h - self.s * I
                mask_I = idx_I > 0

                idx_J = w - self.s * J
                mask_J = idx_J > 0

            
                idx_I = idx_I[mask_I]
                idx_J = idx_J[mask_J]
                error_part = self.error[np.ix_(I[mask_I], J[mask_J])]
                if verbose: print error_part.shape
                error_part = error_part[:,:,np.newaxis,:,:]
                if verbose: print error_part.shape

                if verbose: print self.w.shape, idx_I, idx_J
                W_part = self.w[np.ix_(idx_I, idx_J)]
                if verbose: print W_part.shape
                W_part = W_part[...,np.newaxis]
                if verbose: print W_part.shape
                
                if verbose: print 'error_part', error_part.shape,'*', 'W_part', W_part.shape
                back_part = np.sum(error_part * W_part, axis=(0,1,3))
                if verbose: print 'back_part = error_part*W_part shape ',back_part.shape
                self.backerr[h,w,:,:] = back_part
        if verbose: print '\tcalculated back_err_%s for next shallow layer. shape:' % (self.l),  self.backerr.shape
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

    x = np.random.randint(1,5,(28,28,1,20))

    conv = Conv2D((5,5,1,3), 1, 0, relu, 'integers')
    z, a = conv.forward(x)

    w = conv.w
    f1 = w[:,:,0,0]
    f2 = w[:,:,0,1]
    img = x[:,:,0,0]
    a1 = a[:,:,0,0]
    a2 = a[:,:,1,0]

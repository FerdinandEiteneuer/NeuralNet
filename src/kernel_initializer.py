import numpy as np

def integers(shape):
    return np.random.randint(-3,3, size=shape)

def glorot_uniform(shape):
    output_dim, input_dim = shape[:2]
    bound = 6.0/np.sqrt(output_dim + input_dim)
    w = np.random.uniform(-bound, bound, size=shape)
    return w

def normal(shape):
    w = 0.01 * np.random.normal(size=shape)
    return w
   
def xavier(shape):
    return None

kernels = { 'glorot_uniform': glorot_uniform,\
            'normal': normal,\
            'xavier': xavier,\
            'integers': integers}


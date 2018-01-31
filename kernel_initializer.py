import numpy as np

def glorot_uniform(shape):
    output_dim, input_dim = shape[:2]
    bound = 6.0/np.sqrt(output_dim + input_dim)
    w = np.random.uniform(-bound, bound, size=shape)
    return w

def normal(shape):
    output_dim, input_dim = shape[:2]
    w = 0.01 * np.random.normal(size=(output_dim, input_dim))
    return w
   
def xavier(shape):
    return None

kernels = { 'glorot_uniform': glorot_uniform,\
            'normal': normal,\
            'xavier': xavier}

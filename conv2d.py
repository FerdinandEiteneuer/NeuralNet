from kernel_initializer import kernels
import numpy as np

class Conv2D:
    def __init__(self, filtersize, channels, stride, padding, kernel_initializer):
        self.f = filtersize
        self.s = stride
        self.p = padding
        self.c_prev = channels[1]
        self.c = channels[0]
        self.initialize_params

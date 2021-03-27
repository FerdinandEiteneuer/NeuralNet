import numpy as np

class Regularizer:

    def __call__(self, layer, param_id):
        self.layer = layer
        self.param_id = param_id
        return self

    @property
    def param(self):
        try:
            return getattr(self.layer, self.param_id)
        except AttributeError as e:
            print(f'ERROR: regularizer {self.__class__.__name__} could not access param {self.param_id} from layer {self.layer.layer_id} ({self.layer.name})')
            raise

    @property
    def batch_size(self):
        return self.layer.batch_size

class L2(Regularizer):

    def __init__(self, l2=0):
        self.l2 = l2

    def loss(self):
        return 1 / self.batch_size * (self.l2 * np.sum(self.param**2))

    def derivative(self):
        return 1 / self.batch_size * (2 * self.l2 * self.param)


class L1(Regularizer):

    def __init__(self, l1=0):
        self.l1 = l1

    def loss(self):
        return 1 / self.batch_size * (self.l1 * np.sum(np.abs(self.param)))

    def derivative(self):
        return 1 / self.batch_size * (self.l1 *  np.sign(self.param))


class L1_L2(L1, L2):

    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

    def loss(self):
        return L1.loss(self) + L2.loss(self)

    def derivative(self):
        return L1.derivative(self) + L2.derivative(self)

if __name__ == '__main__':

    par = 123

    class X: pass

    x=X()
    x.w = 2
    x.layer_id=1

    l1_2 = L1_L2(l1 = 1, l2 = 2)(x, 'w')


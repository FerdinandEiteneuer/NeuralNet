import numpy as np

class Regularizer:

    def set_param(self, param):
        self.param = param


class L2(Regularizer):

    def __init__(self, l2=0):
        self.l2 = l2
        self.param = None

    def cost(self):
        return self.l2 * np.sum(self.param**2)

    def derivative(self, param):
        return 2 * self.l2 * self.param


class L1(Regularizer):

    def __init__(self, l1=0):
        self.l1 = l1
        self.param = None

    def cost(self):
        return self.l1 * np.sum(np.abs(self.param))

    def derivative(self):
        return self.l1


class L1_L2(L1, L2):

    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2
        self.param = None

    def cost(self):
        return L1.cost(self) + L2.cost(self)

    def derivative(self):
        return L1.derivative(self) + L2.derivative(self)

if __name__ == '__main__':

    par = 123

    l1 = L1(l1 = 0)
    l1.set_param(10**12)

    l2 = L2(l2 = 0)
    l2.set_param(10**33)

    l1_2 = L1_L2(l1 = 1, l2 = 2)
    l1_2.set_param(par)

    print(l1_2.cost())

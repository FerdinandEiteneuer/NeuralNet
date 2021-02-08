import itertools
import numpy as np

class Layer():
    ''' Base Layer'''

    _ids = itertools.count(0)

    def __init__(self):
        self.layer_id = next(self.__class__._ids)  # layer number automatically generated.

    def __str__(self):
        '''
        possible result of __str__:
        dense_1 (Dense)              (10, 10)                 110
        '''
        name = self.__class__.__name__                     # e.g: Dense
        lower_with_id = f'{name.lower()}_{self.layer_id}'  # e.g: dense_1

        shape = self.w.shape if hasattr(self, 'w') else 'input data storage helper layer'

        if hasattr(self, 'w') and hasattr(self, 'b'):
            n_parameters = np.product(self.w.shape) + np.product(self.b.shape)
        else:
            n_parameters = 'n/a'

        s = f'{lower_with_id} ({name}){"":<13} {str(shape):<25} {n_parameters}'

        return s

#    def __str__(self):
#        return f'layer {self.layer_id} (input storing layer)'

    def __call__(self, a):
        return self.forward(a)

    def forward(self, a):
        '''The forward bethod of the base layer passes the activation along.'''
        return a


if __name__ == '__main__':
    l = Layer()
    print(str(l))

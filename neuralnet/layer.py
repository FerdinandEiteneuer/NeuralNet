import itertools
import numpy as np

class Layer():
    ''' Base Layer'''

    _ids = itertools.count(0)

    def __init__(self, layer_id = None):
        #self.layer_id = next(self.__class__._ids)  # layer number automatically generated.
        #self.layer_id = next(self._ids)  # layer number automatically generated.
        self.layer_id = layer_id
        self.class_layer_id = None
        self.output_dim = None

    def __str__(self):
        '''
        possible result of __str__:
        dense_1 (Dense)              (10, 10)                 110
        '''
        name = self.__class__.__name__                     # e.g: Dense
        lower_with_id = f'{name.lower()}_{self.class_layer_id}'  # e.g: dense_1

        shape = self.w.shape if hasattr(self, 'w') else 'n/a'

        if hasattr(self, 'w') and hasattr(self, 'b'):
            n_parameters = np.product(self.w.shape) + np.product(self.b.shape)
        else:
            n_parameters = 'n/a'


        output_dim = self.output_dim + (None, )

        s = f'{lower_with_id} ({name}){"":<11} {str(output_dim):<25} {n_parameters}'

        return s

    def __call__(self, a):
        return self.forward(a)

    def forward(self, a):
        '''The forward bethod of the base layer passes the activation along.'''
        return a


if __name__ == '__main__':
    l = Layer()
    print(str(l))

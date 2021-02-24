import itertools
import numpy as np
import abc


class Layer():
    ''' Base Layer'''

    _ids = itertools.count(0)

    def __init__(self, layer_id = None):
        self.layer_id = layer_id
        self.class_layer_id = None
        self.output_dim = None

    def __str__(self):
        name = f'({self.__class__.__name__})'  # e.g: (Dense)
        lower_with_id = f'{self.__class__.__name__.lower()}_{self.class_layer_id}'  # e.g: dense_1

        shape = self.w.shape if hasattr(self, 'w') else 'n/a'

        if hasattr(self, 'w') and hasattr(self, 'b'):
            n_parameters = np.product(self.w.shape) + np.product(self.b.shape)
        else:
            n_parameters = 'n/a'


        output_dim = tuple(self.output_dim) + (None, )

        #s = f'{lower_with_id + " " + name:29}{str(output_dim):26}{n_parameters:<}'
        s = f'{self.name + " " + name:29}{str(output_dim):26}{n_parameters:<}'

        #possible result of __str__:
        #dense_1 (Dense)              (10, None)                 110

        return s

    def __call__(self, a):
        return self.forward(a)

    def forward(self, a):
        '''The forward bethod of the base layer passes the activation along.'''
        return a

    @property
    def name(self):
        return f'{self.__class__.__name__.lower()}_{self.class_layer_id}'  # e.g: dense_1

if __name__ == '__main__':
    l = Layer()
    print(str(l))

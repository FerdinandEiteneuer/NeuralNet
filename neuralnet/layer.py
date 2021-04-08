import itertools
import numpy as np
import abc


class Layer():
    """ Base Layer"""

    _ids = itertools.count(0)

    def __init__(self, layer_id = None):
        self.layer_id = layer_id
        self.class_layer_id = None
        self.output_dim = None
        self.kernel_regularizer = None
        self.bias_regularizer = None
        self.batch_size = None
        self.dropout_seed = 0
        self.trainable_parameters = []

    @property
    def n_parameters(self):
        n = 0
        for par in self.trainable_parameters:
            n += np.product(getattr(self, par).shape)
        return n

    def __str__(self):
        """
        possible result of __str__:
        dense_1 (Dense)              (10, None)                 110
        """
        class_name = '(' + self.__class__.__name__ + ')'  # e.g: (Dense)

        n_parameters = 0

        for par in self.trainable_parameters:
            n_parameters += np.product(getattr(self, par).shape)

        #if hasattr(self, 'w') and hasattr(self, 'b'):
        #    n_parameters = np.product(self.w.shape) + np.product(self.b.shape)
        #elif hasattr(self, 'γ') and hasattr(self, 'β'):
        #    n_parameters = np.product(self.γ.shape) + np.product(self.β.shape)
        #else:
        #    n_parameters = '0'


        output_dim = tuple(self.output_dim) + (None, )

        s = f'{self.name + " " + class_name:34}{str(output_dim):26}{self.n_parameters:<}'
        return s

    def __call__(self, a, mode='test'):
        return self.forward(a, mode)

    def forward(self, a):
        """The forward bethod of the base layer passes the activation along."""
        return a

    def loss_from_regularizers(self):
        loss = 0
        if self.kernel_regularizer:
            loss += self.kernel_regularizer.loss()
        if self.bias_regularizer:
            loss += self.bias_regularizer.loss()
        return loss

    @property
    def name(self):
        return f'{self.__class__.__name__.lower()}_{self.class_layer_id}'  # e.g: dense_1

    def get_size(self, mode='G'):
        """
        Returns size of layer. Counts only the numpy arrays.
        """
        variables = ['w','dw', 'b', 'db', 'a','z','x']
        size = 0
        for var in dir(c):
            try:
                size += getattr(self, var).nbytes
            except AttributeError:
                pass

        K = 1024
        factor = {'1': 1, None: 1, 'K': K, 'M': K**2, 'G': K**3}
        return size / factor[mode]

if __name__ == '__main__':
    l = Layer()
    print(str(l))

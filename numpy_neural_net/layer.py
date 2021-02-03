import itertools

class MetaLayer(type):

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._ids = itertools.count(0)  # each new layer object gets a layer id


class Layer():
#class Layer(metaclass=MetaLayer):
    ''' Base Layer'''

    _ids = itertools.count(0)

    def __init__(self):
        self.layer_id = next(self.__class__._ids)  # layer number automatically generated.

    def __str__(self):
        return f'layer {self.layer_id} (input storing layer)'

    def __call__(self, a):
        return self.forward(a)

    def forward(self, a):
        '''The forward bethod of the base layer passes the activation along.'''
        return a

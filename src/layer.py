import itertools

class MetaLayer(type):

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._ids = itertools.count(0)  # each new layer object gets a layer id


class Layer(metaclass=MetaLayer):
    ''' Base Layer'''
    def __call__(self, a):
        return self.forward(a)

    def forward(self, a):
        return a

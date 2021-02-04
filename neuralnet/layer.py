import itertools

class Layer():
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

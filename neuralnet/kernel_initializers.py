import numpy as np
import inspect

__all__ = ['glorot_uniform', 'normal', 'xavier', 'integers']


def normal(shape, scale=0.001):
    return scale * np.random.normal(size=shape)

def integers(shape):
    return np.random.randint(0, 5, size=shape)

def xavier(shape):
    bound = np.sqrt(6/np.sum(shape))
    return np.random.uniform(-bound, bound, size=shape)

def glorot_uniform(shape):
    return xavier(shape)


def create(initializer, shape, **kwargs):
    '''
    TODO: is this good programming style?
    '''

    if isinstance(initializer, str):

        if initializer in __all__:
            initializer = eval(initializer)
        else:
            raise ValueError(f'no kernel initializer named {initializer}')

    elif not callable(initializer):

        raise ValueError(f'{initializer} is not a valid kernel initializer')


    return initializer(shape, **kwargs)

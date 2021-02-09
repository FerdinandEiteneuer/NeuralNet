import numpy as np
import inspect

__all__ = ['glorot_uniform', 'normal', 'xavier', 'integers']

def glorot_uniform(shape):
    output_dim, input_dim = shape[:2]
    bound = 6.0/np.sqrt(output_dim + input_dim)
    return np.random.uniform(-bound, bound, size=shape)

def normal(shape):
    return 0.01 * np.random.normal(size=shape)

def integers(shape):
    return np.random.randint(0, 5, size=shape)

def xavier(shape):
    raise NotImplementedError


def create(initializer, shape):
    '''
    TODO: is this good programming style?
    '''

    if isinstance(initializer, str):

        if initializer in __all__:
            initializer = eval(initializer)
        else:
            raise ValueError(f'no kernel initializer named {initializer}')

    elif callable(initializer):

        sig = inspect.signature(initializer)

        if len(sig.parameters) != 1:
            raise ValueError(f'{initializer} is not a valid kernel initializer')


    return initializer(shape)

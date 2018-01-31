import abc

class Layer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def feed_forward(w, a, b):
        return

class fc(Layer):
    pass

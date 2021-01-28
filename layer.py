import abc

class Layer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feed_forward(w, a, b):
        return

class fc(Layer):
    pass

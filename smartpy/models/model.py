import theano.tensor as T
from collections import OrderedDict


class Model(object):
    def get_gradients(self, loss):
        gparams = T.grad(loss, self.parameters.values())
        gradients = dict(zip(self.parameters.values(), gparams))
        return gradients, OrderedDict()

    @property
    def parameters(self):
        raise NotImplementedError("Subclass of 'Model' must implement property 'parameters'")

    def save(path):
        raise NotImplementedError("Subclass of 'Model' must implement 'save(path)'")

    @classmethod
    def load(path):
        raise NotImplementedError("Subclass of 'Model' must implement 'load(path)'")

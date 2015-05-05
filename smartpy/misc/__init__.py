import inspect

import os
import numpy as np
from os.path import join as pjoin
from .utils import load_dict_from_json_file, save_dict_to_json_file


class HyperparamsMeta(type):
    def __new__(cls, name, parents, dct):
        if '__init__' in dct:
            if '__hyperparams_types__' not in dct:
                dct['__hyperparams_types__'] = {}

            argnames, varargs, varkw, defaults = inspect.getargspec(dct['__init__'])
            init_funct = dct['__init__']

            def init_wrapper(obj, *args, **kwargs):
                if not hasattr(obj, '__hyperparams__'):
                    obj.__hyperparams__ = {}

                obj.__hyperparams__.update(dict(zip(argnames[:len(args)], args)))
                obj.__hyperparams__.update(kwargs)

                init_funct(obj, *args, **kwargs)

            dct['__init__'] = init_wrapper

        # we need to call type.__new__ to complete the initialization
        return super(HyperparamsMeta, cls).__new__(cls, name, parents, dct)


class HasHyperparams(object):
    __metaclass__ = HyperparamsMeta


class HasParams(object):
    @property
    def params(self):
        if not hasattr(self, '_params'):
            self._params = {}

        return self._params


def saveable(cls):
    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        save_dict_to_json_file(pjoin(path, "meta.json"), {"name": self.__class__.__name__})

        if issubclass(self.__class__, HasHyperparams):
            save_dict_to_json_file(pjoin(path, "hyperparams.json"), self.__hyperparams__)

        if issubclass(self.__class__, HasParams):
            params = {param_name: param.get_value() for param_name, param in self.params.items()}
            np.savez(pjoin(path, "params.npz"), **params)

    def load(cls, path):
        meta = load_dict_from_json_file(pjoin(path, "meta.json"))
        assert meta['name'] == cls.__name__

        hyperparams = {}
        if issubclass(cls, HasHyperparams):
            hyperparams = load_dict_from_json_file(pjoin(path, "hyperparams.json"))

        model = cls(**hyperparams)

        if issubclass(cls, HasParams):
            params = np.load(pjoin(path, "params.npz"))
            for param_name, param in model.params.item():
                param.set_value(params[param_name])

        return model

    setattr(cls, 'save', save)
    setattr(cls, 'load', classmethod(load))
    return cls

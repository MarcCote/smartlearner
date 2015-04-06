from __future__ import print_function

import sys
import argparse
from time import time
from itertools import izip

import theano
import hashlib

#from collections import defaultdict


def sharedX(value, name=None, borrow=False):
    """ Transform value into a shared variable of type floatX """
    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)


class Timer():
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))


def generate_uid_from_string(value):
    """ Create unique identifier from a string. """
    return hashlib.sha256(value).hexdigest()


class RegistryMeta(type):
    # we use __init__ rather than __new__ here because we want
    # to modify attributes of the class *after* they have been
    # created
    def __init__(cls, name, bases, dct):
        if not hasattr(cls, 'registry'):
            # this is the base class.  Create an empty registry
            cls.registry = {}
        else:
            # this is a derived class.  Add cls to the registry
            cls.registry[name] = cls

        super(RegistryMeta, cls).__init__(name, bases, dct)


class HyperparamsMeta(RegistryMeta):
    def __new__(cls, name, parents, dct):
        # create a class_id if it's not specified
        if '__hyperparams__' not in dct:
            dct['__hyperparams__'] = {}

        if '__optional__' not in dct:
            dct['__optional__'] = []

        # we need to call type.__new__ to complete the initialization
        return super(HyperparamsMeta, cls).__new__(cls, name, parents, dct)


def build_custom_type(name, cls):
    def custom_type(string):
        values = string.split()

        nb_required_hyperparams = len(cls.__hyperparams__) - len(cls.__optional__)

        if len(values) < nb_required_hyperparams:
            raise argparse.ArgumentTypeError("Missing required value(s) for {0}!".format(name))

        try:
            hyperparams = []
            for value, (hyperparam_name, hyperparam_type) in izip(values, cls.__hyperparams__.items()):
                hyperparams.append(hyperparam_type(value))

            update_rule = cls(*hyperparams)
        except Exception as e:
            raise argparse.ArgumentTypeError("{} {}".format(cls, e.message))

        return update_rule
    return custom_type


def create_argument_group_from_hyperparams_registry(parser, registry, dest, title=""):
    group = parser.add_argument_group(title)

    for name, cls in registry.items():

        required_hyperparams = []
        optional_hyperparams = []
        for hyperparam in cls.__hyperparams__.keys():
            if hyperparam in cls.__optional__:
                optional_hyperparams.append('[' + hyperparam + ']')
            else:
                required_hyperparams.append(hyperparam)

        custom_type = build_custom_type(name, cls)
        metavar = " ".join(required_hyperparams + optional_hyperparams)
        group.add_argument("--" + name, type=custom_type, metavar=metavar, dest=dest, action="append", help=cls.__doc__)

    return group


# class CustomDefaultDict(defaultdict):
#     def __init__(self, *args, **kwargs):
#         defaultdict.__init__(self, *args, **kwargs)

#     def __getitem__(self, key):
#         return defaultdict.__getitem__(self, str(key))

#     def __setitem__(self, key, val):
#         defaultdict.__setitem__(self, str(key), val)


# class CustomDict(dict):
#     def __getitem__(self, key):
#         return dict.__getitem__(self, str(key))

#     def __setitem__(self, key, val):
#         dict.__setitem__(self, str(key), val)

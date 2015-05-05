import inspect


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

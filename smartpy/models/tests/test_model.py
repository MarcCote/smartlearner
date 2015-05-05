import unittest
import tempfile as tmp
import shutil
from numpy.testing import assert_array_equal
from nose.tools import assert_equal, assert_true

import os
from os.path import join as pjoin
import numpy as np
import theano
import theano.tensor as T

from smartpy import Model
from smartpy.tests import tmp_folder
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.utils import save_dict_to_json_file, load_dict_from_json_file


dtype = theano.config.floatX


class Perceptron(Model):
    def __init__(self, input_size, output_size, activation_type="sigmoid"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation_type
        self.activation_function = ACTIVATION_FUNCTIONS[activation_type]
        self.W = theano.shared(value=np.zeros((input_size, output_size), dtype=dtype), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(output_size, dtype=dtype), name='b', borrow=True)

    @property
    def parameters(self):
        return {'W': self.W, 'b': self.b}

    def fprop(self, X):
        preactivation = T.dot(X, self.W) + self.b
        output = self.activation_function(preactivation)
        return output

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        hyperparameters = {'input_size': self.input_size,
                           'output_size': self.output_size,
                           'activation_type': self.activation_type}
        save_dict_to_json_file(pjoin(path, "meta.json"), {"name": self.__class__.__name__})
        save_dict_to_json_file(pjoin(path, "hyperparams.json"), hyperparameters)

        params = {param_name: param.get_value() for param_name, param in self.parameters.items()}
        np.savez(pjoin(path, "params.npz"), **params)

    @classmethod
    def load(cls, path):
        meta = load_dict_from_json_file(pjoin(path, "meta.json"))
        assert meta['name'] == cls.__name__

        hyperparams = load_dict_from_json_file(pjoin(path, "hyperparams.json"))

        model = cls(**hyperparams)
        parameters = np.load(pjoin(path, "params.npz"))
        for param_name, param in model.parameters.items():
            param.set_value(parameters[param_name])

        return model


class TestModel(unittest.TestCase):

    def setUp(self):
        self._base_dir = tmp.mkdtemp()

        self.input_size = 2
        self.output_size = 1
        self.activation_type = "sigmoid"
        self.W = np.arange(self.input_size*self.output_size, dtype=theano.config.floatX).reshape((self.input_size, self.output_size))
        self.b = np.arange(self.output_size, dtype=theano.config.floatX)
        self.model = Perceptron(input_size=self.input_size, output_size=self.output_size, activation_type=self.activation_type)
        self.model.W.set_value(self.W)
        self.model.b.set_value(self.b)

    def tearDown(self):
        shutil.rmtree(self._base_dir)

    def test_model_save_load(self):
        with tmp_folder(pjoin(self._base_dir, "model")) as path:
            # Test save
            self.model.save(path)
            assert_true(os.path.isdir(path))
            assert_true(os.path.isfile(pjoin(path, 'meta.json')))
            assert_true(os.path.isfile(pjoin(path, 'hyperparams.json')))
            assert_true(os.path.isfile(pjoin(path, 'params.npz')))

            meta = load_dict_from_json_file(pjoin(path, 'meta.json'))
            assert_equal(meta['name'], 'Perceptron')

            hyperparams = load_dict_from_json_file(pjoin(path, 'hyperparams.json'))
            assert_equal(hyperparams['input_size'], self.input_size)
            assert_equal(hyperparams['output_size'], self.output_size)

            params = np.load(pjoin(path, 'params.npz'))
            assert_array_equal(params['W'], self.W)
            assert_array_equal(params['b'], self.b)

            # Test load
            model2 = Perceptron.load(path)
            assert_equal(model2.input_size, self.input_size)
            assert_equal(model2.output_size, self.output_size)
            assert_equal(model2.activation_type, self.activation_type)
            assert_array_equal(self.model.W.get_value(), self.W)
            assert_array_equal(self.model.b.get_value(), self.b)

    def test_get_gradient(self):
        target = 0
        X = np.array([0, 1], dtype=dtype)
        loss = target - self.model.fprop(X)[0]
        gradients, updates = self.model.get_gradients(loss)

        #TODO: check if gradients are correct

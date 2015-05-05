import unittest
import tempfile as tmp
import shutil
from nose.tools import assert_equal, assert_true

import os
from os.path import join as pjoin
import numpy as np
import theano
import theano.tensor as T

from smartpy import SmartModel
from smartpy.tests import tmp_folder
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.utils import load_dict_from_json_file


class Perceptron(SmartModel):
    def __init__(self, input_size, output_size, activation_function="sigmoid"):
        self.activation_function = ACTIVATION_FUNCTIONS[activation_function]
        self.W = theano.shared(value=np.zeros((input_size, output_size), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(output_size, dtype=theano.config.floatX), name='b', borrow=True)

        # Define params of the model
        self.params['W'] = self.W
        self.params['b'] = self.b

    def fprop(self, X):
        preactivation = T.dot(X, self.W) + self.b
        output = self.activation_function(preactivation)
        return output


class TestSmartModel(unittest.TestCase):

    def setUp(self):
        self._base_dir = tmp.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._base_dir)

    def test_model_save(self):
        input_size = 2
        output_size = 1
        activation_function = "sigmoid"
        model = Perceptron(input_size=input_size, output_size=output_size, activation_function=activation_function)

        with tmp_folder(pjoin(self._base_dir, "model")) as path:
            model.save(path)
            assert_true(os.path.isdir(path))
            assert_true(os.path.isfile(pjoin(path, 'meta.json')))
            assert_true(os.path.isfile(pjoin(path, 'hyperparams.json')))
            assert_true(os.path.isfile(pjoin(path, 'params.npz')))

            meta = load_dict_from_json_file(pjoin(path, 'meta.json'))
            assert_equal(meta['name'], 'Perceptron')

            hyperparams = load_dict_from_json_file(pjoin(path, 'hyperparams.json'))
            assert_equal(hyperparams['input_size'], input_size)
            assert_equal(hyperparams['output_size'], output_size)

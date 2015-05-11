import theano
import theano.tensor as T

import numpy as np

from smartpy.models import Model
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.weights_initializer import WeightsInitializer


class ConvolutionalDeepNADE(Model):
    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_activation="sigmoid",
                 tied_weights=False,
                 ordering_seed=None,
                 *args, **kwargs):
        super(ConvolutionalDeepNADE, self).__init__(*args, **kwargs)

        # Set the hyperparameters of the model (must be JSON serializable)
        self.hyperparams['input_size'] = input_size
        self.hyperparams['hidden_size'] = hidden_size
        self.hyperparams['hidden_activation'] = hidden_activation
        self.hyperparams['tied_weights'] = tied_weights
        self.hyperparams['ordering_seed'] = ordering_seed

        # Internal attributes of the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_activation = ACTIVATION_FUNCTIONS[hidden_activation]
        self.tied_weights = tied_weights
        self.ordering_seed = ordering_seed

        # Define layers weights and biases (a.k.a parameters)
        self.W = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='W', borrow=True)
        self.bhid = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.bvis = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)

        # Set parameters that will be used in theano.grad
        self.parameters.extend([self.W, self.bhid, self.bvis])

        self.V = self.W
        if not tied_weights:
            self.V = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='V', borrow=True)
            self.parameters.append(self.V)

    def initialize(self, weights_initialization=None):
        """ Initialize weights of the model.

        Parameters
        ----------
        weights_initialization : (see smartpy.misc.weights_initializer.WeightsInitializer)
        """
        if weights_initialization is None:
            weights_initialization = WeightsInitializer().uniform

        self.W.set_value(weights_initialization(self.W.get_value().shape))

        if not self.tied_weights:
            self.V.set_value(weights_initialization(self.V.get_value().shape))

    def fprop(self, input, return_output_preactivation=False):
        """ Returns the theano graph that computes the fprop given `input` """
        input = input[:, self.ordering]  # Does not matter if ordering is the default one, indexing is fast.
        input_times_W = input.T[:, :, None] * self.W[:, None, :]

        # This uses the SplitOp which isn't available yet on the GPU.
        # acc_input_times_W = T.concatenate([T.zeros_like(input_times_W[[0]]), T.cumsum(input_times_W, axis=0)[:-1]], axis=0)
        # Hack to stay on the GPU
        acc_input_times_W = T.cumsum(input_times_W, axis=0)
        acc_input_times_W = T.set_subtensor(acc_input_times_W[1:], acc_input_times_W[:-1])
        acc_input_times_W = T.set_subtensor(acc_input_times_W[0, :], 0.0)

        acc_input_times_W += self.bhid[None, None, :]
        h = self.hidden_activation(acc_input_times_W)

        pre_output = T.sum(h * self.V[:, None, :], axis=2) + self.bvis[:, None]
        output = T.nnet.sigmoid(pre_output)

        if return_output_preactivation:
            return output.T, pre_output.T

        return output.T

    def get_nll(self, input, target):
        """ Returns the theano graph that computes the NLL given `input` and `target`. """

        target = target[:, self.ordering]  # Does not matter if ordering is the default one, indexing is fast.
        output, pre_output = self.fprop(input, return_output_preactivation=True)
        nll = T.sum(T.nnet.softplus(-target.T * pre_output.T + (1 - target.T) * pre_output.T), axis=0)

        return nll

    def mean_nll_loss(self, input, target):
        """ Returns the theano graph that computes the loss given `input` and `target`. """

        nll = self.get_nll(input, target)
        return nll.mean()

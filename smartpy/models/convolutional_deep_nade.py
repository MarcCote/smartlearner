from __future__ import division

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np

from smartpy.models import Model
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.weights_initializer import WeightsInitializer


class ConvolutionalDeepNADE(Model):
    """
    Parameters
    ----------
    image_shape: 2-tuple
        Shape of the input images i.e. (height, width).
    nb_channels: int
        Number of channels in the images (e.g. 3 for RGB images).
    kernel_shape: 2-tuple
        Shape of the convolutional kernels/filters i.e. (height, width).
    nb_kernels: int
        Number of kernels/filters to use in the convolutional layer.
    hidden_activation: str (optional)
        Name of the activation function to use in the fully-connected layer.
    ordering_seed: int (optional)
        Seed to use for generating ordering.
    """
    def __init__(self,
                 image_shape,
                 nb_channels,
                 kernel_shape,
                 nb_kernels,
                 hidden_activation="sigmoid",
                 ordering_seed=None,
                 *args, **kwargs):
        super(ConvolutionalDeepNADE, self).__init__(*args, **kwargs)

        # Set the hyperparameters of the model (must be JSON serializable)
        self.hyperparams['image_shape'] = image_shape
        self.hyperparams['nb_channels'] = nb_channels
        self.hyperparams['kernel_shape'] = kernel_shape
        self.hyperparams['nb_kernels'] = nb_kernels
        self.hyperparams['hidden_activation'] = hidden_activation
        self.hyperparams['ordering_seed'] = ordering_seed

        # Internal attributes of the model
        self.image_shape = image_shape
        self.nb_channels = nb_channels
        self.kernel_shape = kernel_shape
        self.nb_kernels = nb_kernels
        self.hidden_activation = ACTIVATION_FUNCTIONS[hidden_activation]
        self.ordering_seed = ordering_seed

        # Parameters of the convolutional layer (i.e. kernel weights and biases).
        W_shape = (self.nb_kernels, self.nb_channels) + self.kernel_shape
        self.W = theano.shared(value=np.zeros(W_shape, dtype=theano.config.floatX), name='W', borrow=True)
        self.bhid = theano.shared(value=np.zeros(self.nb_kernels, dtype=theano.config.floatX), name='bhid', borrow=True)

        # Parameters of the fully-connected layer (i.e. layer weights and biases).
        input_size = np.prod(image_shape)
        hidden_size = np.prod((self.nb_kernels,) + self.kernel_shape)
        self.V = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='V', borrow=True)
        self.bvis = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)

        # Set parameters that will be used in theano.grad
        self.parameters.extend([self.W, self.bhid, self.V, self.bvis])

        # Hack: ordering_mask will be modified by a training task in the trainer. (see the training scripts).
        # This hack is necessary as the random generator of Theano does not support functions randint nor shuffle.
        # Its shape will be (batch_size, image_height, image_width.)
        self.ordering_mask = theano.shared(np.array([], ndmin=3, dtype=theano.config.floatX), name='ordering_mask', borrow=True)

    def initialize(self, weights_initialization=None):
        """ Initialize weights of the model.

        Parameters
        ----------
        weights_initialization : (see smartpy.misc.weights_initializer.WeightsInitializer)
        """
        if weights_initialization is None:
            weights_initialization = WeightsInitializer().uniform

        self.W.set_value(weights_initialization(self.W.get_value().shape))
        self.V.set_value(weights_initialization(self.V.get_value().shape))

    def fprop(self, input, return_output_preactivation=False):
        """ Returns the theano graph that computes the fprop given `input`.

        Parameters
        ----------
        input: 4D tensor
            Batch of images. The shape is (batch_size, nb_channels, images_height, images_width).

        Return
        ------

        Notes
        -----
        We assume `self.ordering_mask` is modified externally (by a training task) before each update.
        """
        # Hack: Sampling the ordering is done by a training task (TODO: needs to implement shuffle in Theano).
        input_masked = input * self.ordering_mask

        pre_conv_out = conv(input_masked, filters=self.W, filter_shape=self.kernel_shape, image_shape=self.image_shape)
        conv_out = self.hidden_activation(pre_conv_out + self.bhid.dimshuffle('x', 0, 'x', 'x'))

        # The fully-connected layer operates on 2D matrices of shape (batch_size, num_pixels)
        # This will generate a matrix of shape (batch_size, nb_kernels * kernel_height * kernel_width).
        conv_out_flatten = conv_out.flatten(2)

        pre_output = T.dot(conv_out_flatten, self.V) + self.hvis
        output = T.nnet.sigmoid(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def get_nll(self, input, target):
        """ Returns the theano graph that computes the NLL given `input` and `target`. """

        _, pre_output = self.fprop(input, return_output_preactivation=True)
        cross_entropies = T.nnet.softplus(-target.T * pre_output.T + (1 - target.T) * pre_output.T)
        cross_entropies_masked = cross_entropies * (1-self.ordering_mask)

        nll = T.sum(cross_entropies_masked, axis=0)
        # TODO: check if scaling factor is correct
        D = np.prod(self.image_shape)
        d = self.ordering_mask.sum(axis=1)  # Batch of ordering.
        weighted_nll = nll * (D / (D-d+1))
        return weighted_nll

    def mean_nll_loss(self, input, target):
        """ Returns the theano graph that computes the loss given `input` and `target`. """

        nll = self.get_nll(input, target)
        return nll.mean()

from __future__ import division

import re
import pickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np
from os.path import join as pjoin

from smartpy.models import Model
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.weights_initializer import WeightsInitializer

from abc import ABCMeta, abstractmethod
from types import MethodType
from time import time

from smartpy.misc.utils import load_dict_from_json_file


def generate_blueprints(seed, image_shape):
    rng = np.random.RandomState(seed)

    # Generate convoluational layers blueprint
    convnet_blueprint = []
    convnet_blueprint_inverse = []  # We want convnet to be symmetrical
    nb_layers = rng.randint(1, 5+1)
    layer_id_first_conv = -1
    for layer_id in range(nb_layers):
        if image_shape <= 2:
            # Too small
            continue

        if rng.rand() <= 0.8:
            # 70% of the time do a convolution
            nb_filters = rng.choice([16, 32, 64, 128, 256, 512])
            filter_shape = rng.randint(2, min(image_shape, 8+1))
            image_shape = image_shape-filter_shape+1

            filter_shape = (filter_shape, filter_shape)
            convnet_blueprint.append("{nb_filters}@{filter_shape}(valid)".format(nb_filters=nb_filters,
                                                                                 filter_shape="x".join(map(str, filter_shape))))
            convnet_blueprint_inverse.append("{nb_filters}@{filter_shape}(full)".format(nb_filters=nb_filters,
                                                                                        filter_shape="x".join(map(str, filter_shape))))
            if layer_id_first_conv == -1:
                layer_id_first_conv = layer_id
        else:
            # 30% of the time do a max pooling
            pooling_shape = rng.randint(2, 5+1)
            while not image_shape % pooling_shape == 0:
                pooling_shape = rng.randint(2, 5+1)

            image_shape = image_shape / pooling_shape
            #pooling_shape = 2  # For now, we limit ourselves to pooling of 2x2
            pooling_shape = (pooling_shape, pooling_shape)
            convnet_blueprint.append("max@{pooling_shape}".format(pooling_shape="x".join(map(str, pooling_shape))))
            convnet_blueprint_inverse.append("up@{pooling_shape}".format(pooling_shape="x".join(map(str, pooling_shape))))

    # Need to make sure there is only one channel in output
    infos = convnet_blueprint_inverse[layer_id_first_conv].split("@")[-1]
    convnet_blueprint_inverse[layer_id_first_conv] = "1@" + infos

    # Connect first part and second part of the convnet
    convnet_blueprint = "->".join(convnet_blueprint) + "->" + "->".join(convnet_blueprint_inverse[::-1])

    # Generate fully connected layers blueprint
    fullnet_blueprint = []
    nb_layers = rng.randint(1, 4+1)  # Deep NADE only used up to 4 hidden layers
    for layer_id in range(nb_layers):
        hidden_size = 500  # Deep NADE only used hidden layer of 500 units
        fullnet_blueprint.append("{hidden_size}".format(hidden_size=hidden_size))

    fullnet_blueprint.append("784")  # Output layer
    fullnet_blueprint = "->".join(fullnet_blueprint)

    return convnet_blueprint, fullnet_blueprint


class LayerDecorator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def decorate(self, layer):
        raise NotImplementedError("Subclass of 'LayerDecorator' must implement 'decorate(layer)'.")


class MaxPoolDecorator(LayerDecorator):
    def __init__(self, pool_shape, ignore_border=True):
        self.pool_shape = pool_shape
        self.ignore_border = ignore_border

    def decorate(self, layer):
        self._decorate_fprop(layer)
        self._decorate_infer_shape(layer)
        self._decorate_to_text(layer)

    def _decorate_fprop(self, layer):
        layer_fprop = layer.fprop

        def decorated_fprop(instance, input, return_output_preactivation=False):
            if return_output_preactivation:
                output, pre_output = layer_fprop(input, return_output_preactivation)
                pooled_output = downsample.max_pool_2d(output, self.pool_shape, ignore_border=self.ignore_border)
                pooled_pre_output = downsample.max_pool_2d(pre_output, self.pool_shape, ignore_border=self.ignore_border)
                return pooled_output, pooled_pre_output

            output = layer_fprop(input, return_output_preactivation)
            pooled_output = downsample.max_pool_2d(output, self.pool_shape, ignore_border=self.ignore_border)
            return pooled_output

        layer.fprop = MethodType(decorated_fprop, layer, type(layer))

    def _decorate_infer_shape(self, layer):
        layer_infer_shape = layer.infer_shape

        def decorated_infer_shape(instance, input_shape):
            input_shape = layer_infer_shape(input_shape)
            output_shape = np.array(input_shape[2:]) / np.array(self.pool_shape)
            if self.ignore_border:
                output_shape = np.floor(output_shape)
            else:
                output_shape = np.ceil(output_shape)

            output_shape = input_shape[:2] + tuple(output_shape.astype(int))
            return output_shape

        layer.infer_shape = MethodType(decorated_infer_shape, layer, type(layer))

    def _decorate_to_text(self, layer):
        layer_to_text = layer.to_text

        def decorated_to_text(instance):
            text = layer_to_text()
            text += " -> max@{0}".format("x".join(map(str, self.pool_shape)))
            return text

        layer.to_text = MethodType(decorated_to_text, layer)


class UpSamplingDecorator(LayerDecorator):
    def __init__(self, up_shape):
        self.up_shape = up_shape

    def decorate(self, layer):
        self._decorate_fprop(layer)
        self._decorate_infer_shape(layer)
        self._decorate_str(layer)

    def _upsample_tensor(self, input):
        shp = input.shape
        upsampled_out = T.zeros((shp[0], shp[1], shp[2]*self.up_shape[0], shp[3]*self.up_shape[1]), dtype=input.dtype)
        upsampled_out = T.set_subtensor(upsampled_out[:, :, ::self.up_shape[0], ::self.up_shape[1]], input)
        return upsampled_out

    def _decorate_fprop(self, layer):
        layer_fprop = layer.fprop

        def decorated_fprop(instance, input, return_output_preactivation=False):
            if return_output_preactivation:
                output, pre_output = layer_fprop(input, return_output_preactivation)
                upsampled_output = self._upsample_tensor(output)
                upsampled_pre_output = self._upsample_tensor(pre_output)
                return upsampled_output, upsampled_pre_output

            output = layer_fprop(input, return_output_preactivation)
            upsampled_output = self._upsample_tensor(output)
            return upsampled_output

        layer.fprop = MethodType(decorated_fprop, layer, type(layer))

    def _decorate_infer_shape(self, layer):
        layer_infer_shape = layer.infer_shape

        def decorated_infer_shape(instance, input_shape):
            input_shape = layer_infer_shape(input_shape)
            output_shape = np.array(input_shape[2:]) * np.array(self.up_shape)
            output_shape = input_shape[:2] + tuple(output_shape.astype(int))
            return output_shape

        layer.infer_shape = MethodType(decorated_infer_shape, layer, type(layer))

    def _decorate_str(self, layer):
        layer_to_text = layer.to_text

        def decorated_to_text(instance):
            text = layer_to_text()
            text += " -> up@{0}".format("x".join(map(str, self.up_shape)))
            return text

        layer.to_text = MethodType(decorated_to_text, layer, type(layer))


class Layer(object):
    def __init__(self, size, name=""):
        self.size = size
        self.name = name
        self.prev_layer = None
        self.next_layer = None

    @property
    def hyperparams(self):
        return {}

    @property
    def params(self):
        return {}

    def allocate(self):
        pass

    def initialize(self, weight_initializer):
        pass

    def fprop(self, input, return_output_preactivation=False):
        if return_output_preactivation:
            return input, input

        return input

    def to_text(self):
        return self.name

    def __str__(self):
        return self.to_text()

    def infer_shape(self, input_shape):
        return input_shape


class ConvolutionalLayer(Layer):
    def __init__(self, nb_filters, filter_shape, border_mode, activation="sigmoid"):
        super(ConvolutionalLayer, self).__init__(size=nb_filters)
        self.nb_filters = nb_filters
        self.filter_shape = filter_shape
        self.border_mode = border_mode
        self.activation = activation

        self.activation_fct = ACTIVATION_FUNCTIONS[self.activation]

    def allocate(self):
        # Allocating memory for parameters
        nb_input_feature_maps = self.prev_layer.size
        W_shape = (self.nb_filters, nb_input_feature_maps) + self.filter_shape
        self.W = theano.shared(value=np.zeros(W_shape, dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(self.nb_filters, dtype=theano.config.floatX), name='b', borrow=True)
        print W_shape, self.border_mode

    def initialize(self, weight_initializer):
        if weight_initializer is None:
            weight_initializer = WeightsInitializer().uniform

        self.W.set_value(weight_initializer(self.W.get_value().shape))

    @property
    def hyperparams(self):
        return {'nb_filters': self.nb_filters,
                'filter_shape': self.filter_shape,
                'border_mode': self.border_mode,
                'activation': self.activation}

    @property
    def params(self):
        return {'W': self.W,
                'b': self.b}

    def fprop(self, input, return_output_preactivation=False):
        conv_out = conv.conv2d(input, filters=self.W, border_mode=self.border_mode)
        # TODO: Could be faster if pooling was done here instead
        pre_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        output = self.activation_fct(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def to_text(self):
        return "{0}@{1}({2})".format(self.nb_filters, "x".join(map(str, self.filter_shape)), self.border_mode)

    def infer_shape(self, input_shape):
        assert len(input_shape) == 4
        if self.border_mode == "valid":
            output_shape = np.array(input_shape[2:]) - np.array(self.filter_shape) + 1
        else:
            output_shape = np.array(input_shape[2:]) + np.array(self.filter_shape) - 1

        output_shape = (input_shape[0], self.nb_filters) + tuple(output_shape.astype(int))
        return output_shape


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, hidden_size, activation="sigmoid"):
        super(FullyConnectedLayer, self).__init__(size=hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.activation_fct = ACTIVATION_FUNCTIONS[self.activation]

    def allocate(self):
        # Allocating memory for parameters
        W_shape = (self.input_size, self.hidden_size)
        self.W = theano.shared(value=np.zeros(W_shape, dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(self.hidden_size, dtype=theano.config.floatX), name='b', borrow=True)
        print W_shape

    def initialize(self, weight_initializer):
        if weight_initializer is None:
            weight_initializer = WeightsInitializer().uniform

        self.W.set_value(weight_initializer(self.W.get_value().shape))

    @property
    def hyperparams(self):
        return {'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'activation': self.activation}

    @property
    def params(self):
        return {'W': self.W,
                'b': self.b}

    def fprop(self, input, return_output_preactivation=False):
        pre_output = T.dot(input, self.W) + self.b
        output = self.activation_fct(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def to_text(self):
        return "{0}".format(self.hidden_size)

    def infer_shape(self, input_shape):
        return (input_shape[0], self.hidden_size)


class DeepModel(Model):
    def __init__(self, layers, name=""):
        self.layers = layers
        self.name = name

    def fprop(self, input, return_output_preactivation=False):
        output = input
        for layer in self.layers:
            output, pre_output = layer.fprop(output, return_output_preactivation=True)

        if return_output_preactivation:
            return output, pre_output

        return output

    @property
    def hyperparams(self):
        hyperparams = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.hyperparams.items():
                hyperparams[self.name + "layer{0}_{1}".format(i, k)] = v

        return hyperparams

    @property
    def params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.params.items():
                # TODO: Changing the variable name till first smartpy's PR is merged.
                v.name = self.name + "layer{0}_{1}".format(i, k)
                params[v.name] = v

        return params

    def initialize(self, weight_initializer):
        if weight_initializer is None:
            weight_initializer = WeightsInitializer().uniform

        for layer in self.layers:
            layer.initialize(weight_initializer)

    def __str__(self):
        return " -> ".join(map(str, self.layers))

    def infer_shape(self, input_shape):
        out_shape = input_shape
        for layer in self.layers:
            out_shape = layer.infer_shape(out_shape)

        return out_shape


class DeepConvNADE(Model):
    def __init__(self,
                 image_shape,
                 nb_channels,
                 convnet_layers,
                 fullnet_layers,
                 ordering_seed=1234,
                 consider_mask_as_channel=False):
        #super(DeepConvNADE, self).__init__()
        self.has_convnet = len(convnet_layers) > 0
        self.has_fullnet = len(fullnet_layers) > 0

        self.convnet = DeepModel(convnet_layers, name="convnet_")
        self.fullnet = DeepModel(fullnet_layers, name="fullnet_")

        self.image_shape = image_shape
        self.nb_channels = nb_channels
        self.ordering_seed = ordering_seed
        self.consider_mask_as_channel = consider_mask_as_channel

        if self.has_convnet:
            # Make sure the convolutional network outputs 'np.prod(image_shape)' units.
            input_shape = (1, nb_channels) + image_shape
            out_shape = self.convnet.infer_shape(input_shape)
            if out_shape != (1, 1) + image_shape:
                raise ValueError("(Convnet) Output shape mismatched: {} != {}".format(out_shape, (1, 1) + image_shape))

        if self.fullnet:
            # Make sure the fully connected network outputs 'np.prod(image_shape)' units.
            input_shape = (1, int(np.prod(image_shape)))
            out_shape = self.fullnet.infer_shape(input_shape)
            if out_shape != (1, int(np.prod(image_shape))):
                raise ValueError("(Fullnet) Output shape mismatched: {} != {}".format(out_shape, (1, int(np.prod(image_shape)))))

    @property
    def hyperparams(self):
        #hyperparams = super(DeepConvNADE, self).hyperparams
        hyperparams = {}
        hyperparams.update(self.convnet.hyperparams)
        hyperparams.update(self.fullnet.hyperparams)
        hyperparams['image_shape'] = self.image_shape
        hyperparams['nb_channels'] = self.nb_channels
        hyperparams['ordering_seed'] = self.ordering_seed
        hyperparams['consider_mask_as_channel'] = self.consider_mask_as_channel
        return hyperparams

    @property
    def params(self):
        #params = super(DeepConvNADE, self).params
        params = {}
        params.update(self.convnet.params)
        params.update(self.fullnet.params)
        return params

    @property
    def parameters(self):
        """ TODO: choose between parameters or params """
        return self.params.values()

    def initialize(self, weight_initializer=None):
        if weight_initializer is None:
            weight_initializer = WeightsInitializer().uniform

        #super(DeepConvNADE, self).initialize(weight_initializer)
        self.convnet.initialize(weight_initializer)
        self.fullnet.initialize(weight_initializer)

    def fprop(self, input, mask_o_lt_d, return_output_preactivation=False):
        """ Returns the theano graph that computes the fprop given an `input` and an `ordering`.

        Parameters
        ----------
        input: 2D matrix
            Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).

        mask_o_lt_d: 1D vector or 2D matrix
            Mask allowing only the $d-1$ first dimensions in the ordering i.e. $x_i : i \in o_{<d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_lt_d.shape[0] == input.shape[0].
        """
        pre_output_convnet = 0
        if self.has_convnet:
            input_masked = input * mask_o_lt_d

            nb_input_feature_maps = self.nb_channels
            if self.consider_mask_as_channel:
                nb_input_feature_maps += 1
                if mask_o_lt_d.ndim == 1:
                    # TODO: changed this hack
                    input_masked = T.concatenate([input_masked, T.ones_like(input_masked)*mask_o_lt_d], axis=1)
                else:
                    input_masked = T.concatenate([input_masked, mask_o_lt_d], axis=1)

            # Hack: input_masked is a 2D matrix instead of a 4D tensor, but we have all the information to fix that.
            input_masked = input_masked.reshape((-1, nb_input_feature_maps) + self.image_shape)

            # fprop through all layers
            #_, pre_output = super(DeepConvNADE, self).fprop(input_masked, return_output_preactivation=True)
            _, pre_output = self.convnet.fprop(input_masked, return_output_preactivation=True)

            # This will generate a matrix of shape (batch_size, nb_kernels * kernel_height * kernel_width).
            pre_output_convnet = pre_output.flatten(2)

        pre_output_fully = 0
        if self.has_fullnet:
            input_masked_fully_connected = input * mask_o_lt_d
            if self.consider_mask_as_channel:
                if mask_o_lt_d.ndim == 1:
                    input_masked_fully_connected = T.concatenate([input_masked_fully_connected, T.ones_like(input_masked_fully_connected)*mask_o_lt_d], axis=1)
                else:
                    input_masked_fully_connected = T.concatenate([input_masked_fully_connected, mask_o_lt_d], axis=1)

            _, pre_output_fully = self.fullnet.fprop(input_masked_fully_connected, return_output_preactivation=True)

            #fully_conn_hidden = T.nnet.sigmoid(T.dot(input_masked_fully_connected, self.W) + self.bhid)
            #pre_output_fully = T.dot(fully_conn_hidden, self.V)

        pre_output = pre_output_convnet + pre_output_fully
        output = T.nnet.sigmoid(pre_output)  # Force use of sigmoid for the output layer

        if return_output_preactivation:
            return output, pre_output

        return output

    def get_cross_entropies(self, input, mask_o_lt_d):
        """ Returns the theano graph that computes the cross entropies for all input dimensions
        allowed by the mask `1-mask_o_lt_d` (i.e. the complementary mask).

        Parameters
        ----------
        input: 2D matrix
            Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).

        mask_o_lt_d: 1D vector or 2D matrix
            Mask allowing only the $d-1$ first dimensions in the ordering i.e. $x_i : i \in o_{<d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_lt_d.shape[0] == input.shape[0].
        """
        _, pre_output = self.fprop(input, mask_o_lt_d, return_output_preactivation=True)
        cross_entropies = T.nnet.softplus(-input * pre_output + (1 - input) * pre_output)
        cross_entropies_masked = cross_entropies * (1-mask_o_lt_d)
        return cross_entropies_masked

    def get_nll_estimate(self, input, mask_o_lt_d):
        """ Returns the theano graph that computes the NLL estimate given `input` and `mask_o_lt_d`.

        Parameters
        ----------
        input: 2D matrix
            Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).

        mask_o_lt_d: 1D vector or 2D matrix
            Mask allowing only the $d-1$ first dimensions in the ordering i.e. $x_i : i \in o_{<d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_lt_d.shape[0] == input.shape[0].
        """
        cross_entropies = self.get_cross_entropies(input, mask_o_lt_d)
        nll_estimate = T.sum(cross_entropies, axis=1)

        # Scaling factor
        D = np.float32(np.prod(self.image_shape))
        d = mask_o_lt_d.sum() if mask_o_lt_d.ndim == 1 else mask_o_lt_d.sum(axis=1)
        weighted_nll_estimate = nll_estimate * (D / (D-d+1))
        return weighted_nll_estimate

    def mean_nll_estimate_loss(self, input, mask_o_lt_d):
        """ Returns the theano graph that computes the loss given `input`. """
        nll = self.get_nll_estimate(input, mask_o_lt_d)
        return nll.mean()

    def lnp_x_o_d_given_x_o_lt_d(self, input, mask_o_d, mask_o_lt_d):
        """ Returns the theano graph that computes $ln p(x_{o_d}|x_{o_{<d}})$.

        Parameters
        ----------
        input: 2D matrix
            Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).

        mask_o_d: 1D vector or 2D matrix
            Mask allowing only the $d$-th dimension in the ordering i.e. $x_{o_d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_d.shape[0] == input.shape[0].

        mask_o_lt_d: 1D vector or 2D matrix
            Mask allowing only the $d-1$ first dimensions in the ordering i.e. $x_i : i \in o_{<d}$.
            If 1D vector, the same mask is applied to all images in the batch.
            If 2D matrix, each images in the batch will have a different mask meaning that
            mask_o_lt_d.shape[0] == input.shape[0].
        """
        # Retrieves cross entropies for all possible $p(x_i|x_{o_{<d}})$ where $i \in o_{>=d}$.
        cross_entropies = self.get_cross_entropies(input, mask_o_lt_d)
        # We keep only the cross entropy corresponding to $p(x_{o_d}|x_{o_{<d}})$
        cross_entropies_masked = cross_entropies * mask_o_d
        ln_dth_conditional = -T.sum(cross_entropies_masked, axis=1)  # Keep only the d-th conditional
        return ln_dth_conditional

    def nll_of_x_given_o(self, input, ordering):
        """ Returns the theano graph that computes $-ln p(\bx|o)$.

        Parameters
        ----------
        input: 1D vector
            One image with shape (nb_channels * images_height * images_width).

        ordering: 1D vector of int
            List of pixel indices representing the order of the input dimensions.
        """

        D = int(np.prod(self.image_shape))
        mask_o_d = T.zeros((D, D), dtype=theano.config.floatX)
        mask_o_d = T.set_subtensor(mask_o_d[T.arange(D), ordering], 1.)

        mask_o_lt_d = T.cumsum(mask_o_d, axis=0)
        mask_o_lt_d = T.set_subtensor(mask_o_lt_d[1:], mask_o_lt_d[:-1])
        mask_o_lt_d = T.set_subtensor(mask_o_lt_d[0, :], 0.)

        input = T.tile(input[None, :], (D, 1))
        nll = -T.sum(self.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d))
        return nll

    def __str__(self):
        text = ""

        if self.has_convnet:
            text += self.convnet.__str__() + " -> output\n"

        if self.has_fullnet:
            text += self.fullnet.__str__() + " -> output\n"

        return text[:-1]  # Do not return last \n

    # @classmethod
    # def create(cls, loaddir="./", hyperparams_filename="hyperparams", params_filename="params"):
    #     hyperparams = load_dict_from_json_file(pjoin(loaddir, hyperparams_filename + ".json"))

    #     hidden_activation = [v for k, v in hyperparams.items() if "activation" in k][0]
    #     builder = DeepConvNADEBuilder(image_shape=hyperparams["image_shape"],
    #                                   nb_channels=hyperparams["nb_channels"],
    #                                   ordering_seed=hyperparams["ordering_seed"],
    #                                   consider_mask_as_channel=hyperparams["consider_mask_as_channel"],
    #                                   hidden_activation=hidden_activation)

    #     # Rebuild convnet layers
    #     layers_id = set()
    #     for k, v in hyperparams.items():
    #         if k.startswith("convnet_layer"):
    #             layer_id = int(re.findall("convnet_layer([0-9]+)_", k)[0])
    #             layers_id.add(layer_id)

    #     convnet_blueprint = ""
    #     for layer_id in sorted(layers_id):
    #         border_mode = hyperparams["convnet_layer{}_border_mode".format(layer_id)]
    #         nb_filters = hyperparams["convnet_layer{}_nb_filters".format(layer_id)]
    #         filter_shape = tuple(hyperparams["convnet_layer{}_filter_shape".format(layer_id)])
    #         convnet_blueprint += ""

    #     # Rebuild fullnet layers

    #     if args.convnet_blueprint is not None:
    #         builder.build_convnet_from_blueprint(convnet_blueprint)

    #     if args.fullnet_blueprint is not None:
    #         builder.build_fullnet_from_blueprint(args.fullnet_blueprint)

    #     model = builder.build()
    #     model.load(loaddir, params_filename)
    #     return model

    def build_sampling_function(self, seed=None):
        from smartpy.misc.utils import Timer
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        rng = np.random.RandomState(seed)
        theano_rng = RandomStreams(rng.randint(2**30))

        # Build theano function
        # $X$: batch of inputs (flatten images)
        input = T.matrix('input')
        # $o_d$: index of d-th dimension in the ordering.
        mask_o_d = T.vector('mask_o_d')
        # $o_{<d}$: indices of the d-1 first dimensions in the ordering.
        mask_o_lt_d = T.vector('mask_o_lt_d')

        output = self.fprop(input, mask_o_lt_d)
        probs = T.sum(output*mask_o_d, axis=1)
        bits = theano_rng.binomial(p=probs, size=probs.shape, n=1, dtype=theano.config.floatX)
        sample_bit_plus = theano.function([input, mask_o_d, mask_o_lt_d], [bits, probs])

        def _sample(nb_samples, ordering_seed=1234):
            rng = np.random.RandomState(ordering_seed)
            D = int(np.prod(self.image_shape))
            ordering = np.arange(D)
            rng.shuffle(ordering)

            with Timer("Generating {} samples from ConvDeepNADE".format(nb_samples)):
                o_d = np.zeros((D, D), dtype=theano.config.floatX)
                o_d[np.arange(D), ordering] = 1

                o_lt_d = np.cumsum(o_d, axis=0)
                o_lt_d[1:] = o_lt_d[:-1]
                o_lt_d[0, :] = 0

                samples = np.zeros((nb_samples, D), dtype="float32")
                samples_probs = np.zeros((nb_samples, D), dtype="float32")
                for d, bit in enumerate(ordering):
                    if d % 100 == 0:
                        print d
                    bits, probs = sample_bit_plus(samples, o_d[d], o_lt_d[d])
                    samples[:, bit] = bits
                    samples_probs[:, bit] = probs

                return samples_probs

        return _sample


class DeepConvNADEBuilder(object):
    def __init__(self,
                 image_shape,
                 nb_channels,
                 ordering_seed=1234,
                 consider_mask_as_channel=False,
                 hidden_activation="sigmoid"):

        self.image_shape = image_shape
        self.nb_channels = nb_channels
        self.ordering_seed = ordering_seed
        self.consider_mask_as_channel = consider_mask_as_channel
        self.hidden_activation = hidden_activation

        self.convnet_layers = []
        self.fullnet_layers = []

    def stack(self, layer, layers):
        # Connect layers, if needed
        if len(layers) > 0:
            layers[-1].next_layer = layer
            layer.prev_layer = layers[-1]

        layers.append(layer)

    def build(self):
        for layer in self.convnet_layers:
            layer.allocate()

        for layer in self.fullnet_layers:
            layer.allocate()

        model = DeepConvNADE(image_shape=self.image_shape,
                             nb_channels=self.nb_channels,
                             convnet_layers=self.convnet_layers,
                             fullnet_layers=self.fullnet_layers,
                             ordering_seed=self.ordering_seed,
                             consider_mask_as_channel=self.consider_mask_as_channel)

        return model

    def build_convnet_from_blueprint(self, blueprint):
        """
        Example:
        64@5x5(valid) -> max@2x2 -> 256@2x2(valid) -> 256@2x2(full) -> up@2x2 -> 64@5x5(full)
        """
        input_layer = Layer(size=self.nb_channels + self.consider_mask_as_channel, name="convnet_input")
        self.stack(input_layer, self.convnet_layers)

        layers_blueprint = map(str.strip, blueprint.split("->"))

        for layer_blueprint in layers_blueprint:
            infos = layer_blueprint.lower().split("@")
            if infos[0] == "max":
                pool_shape = tuple(map(int, infos[1].split("x")))
                MaxPoolDecorator(pool_shape).decorate(self.convnet_layers[-1])
            elif infos[0] == "up":
                up_shape = tuple(map(int, infos[1].split("x")))
                UpSamplingDecorator(up_shape).decorate(self.convnet_layers[-1])
            else:
                nb_filters = int(infos[0])
                if "valid" in infos[1]:
                    border_mode = "valid"
                elif "full" in infos[1]:
                    border_mode = "full"
                else:
                    raise ValueError("Unknown border mode for '{}'".format(layer_blueprint))

                filter_shape = tuple(map(int, infos[1][:-len("(" + border_mode + ")")].split("x")))
                layer = ConvolutionalLayer(nb_filters=nb_filters,
                                           filter_shape=filter_shape,
                                           border_mode=border_mode,
                                           activation=self.hidden_activation)
                self.stack(layer, self.convnet_layers)

    def build_fullnet_from_blueprint(self, blueprint):
        """
        Example:
        500 -> 256 -> 300 -> 784
        """
        input_layer = Layer(size=int(np.prod(self.image_shape)) * 2**int(self.consider_mask_as_channel), name="fullnet_input")
        self.stack(input_layer, self.fullnet_layers)

        layers_blueprint = map(str.strip, blueprint.split("->"))

        for layer_blueprint in layers_blueprint:
            hidden_size = int(layer_blueprint)

            layer = FullyConnectedLayer(input_size=self.fullnet_layers[-1].size,
                                        hidden_size=hidden_size,
                                        activation=self.hidden_activation)
            self.stack(layer, self.fullnet_layers)

#
# TASKS
#
from smartpy.trainers.tasks import Task, ItemGetter
from smartpy.trainers.tasks import Evaluate


class DeepNadeOrderingTask(Task):
    """ This task changes the ordering before each update. """
    def __init__(self, D, batch_size, ordering_seed=1234):
        super(DeepNadeOrderingTask, self).__init__()
        self.rng = np.random.RandomState(ordering_seed)
        self.batch_size = batch_size
        self.D = D
        self.ordering_mask = theano.shared(np.zeros((batch_size, D), dtype=theano.config.floatX), name='ordering_mask', borrow=False)

    def pre_update(self, status):
        # Thanks to the broadcasting and `np.apply_along_axis`, we easily
        # generate `batch_size` orderings and compute their corresponding
        # $o_{<d}$ mask.
        d = self.rng.randint(self.D, size=(self.batch_size, 1))
        masks_o_lt_d = np.arange(self.D) < d
        map(self.rng.shuffle, masks_o_lt_d)  # Inplace shuffling each row.
        self.ordering_mask.set_value(masks_o_lt_d)

    def save(self, savedir="./"):
        filename = pjoin(savedir, "DeepNadeOrderingTask.pkl")
        pickle.dump(self.rng, open(filename, 'w'))

    def load(self, loaddir="./"):
        filename = pjoin(loaddir, "DeepNadeOrderingTask.pkl")
        self.rng = pickle.load(open(filename))


class DeepNadeTrivialOrderingsTask(Task):
    """ This task changes the ordering before each update.

    The ordering are sampled from the 8 trivial orderings.
    """
    def __init__(self, image_shape, batch_size, ordering_seed=1234):
        super(DeepNadeTrivialOrderingsTask, self).__init__()
        self.rng = np.random.RandomState(ordering_seed)
        self.batch_size = batch_size
        self.D = int(np.prod(image_shape))
        self.mask_o_d = theano.shared(np.zeros((batch_size, self.D), dtype=theano.config.floatX), name='mask_o_d', borrow=False)
        self.mask_o_lt_d = theano.shared(np.zeros((batch_size, self.D), dtype=theano.config.floatX), name='mask_o_lt_d', borrow=False)
        self.ordering_mask = self.mask_o_lt_d

        self.orderings = []
        base_ordering = np.arange(self.D).reshape(image_shape)
        # 8 trivial orderings
        # Top-left to bottom-right (row-major)
        self.orderings.append(base_ordering.flatten("C"))
        # Top-right to bottom-left (row-major)
        self.orderings.append(base_ordering[:, ::-1].flatten("C"))
        # Bottom-left to top-right (row-major)
        self.orderings.append(base_ordering[::-1, :].flatten("C"))
        # Bottom-right to top-left (row-major)
        self.orderings.append(base_ordering[::-1, ::-1].flatten("C"))
        # Top-left to bottom-right (column-major)
        self.orderings.append(base_ordering.flatten("F"))
        # Top-right to bottom-left (column-major)
        self.orderings.append(base_ordering[:, ::-1].flatten("F"))
        # Bottom-left to top-right (column-major)
        self.orderings.append(base_ordering[::-1, :].flatten("F"))
        # Bottom-right to top-left (column-major)
        self.orderings.append(base_ordering[::-1, ::-1].flatten("F"))

    def pre_update(self, status):
        # Compute the next $o_{<d}$ mask.
        idx_ordering = self.rng.randint(8)
        d = self.rng.randint(self.D, size=(self.batch_size, 1))
        masks_o_d = self.orderings[idx_ordering] == d
        masks_o_lt_d = self.orderings[idx_ordering] < d
        self.mask_o_d.set_value(masks_o_d)
        self.mask_o_lt_d.set_value(masks_o_lt_d)

    def save(self, savedir="./"):
        filename = pjoin(savedir, "DeepNadeOrderingTask.pkl")
        pickle.dump(self.rng, open(filename, 'w'))

    def load(self, loaddir="./"):
        filename = pjoin(loaddir, "DeepNadeOrderingTask.pkl")
        self.rng = pickle.load(open(filename))


class EvaluateDeepNadeNLLEstimateOnTrivial(Evaluate):
    """ This tasks compute the mean/stderr NLL estimate for a Deep NADE model.  """
    def __init__(self, conv_nade, dataset, batch_size=None, ordering_seed=1234):

        dataset_shared = dataset
        if isinstance(dataset, np.ndarray):
            dataset_shared = theano.shared(dataset, name='dataset', borrow=True)

        if batch_size is None:
            batch_size = len(dataset_shared.get_value())

        nb_batches = int(np.ceil(len(dataset_shared.get_value()) / batch_size))

        # Pre-generate the orderings that will be used to estimate the NLL of the Deep NADE model.
        D = int(np.prod(conv_nade.image_shape))
        ordering_task = DeepNadeTrivialOrderingsTask(conv_nade.image_shape, len(dataset_shared.get_value()), ordering_seed)

        # $X$: batch of inputs (flatten images)
        input = T.matrix('input')
        mask_o_d = theano.shared(np.zeros((batch_size, D), dtype=theano.config.floatX), name='mask_o_d', borrow=False)
        mask_o_lt_d = theano.shared(np.zeros((batch_size, D), dtype=theano.config.floatX), name='mask_o_lt_d', borrow=False)
        loss = T.mean(-conv_nade.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d))

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size]}
        compute_loss = theano.function([no_batch], loss, givens=givens, name="NLL Estimate")
        #theano.printing.pydotprint(compute_loss, '{0}_compute_nll_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

        def _nll_mean_and_std():
            nlls = np.zeros(len(dataset_shared.get_value()))
            for i in range(nb_batches):
                # Hack: Change ordering mask in the model before computing the NLL estimate.
                mask_o_d.set_value(ordering_task.mask_o_d.get_value()[i*batch_size:(i+1)*batch_size])
                mask_o_lt_d.set_value(ordering_task.mask_o_lt_d.get_value()[i*batch_size:(i+1)*batch_size])
                nlls[i*batch_size:(i+1)*batch_size] = compute_loss(i)

            return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

        super(EvaluateDeepNadeNLLEstimateOnTrivial, self).__init__(_nll_mean_and_std)

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def std(self):
        return ItemGetter(self, attribute=1)


class EvaluateDeepNadeNLLEstimate(Evaluate):
    """ This tasks compute the mean/stderr NLL estimate for a Deep NADE model.  """
    def __init__(self, conv_nade, dataset, ordering_mask, batch_size=None, ordering_seed=1234):

        dataset_shared = dataset
        if isinstance(dataset, np.ndarray):
            dataset_shared = theano.shared(dataset, name='dataset', borrow=True)

        if batch_size is None:
            batch_size = len(dataset_shared.get_value())

        nb_batches = int(np.ceil(len(dataset_shared.get_value()) / batch_size))

        # Pre-generate the orderings that will be used to estimate the NLL of the Deep NADE model.
        rng = np.random.RandomState(ordering_seed)
        D = dataset_shared.get_value().shape[1]
        d = rng.randint(D, size=(len(dataset_shared.get_value()), 1))
        masks_o_lt_d = np.arange(D) < d
        map(rng.shuffle, masks_o_lt_d)  # Inplace shuffling along axis=1.

        # $X$: batch of inputs (flatten images)
        input = T.matrix('input')
        loss = conv_nade.mean_nll_estimate_loss(input, ordering_mask)
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size]}
        compute_loss = theano.function([no_batch], loss, givens=givens, name="NLL Estimate")
        #theano.printing.pydotprint(compute_loss, '{0}_compute_nll_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

        def _nll_mean_and_std():
            nlls = np.zeros(len(dataset_shared.get_value()))
            for i in range(nb_batches):
                # Hack: Change ordering mask in the model before computing the NLL estimate.
                ordering_mask.set_value(masks_o_lt_d[i*batch_size:(i+1)*batch_size])
                nlls[i*batch_size:(i+1)*batch_size] = compute_loss(i)

            return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

        super(EvaluateDeepNadeNLLEstimate, self).__init__(_nll_mean_and_std)

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def std(self):
        return ItemGetter(self, attribute=1)


class EvaluateDeepNadeNLLParallel(Evaluate):
    """ This tasks compute the mean/stderr NLL (averaged across multiple orderings) for a Deep NADE model.

    Notes
    -----
    This is slow but tractable.
    """

    def __init__(self, conv_nade, dataset,
                 batch_size=None, no_part=1, nb_parts=1,
                 no_ordering=None, nb_orderings=8, orderings_seed=None):

        print "Part: {}/{}".format(no_part, nb_parts)
        part_size = int(np.ceil(len(dataset.get_value()) / nb_parts))
        dataset = dataset.get_value()[(no_part-1)*part_size:no_part*part_size]

        dataset = theano.shared(dataset, name='dataset', borrow=True)

        if batch_size is None:
            batch_size = len(dataset.get_value())

        #batch_size = min(batch_size, part_size)
        nb_batches = int(np.ceil(len(dataset.get_value()) / batch_size))

        # Generate the orderings that will be used to evaluate the Deep NADE model.
        D = dataset.get_value().shape[1]
        orderings = []
        if orderings_seed is None:
            base_ordering = np.arange(D).reshape(conv_nade.image_shape)
            # 8 trivial orderings
            # Top-left to bottom-right (row-major)
            orderings.append(base_ordering.flatten("C"))
            # Top-right to bottom-left (row-major)
            orderings.append(base_ordering[:, ::-1].flatten("C"))
            # Bottom-left to top-right (row-major)
            orderings.append(base_ordering[::-1, :].flatten("C"))
            # Bottom-right to top-left (row-major)
            orderings.append(base_ordering[::-1, ::-1].flatten("C"))
            # Top-left to bottom-right (column-major)
            orderings.append(base_ordering.flatten("F"))
            # Top-right to bottom-left (column-major)
            orderings.append(base_ordering[:, ::-1].flatten("F"))
            # Bottom-left to top-right (column-major)
            orderings.append(base_ordering[::-1, :].flatten("F"))
            # Bottom-right to top-left (column-major)
            orderings.append(base_ordering[::-1, ::-1].flatten("F"))
        else:
            rng = np.random.RandomState(orderings_seed)
            for i in range(nb_orderings):
                ordering = np.arange(D)
                rng.shuffle(ordering)
                orderings.append(ordering)

        if no_ordering is not None:
            orderings = [orderings[no_ordering]]

        masks_o_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_d', borrow=True)
        masks_o_lt_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_lt_d', borrow=True)

        # Build theano function
        # $X$: batch of inputs (flatten images)
        input = T.matrix('input')
        # $o_d$: index of d-th dimension in the ordering.
        mask_o_d = T.vector('mask_o_d')
        # $o_{<d}$: indices of the d-1 first dimensions in the ordering.
        mask_o_lt_d = T.vector('mask_o_lt_d')

        lnp_x_o_d_given_x_o_lt_d = conv_nade.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d)

        no_batch = T.iscalar('no_batch')
        d = T.iscalar('d')
        givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size],
                  mask_o_d: masks_o_d[d],
                  mask_o_lt_d: masks_o_lt_d[d]}
        compute_lnp_x_o_d_given_x_o_lt_d = theano.function([no_batch, d], lnp_x_o_d_given_x_o_lt_d, givens=givens, name="nll_of_x_o_d_given_x_o_lt_d")
        #theano.printing.pydotprint(compute_lnp_x_o_d_given_x_o_lt_d, '{0}_compute_lnp_x_o_d_given_x_o_lt_d_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

        def _nll():
            nlls = -np.inf * np.ones(len(dataset.get_value()))
            for o, ordering in enumerate(orderings):
                o_d = np.zeros((D, D), dtype=theano.config.floatX)
                o_d[np.arange(D), ordering] = 1
                masks_o_d.set_value(o_d)

                o_lt_d = np.cumsum(o_d, axis=0)
                o_lt_d[1:] = o_lt_d[:-1]
                o_lt_d[0, :] = 0
                masks_o_lt_d.set_value(o_lt_d)

                for i in range(nb_batches):
                    print "Batch: {0}/{1}".format(i+1, nb_batches)
                    ln_dth_conditionals = []
                    start = time()
                    for d in range(D):
                        if d % 100 == 0:
                            print "{0}/{1} dth conditional ({2:.2f} sec.)".format(d, D, time()-start)
                            start = time()

                        ln_dth_conditionals.append(compute_lnp_x_o_d_given_x_o_lt_d(i, d))

                    from ipdb import set_trace as dbg
                    dbg()

                    # We average p(x) on different orderings, if needed.
                    nlls[i*batch_size:(i+1)*batch_size] = np.logaddexp(nlls[i*batch_size:(i+1)*batch_size],
                                                                       -np.sum(np.vstack(ln_dth_conditionals).T, axis=1))

            nlls -= np.log(len(orderings))  # Average across all orderings
            return nlls

        super(EvaluateDeepNadeNLLParallel, self).__init__(_nll)


class EvaluateDeepNadeNLL(Evaluate):
    """ This tasks compute the mean/stderr NLL (averaged across multiple orderings) for a Deep NADE model.

    Notes
    -----
    This is slow but tractable.
    """

    def __init__(self, conv_nade, dataset, batch_size=None, nb_orderings=10, ordering_seed=1234):

        dataset_shared = dataset
        if isinstance(dataset, np.ndarray):
            dataset_shared = theano.shared(dataset, name='dataset', borrow=True)

        if batch_size is None:
            batch_size = len(dataset_shared.get_value())

        nb_batches = int(np.ceil(len(dataset_shared.get_value()) / batch_size))

        # Generate the orderings that will be used to evaluate the Deep NADE model.
        D = dataset_shared.get_value().shape[1]
        orderings = []
        if nb_orderings > 0:
            rng = np.random.RandomState(ordering_seed)
            for i in range(nb_orderings):
                ordering = np.arange(D)
                rng.shuffle(ordering)
                orderings.append(ordering)

        elif nb_orderings == 0:
            base_ordering = np.arange(D).reshape(conv_nade.image_shape)
            # 8 trivial orderings
            # Top-left to bottom-right (row-major)
            orderings.append(base_ordering.flatten("C"))
            # Top-right to bottom-left (row-major)
            orderings.append(base_ordering[:, ::-1].flatten("C"))
            # Bottom-left to top-right (row-major)
            orderings.append(base_ordering[::-1, :].flatten("C"))
            # Bottom-right to top-left (row-major)
            orderings.append(base_ordering[::-1, ::-1].flatten("C"))
            # Top-left to bottom-right (column-major)
            orderings.append(base_ordering.flatten("F"))
            # Top-right to bottom-left (column-major)
            orderings.append(base_ordering[:, ::-1].flatten("F"))
            # Bottom-left to top-right (column-major)
            orderings.append(base_ordering[::-1, :].flatten("F"))
            # Bottom-right to top-left (column-major)
            orderings.append(base_ordering[::-1, ::-1].flatten("F"))
        else:
            raise ValueError("Unknown value for 'nb_orderings': {0}".format(nb_orderings))

        masks_o_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_d', borrow=True)
        masks_o_lt_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_lt_d', borrow=True)

        # Build theano function
        # $X$: batch of inputs (flatten images)
        input = T.matrix('input')
        # $o_d$: index of d-th dimension in the ordering.
        mask_o_d = T.vector('mask_o_d')
        # $o_{<d}$: indices of the d-1 first dimensions in the ordering.
        mask_o_lt_d = T.vector('mask_o_lt_d')

        lnp_x_o_d_given_x_o_lt_d = conv_nade.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d)

        no_batch = T.iscalar('no_batch')
        d = T.iscalar('d')
        givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size],
                  mask_o_d: masks_o_d[d],
                  mask_o_lt_d: masks_o_lt_d[d]}
        compute_lnp_x_o_d_given_x_o_lt_d = theano.function([no_batch, d], lnp_x_o_d_given_x_o_lt_d, givens=givens, name="nll_of_x_o_d_given_x_o_lt_d")
        #theano.printing.pydotprint(compute_lnp_x_o_d_given_x_o_lt_d, '{0}_compute_lnp_x_o_d_given_x_o_lt_d_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

        def _nll_mean_and_std():
            nlls = -np.inf * np.ones(len(dataset_shared.get_value()))
            for o, ordering in enumerate(orderings):
                o_d = np.zeros((D, D), dtype=theano.config.floatX)
                o_d[np.arange(D), ordering] = 1
                masks_o_d.set_value(o_d)

                o_lt_d = np.cumsum(o_d, axis=0)
                o_lt_d[1:] = o_lt_d[:-1]
                o_lt_d[0, :] = 0
                masks_o_lt_d.set_value(o_lt_d)

                for i in range(nb_batches):
                    print "Batch {0}/{1}".format(i, nb_batches)
                    ln_dth_conditionals = []
                    start = time()
                    for d in range(D):
                        if d % 100 == 0:
                            print "{0}/{1} dth conditional ({2:.2f} sec.)".format(d, D, time()-start)
                            start = time()

                        ln_dth_conditionals.append(compute_lnp_x_o_d_given_x_o_lt_d(i, d))

                    # We average p(x) on different orderings
                    #nlls[i*batch_size:(i+1)*batch_size] += -np.sum(np.vstack(ln_dth_conditionals).T, axis=1)
                    nlls[i*batch_size:(i+1)*batch_size] = np.logaddexp(nlls[i*batch_size:(i+1)*batch_size], -np.sum(np.vstack(ln_dth_conditionals).T, axis=1))

            nlls -= np.log(len(orderings))  # Average across all orderings
            return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

        super(EvaluateDeepNadeNLL, self).__init__(_nll_mean_and_std)

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def std(self):
        return ItemGetter(self, attribute=1)

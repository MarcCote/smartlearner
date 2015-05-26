from __future__ import division

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

import numpy as np

from smartpy.models import Model
from smartpy.misc.utils import ACTIVATION_FUNCTIONS
from smartpy.misc.weights_initializer import WeightsInitializer


class DeepConvolutionalDeepNADE(Model):
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
                 list_of_kernel_shapes,
                 list_of_nb_kernels,
                 list_of_border_modes,
                 hidden_activation="sigmoid",
                 ordering_seed=None,
                 consider_mask_as_channel=False,
                 *args, **kwargs):
        super(DeepConvolutionalDeepNADE, self).__init__(*args, **kwargs)

        assert len(list_of_nb_kernels) == len(list_of_kernel_shapes)

        # Just to be sure we have tuples for shapes.
        image_shape = tuple(image_shape)
        list_of_kernel_shapes = map(tuple, list_of_kernel_shapes)

        # Set the hyperparameters of the model (must be JSON serializable)
        self.hyperparams['image_shape'] = image_shape
        self.hyperparams['nb_channels'] = nb_channels
        self.hyperparams['list_of_kernel_shapes'] = list_of_kernel_shapes
        self.hyperparams['list_of_nb_kernels'] = list_of_nb_kernels
        self.hyperparams['list_of_border_modes'] = list_of_border_modes
        self.hyperparams['hidden_activation'] = hidden_activation
        self.hyperparams['ordering_seed'] = ordering_seed
        self.hyperparams['consider_mask_as_channel'] = consider_mask_as_channel

        # Internal attributes of the model
        self.image_shape = image_shape
        self.nb_channels = nb_channels
        self.list_of_kernel_shapes = list_of_kernel_shapes
        self.list_of_nb_kernels = list_of_nb_kernels
        self.list_of_border_modes = list_of_border_modes
        self.hidden_activation = ACTIVATION_FUNCTIONS[hidden_activation]
        self.ordering_seed = ordering_seed
        self.consider_mask_as_channel = consider_mask_as_channel
        self.kernels = []
        self.kernel_biases = []

        # Add a final output convolutional layer, if needed.
        output_kernel_shape = np.array((0, 0))
        for kernel_shape, border_mode in zip(self.list_of_kernel_shapes, self.list_of_border_modes):
            if border_mode == "valid":
                output_kernel_shape -= np.array(kernel_shape) - 1
            elif border_mode == "full":
                output_kernel_shape += np.array(kernel_shape) - 1
            else:
                raise ValueError("Unknown border mode: {}".format(border_mode))

        if np.any(output_kernel_shape):
            self.list_of_kernel_shapes += [tuple(np.abs(output_kernel_shape)+1)]
            #self.list_of_nb_kernels += [int(np.prod(self.image_shape))]
            self.list_of_nb_kernels += [1]
            self.list_of_border_modes += ['valid'] if output_kernel_shape.sum() > 0 else ['full']

        if consider_mask_as_channel:
            self.nb_channels += 1

        nb_input_features = self.nb_channels

        # Parameters of the convolutional layers (i.e. kernels weights and biases).
        for layer_id, (nb_kernels, kernel_shape) in enumerate(zip(self.list_of_nb_kernels, self.list_of_kernel_shapes)):
            W_shape = (nb_kernels, nb_input_features) + kernel_shape
            W = theano.shared(value=np.zeros(W_shape, dtype=theano.config.floatX), name='W' + str(layer_id), borrow=True)
            bhid = theano.shared(value=np.zeros(nb_kernels, dtype=theano.config.floatX), name='bhid' + str(layer_id), borrow=True)
            nb_input_features = nb_kernels

            # Set parameters that will be used in theano.grad
            self.parameters.extend([W, bhid])
            self.kernels.append(W)
            self.kernel_biases.append(bhid)
            print W_shape, list_of_border_modes[layer_id]

    def build_sampling_function(self, seed=None):
        # Build sampling function
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
                    print d
                    bits, probs = sample_bit_plus(samples, o_d[d], o_lt_d[d])
                    samples[:, bit] = bits
                    samples_probs[:, bit] = probs

                return samples_probs
        return _sample

    def initialize(self, weights_initialization=None):
        """ Initialize weights of the model.

        Parameters
        ----------
        weights_initialization : (see smartpy.misc.weights_initializer.WeightsInitializer)
        """
        if weights_initialization is None:
            weights_initialization = WeightsInitializer().uniform

        for kernel in self.kernels:
            kernel.set_value(weights_initialization(kernel.get_value().shape))

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
        input_masked = input * mask_o_lt_d

        if self.consider_mask_as_channel:
            if mask_o_lt_d.ndim == 1:
                # TODO: changed this hack
                input_masked = T.concatenate([input_masked, T.ones_like(input_masked)*mask_o_lt_d], axis=1)
            else:
                input_masked = T.concatenate([input_masked, mask_o_lt_d], axis=1)

        # Hack: input_masked is a 2D matrix instead of a 4D tensor, but we have all the information to fix that.
        input_masked = input_masked.reshape((-1, self.nb_channels) + self.image_shape)

        out_previous_layer = input_masked
        for W, bhid, border_mode in zip(self.kernels, self.kernel_biases, self.list_of_border_modes):
            conv_out = conv.conv2d(out_previous_layer, filters=W, border_mode=border_mode)
            pre_output = conv_out + bhid.dimshuffle('x', 0, 'x', 'x')
            out_previous_layer = self.hidden_activation(pre_output)

        # This will generate a matrix of shape (batch_size, nb_kernels * kernel_height * kernel_width).
        pre_output = pre_output.flatten(2)
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



from smartpy.trainers.tasks import Task, ItemGetter
from smartpy.trainers.tasks import Evaluate


class DeepNadeOrderingTask(Task):
    """ This task changes the ordering before each update. """
    def __init__(self, D, batch_size, ordering_seed=None):
        super(DeepNadeOrderingTask, self).__init__()
        self.rng = np.random.RandomState(ordering_seed)
        self.batch_size = batch_size
        self.D = D
        self.ordering_mask = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='ordering_mask', borrow=True)

    def pre_update(self, status):
        # Thanks to the broadcasting and `np.apply_along_axis`, we easily
        # generate `batch_size` orderings and compute their corresponding
        # $o_{<d}$ mask.
        d = self.rng.randint(self.D, size=(self.batch_size, 1))
        masks_o_lt_d = np.arange(self.D) < d
        map(self.rng.shuffle, masks_o_lt_d)  # Inplace shuffling along axis=1.

        self.ordering_mask.set_value(masks_o_lt_d)


class EvaluateDeepNadeNLLEstimate(Evaluate):
    """ This tasks compute the mean/stderr NLL estimate for a Deep NADE model.  """
    def __init__(self, conv_nade, dataset, ordering_mask, batch_size=None, ordering_seed=42):

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


class EvaluateDeepNadeNLL(Evaluate):
    """ This tasks compute the mean/stderr NLL (averaged across multiple orderings) for a Deep NADE model.

    Notes
    -----
    This is slow but tractable.
    """

    def __init__(self, conv_nade, dataset, batch_size=None, nb_orderings=10, ordering_seed=42):

        dataset_shared = dataset
        if isinstance(dataset, np.ndarray):
            dataset_shared = theano.shared(dataset, name='dataset', borrow=True)

        if batch_size is None:
            batch_size = len(dataset_shared.get_value())

        nb_batches = int(np.ceil(len(dataset_shared.get_value()) / batch_size))

        # Generate the orderings that will be used to evaluate the Deep NADE model.
        D = dataset_shared.get_value().shape[1]
        rng = np.random.RandomState(ordering_seed)
        orderings = []
        for i in range(nb_orderings):
            ordering = np.arange(D)
            rng.shuffle(ordering)
            orderings.append(ordering)

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
        theano.printing.pydotprint(compute_lnp_x_o_d_given_x_o_lt_d, '{0}_compute_lnp_x_o_d_given_x_o_lt_d_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

        def _nll_mean_and_std():
            nlls = np.zeros(len(dataset_shared.get_value()))
            for o, ordering in enumerate(orderings):
                o_d = np.zeros((D, D), dtype=theano.config.floatX)
                o_d[np.arange(D), ordering] = 1
                masks_o_d.set_value(o_d)

                o_lt_d = np.cumsum(o_d, axis=0)
                o_lt_d[1:] = o_lt_d[:-1]
                o_lt_d[0, :] = 0
                masks_o_lt_d.set_value(o_lt_d)

                for i in range(nb_batches):
                    ln_dth_conditionals = []
                    for d in range(D):
                        ln_dth_conditionals.append(compute_lnp_x_o_d_given_x_o_lt_d(i, d))

                    nlls[i*batch_size:(i+1)*batch_size] += -np.sum(np.vstack(ln_dth_conditionals).T, axis=1)

            nlls /= len(orderings)  # Average across all orderings
            return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

        super(EvaluateDeepNadeNLL, self).__init__(_nll_mean_and_std)

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def std(self):
        return ItemGetter(self, attribute=1)

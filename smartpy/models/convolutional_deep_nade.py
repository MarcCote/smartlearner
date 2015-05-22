from __future__ import division

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

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
                 consider_mask_as_channel=False,
                 *args, **kwargs):
        super(ConvolutionalDeepNADE, self).__init__(*args, **kwargs)

        # Set the hyperparameters of the model (must be JSON serializable)
        self.hyperparams['image_shape'] = image_shape
        self.hyperparams['nb_channels'] = nb_channels
        self.hyperparams['kernel_shape'] = kernel_shape
        self.hyperparams['nb_kernels'] = nb_kernels
        self.hyperparams['hidden_activation'] = hidden_activation
        self.hyperparams['ordering_seed'] = ordering_seed
        self.hyperparams['consider_mask_as_channel'] = consider_mask_as_channel

        # Internal attributes of the model
        self.image_shape = image_shape
        self.nb_channels = nb_channels
        self.kernel_shape = kernel_shape
        self.nb_kernels = nb_kernels
        self.hidden_activation = ACTIVATION_FUNCTIONS[hidden_activation]
        self.ordering_seed = ordering_seed
        self.consider_mask_as_channel = consider_mask_as_channel

        if consider_mask_as_channel:
            self.nb_channels += 1

        # Parameters of the convolutional layer (i.e. kernel weights and biases).
        W_shape = (self.nb_kernels, self.nb_channels) + self.kernel_shape
        self.W = theano.shared(value=np.zeros(W_shape, dtype=theano.config.floatX), name='W', borrow=True)
        self.bhid = theano.shared(value=np.zeros(self.nb_kernels, dtype=theano.config.floatX), name='bhid', borrow=True)

        # Parameters of the fully-connected layer (i.e. layer weights and biases).
        input_size = np.prod(image_shape)
        hidden_size = self.nb_kernels * np.prod(np.array(self.image_shape) - np.array(self.kernel_shape) + 1)
        self.V = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='V', borrow=True)
        self.bvis = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)

        # Set parameters that will be used in theano.grad
        self.parameters.extend([self.W, self.bhid, self.V, self.bvis])

        # Hack: ordering_mask will be modified by a training task in the trainer. (see the training scripts).
        # This hack is necessary as the random generator of Theano does not support
        # functions randint nor shuffle. Its shape will be (batch_size, image_height, image_width.)
        self.ordering_mask = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='ordering_mask', borrow=True)

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

    def fprop(self, input, ordering, return_output_preactivation=False):
        """ Returns the theano graph that computes the fprop given an `input` and an `ordering`.

        Parameters
        ----------
        input: 2D matrix
            Batch of images. The shape is (batch_size, nb_channels * images_height * images_width).
            TODO: Change it to use 4D tensors.

        Return
        ------
        TODO
        """
        input_masked = input * ordering

        if self.consider_mask_as_channel:
            input_masked = T.concatenate([input_masked, ordering], axis=1)

        # Hack: input_masked is a 2D matrix instead of a 4D tensor, but we have all the information to fix that.
        input_masked = input_masked.reshape((-1, self.nb_channels) + self.image_shape)

        pre_conv_out = conv.conv2d(input_masked,
                                   filters=self.W,
                                   filter_shape=(self.nb_kernels, self.nb_channels) + self.kernel_shape,
                                   image_shape=(None, self.nb_channels) + self.image_shape)
        conv_out = self.hidden_activation(pre_conv_out + self.bhid.dimshuffle('x', 0, 'x', 'x'))

        # The fully-connected layer operates on 2D matrices of shape (batch_size, num_pixels)
        # This will generate a matrix of shape (batch_size, nb_kernels * kernel_height * kernel_width).
        conv_out_flatten = conv_out.flatten(2)

        pre_output = T.dot(conv_out_flatten, self.V.T) + self.bvis
        output = T.nnet.sigmoid(pre_output)

        if return_output_preactivation:
            return output, pre_output

        return output

    def get_nll_estimate(self, input, mask_o_lt_d):
        """ Returns the theano graph that computes the NLL estimate given `input`. """

        _, pre_output = self.fprop(input, mask_o_lt_d, return_output_preactivation=True)
        cross_entropies = T.nnet.softplus(-input * pre_output + (1 - input) * pre_output)
        cross_entropies_masked = cross_entropies * (1-mask_o_lt_d)
        nll = T.sum(cross_entropies_masked, axis=1)

        # TODO: check if scaling factor is correct, is d = sum(mask) for a given example?
        D = np.float32(np.prod(self.image_shape))
        d = mask_o_lt_d.sum(axis=1)
        weighted_nll = nll * (D / (D-d+1))
        return weighted_nll

    def mean_nll_estimate_loss(self, input):
        """ Returns the theano graph that computes the loss given `input`. """
        nll = self.get_nll_estimate(input, self.ordering_mask)
        return nll.mean()



from smartpy.trainers.tasks import Task, ItemGetter
from smartpy.trainers.tasks import Evaluate


class ChangeOrderingTask(Task):
    """ This task changes the ordering before each update. """
    def __init__(self, conv_nade, batch_size, ordering_seed=None):
        super(ChangeOrderingTask, self).__init__()
        self.rng = np.random.RandomState(ordering_seed)
        self.conv_nade = conv_nade
        self.batch_size = batch_size
        self.D = np.prod(conv_nade.image_shape)

    def pre_update(self, status):
        # Thanks to the broadcasting and `np.apply_along_axis`, we easily
        # generate `batch_size` orderings and compute their corresponding
        # $o_{<d}$ mask.
        d = self.rng.randint(self.D, size=(self.batch_size, 1))
        masks_o_lt_d = np.arange(self.D) < d
        map(self.rng.shuffle, masks_o_lt_d)  # Inplace shuffling along axis=1.

        self.conv_nade.ordering_mask.set_value(masks_o_lt_d)


class EvaluateDeepNadeNLLEstimate(Evaluate):
    """ This tasks compute the mean/stderr NLL estimate for a Deep NADE model.  """
    def __init__(self, conv_nade, dataset, batch_size=None, ordering_seed=42):

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
        loss = conv_nade.mean_nll_estimate_loss(input)
        no_batch = T.iscalar('no_batch')
        givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size]}
        compute_loss = theano.function([no_batch], loss, givens=givens, name="NLL Estimate")
        #theano.printing.pydotprint(compute_loss, '{0}_compute_nll_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

        def _nll_mean_and_std():
            nlls = np.zeros(len(dataset_shared.get_value()))
            for i in range(nb_batches):
                # Hack: Change ordering mask in the model before computing the NLL estimate.
                conv_nade.ordering_mask.set_value(masks_o_lt_d[i*batch_size:(i+1)*batch_size])
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

        _, pre_output = conv_nade._fprop(input, mask_o_lt_d, return_output_preactivation=True)
        cross_entropies = T.nnet.softplus(-input * pre_output + (1 - input) * pre_output)

        # Keep only $-log p(x_d | x_{<d}, o_{<d}, o_d)$
        cross_entropies_masked = cross_entropies * mask_o_d
        nll = T.sum(cross_entropies_masked, axis=1)

        no_batch = T.iscalar('no_batch')
        d = T.iscalar('d')
        givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size],
                  mask_o_d: masks_o_d[d],
                  mask_o_lt_d: masks_o_lt_d[d]}
        compute_nll = theano.function([no_batch, d], nll, givens=givens, name="NLL")
        theano.printing.pydotprint(compute_nll, '{0}_compute_nll_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

        def _nll_mean_and_std():
            nlls = np.zeros(len(dataset_shared.get_value()))
            for ordering in orderings:
                o_d = np.zeros((D, D), dtype=theano.config.floatX)
                o_d[np.arange(D), ordering] = 1
                masks_o_d.set_value(o_d)

                o_lt_d = np.cumsum(o_d, axis=0)
                o_lt_d[1:] = o_lt_d[:-1]
                o_lt_d[0, :] = 0
                masks_o_lt_d.set_value(o_lt_d)

                for i in range(nb_batches):
                    print i
                    nlls_d = []
                    for d in range(D):
                        print d
                        nlls_d.append(compute_nll(i, d))

                    nlls[i*batch_size:(i+1)*batch_size] += np.sum(np.vstack(nlls_d).T, axis=1)

            nlls /= len(orderings)  # Average across all orderings
            return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

        super(EvaluateDeepNadeNLL, self).__init__(_nll_mean_and_std)

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def std(self):
        return ItemGetter(self, attribute=1)

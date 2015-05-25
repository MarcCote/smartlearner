#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

from smartpy.misc import utils
from smartpy.misc.dataset import load_unsupervised_dataset
from smartpy.misc.dataset import Dataset, MetaDataset

from smartpy import optimizers

from smartpy import update_rules

from smartpy.trainers.status import Status
from smartpy.trainers.trainer import Trainer
from smartpy.trainers import tasks

from smartpy.models.deep_convolutional_deep_nade import DeepConvolutionalDeepNADE
from smartpy.models.convolutional_deep_nade import DeepNadeOrderingTask
from smartpy.models.convolutional_deep_nade import EvaluateDeepNadeNLL, EvaluateDeepNadeNLLEstimate
from smartpy.misc.weights_initializer import WeightsInitializer


def main():
    list_of_nb_kernels = [3, 2]
    list_of_kernel_shapes = [(5, 5), (2, 2)]
    list_of_border_modes = ['full', 'valid']
    hidden_activation = "sigmoid"
    consider_mask_as_channel = True
    batch_size = 1024
    ordering_seed = 1234
    max_epoch = 3
    nb_orderings = 1

    with utils.Timer("Loading/processing binarized MNIST"):
        dataset = load_unsupervised_dataset("binarized_mnist")

        # Extract the center patch (4x4 pixels) of each image.
        indices_to_keep = [348, 349, 350, 351, 376, 377, 378, 379, 404, 405, 406, 407, 432, 433, 434, 435]

        dataset = MetaDataset("binarized_mnist_cropped",
                              Dataset(dataset.trainset.inputs[:, indices_to_keep], name="trainset"),
                              Dataset(dataset.validset.inputs[:, indices_to_keep], name="validset"),
                              Dataset(dataset.testset.inputs[:, indices_to_keep], name="testset"))

        dataset.image_shape = (4, 4)
        dataset.nb_channels = 1

    with utils.Timer("Building model"):
        model = DeepConvolutionalDeepNADE(image_shape=dataset.image_shape,
                                          nb_channels=dataset.nb_channels,
                                          list_of_nb_kernels=list_of_nb_kernels,
                                          list_of_kernel_shapes=list_of_kernel_shapes,
                                          list_of_border_modes=list_of_border_modes,
                                          hidden_activation=hidden_activation,
                                          consider_mask_as_channel=consider_mask_as_channel
                                          )

        # Uniform initialization (default)
        model.initialize(WeightsInitializer(random_seed=42).uniform)

    with utils.Timer("Building optimizer"):
        ordering_task = DeepNadeOrderingTask(int(np.prod(model.image_shape)), batch_size, ordering_seed)
        loss = lambda input: model.mean_nll_estimate_loss(input, ordering_task.ordering_mask)

        optimizer = optimizers.SGD(loss=loss, batch_size=batch_size)
        optimizer.add_update_rule(update_rules.LearningRate(lr=0.001))

    with utils.Timer("Building trainer"):
        trainer = Trainer(model=model, datasets=[dataset.trainset.inputs_shared], optimizer=optimizer)

        # Add a task that changed the ordering mask
        trainer.add_task(ordering_task)

        # Print time for one epoch
        trainer.add_task(tasks.PrintEpochDuration())
        trainer.add_task(tasks.AverageObjective(trainer))

        nll_valid = EvaluateDeepNadeNLLEstimate(model, dataset.validset.inputs_shared, ordering_task.ordering_mask, batch_size=batch_size)
        trainer.add_task(tasks.Print(nll_valid.mean, msg="Average NLL estimate on the validset: {0}"))

        print "Will train Convoluational Deep NADE for a total of {0} epochs.".format(max_epoch)
        trainer.add_stopping_criterion(tasks.MaxEpochStopping(max_epoch))

    with utils.Timer("Training"):
        trainer.run()

    with utils.Timer("Checking the probs for all possible inputs sum to 1"):
        input = T.vector("input")
        ordering = T.ivector("ordering")
        nll_of_x_given_o = theano.function([input, ordering], model.nll_of_x_given_o(input, ordering), name="nll_of_x_given_o")
        theano.printing.pydotprint(nll_of_x_given_o, '{0}_nll_of_x_given_o_{1}'.format(model.__class__.__name__, theano.config.device), with_ids=True)

        D = np.prod(dataset.image_shape)
        from mlpython.misc.utils import cartesian
        inputs = cartesian([[0, 1]]*int(D), dtype=np.float32)

        rng = np.random.RandomState(ordering_seed)
        for i in range(nb_orderings):
            ordering = np.arange(D, dtype=np.int32)
            rng.shuffle(ordering)

            nlls = []
            for input in inputs:
                nlls.append(nll_of_x_given_o(input, ordering))

            print "Sum of p(x) for all x:", np.exp(np.logaddexp.reduce(-np.array(nlls)))

if __name__ == '__main__':
    main()

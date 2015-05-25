#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from os.path import join as pjoin
import argparse
import datetime

import pickle
import numpy as np

from smartpy.misc import utils
from smartpy.misc.dataset import UnsupervisedDataset as Dataset

from smartpy import optimizers

from smartpy import update_rules
from smartpy.optimizers import OPTIMIZERS
from smartpy.misc.weights_initializer import WEIGHTS_INITIALIZERS
from smartpy.misc.utils import ACTIVATION_FUNCTIONS

from smartpy.trainers.trainer import Trainer
from smartpy.trainers import tasks

from smartpy.models.deep_convolutional_deep_nade import DeepConvolutionalDeepNADE
from smartpy.models.convolutional_deep_nade import DeepNadeOrderingTask
from smartpy.models.convolutional_deep_nade import EvaluateDeepNadeNLL, EvaluateDeepNadeNLLEstimate


DATASETS = ['binarized_mnist']


def build_launch_experiment_argsparser(subparser):
    DESCRIPTION = "Train a Convolutional Deep NADE model on a specific dataset using Theano."

    p = subparser.add_parser("launch",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             #formatter_class=argparse.ArgumentDefaultsHelpFormatter
                             )

    # General parameters (required)
    p.add_argument('--dataset', type=str, help='dataset to use [{0}].'.format(', '.join(DATASETS)),
                   default=DATASETS[0], choices=DATASETS)

    # NADE-like's hyperparameters
    model = p.add_argument_group("Convolutional Deep NADE")
    model.add_argument('--nb_kernels', type=int, action="append", help='number of kernel/filter for the convolutional layer.', required=True, dest="list_of_nb_kernels")
    model.add_argument('--kernel_shape', type=int, nargs=2, action="append", help='height and width of kernel/filter.', required=True, dest="list_of_kernel_shapes")
    model.add_argument('--border_mode', type=str, action="append", help='border mode for convolution: valid or full.', required=True, dest="list_of_border_modes")
    model.add_argument('--ordering_seed', type=int, help='seed used to generate new ordering. Default=1234', default=1234)
    model.add_argument('--consider_mask_as_channel', action='store_true', help='consider the ordering mask as a another channel in the convolutional layer.')

    model.add_argument('--hidden_activation', type=str, help="Activation functions: {}".format(ACTIVATION_FUNCTIONS.keys()), choices=ACTIVATION_FUNCTIONS.keys(), default=ACTIVATION_FUNCTIONS.keys()[0])
    model.add_argument('--weights_initialization', type=str, help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)), default=WEIGHTS_INITIALIZERS[0], choices=WEIGHTS_INITIALIZERS)
    model.add_argument('--initialization_seed', type=int, help='seed used to generate random numbers. Default=1234', default=1234)

    # Update rules hyperparameters
    utils.create_argument_group_from_hyperparams_registry(p, update_rules.UpdateRule.registry, dest="update_rules", title="Update rules")

    # Optimizer hyperparameters
    optimizer = p.add_argument_group("Optimizer")
    optimizer.add_argument('--optimizer', type=str, help='optimizer to use for training: [{0}]'.format(OPTIMIZERS),
                           default=OPTIMIZERS[0], choices=OPTIMIZERS)
    optimizer.add_argument('--batch_size', type=int, help='size of the batch to use when training the model.', default=1)

    # Trainer parameters
    trainer = p.add_argument_group("Trainer")
    trainer.add_argument('--max_epoch', type=int, help='maximum number of epochs.')
    trainer.add_argument('--lookahead', type=int, help='use early stopping with this lookahead.')
    trainer.add_argument('--lookahead_eps', type=float, help='in early stopping, an improvement is whenever the objective improve of at least `eps`.', default=1e-3)

    # General parameters (optional)
    p.add_argument('--name', type=str, help='name of the experiment.')
    p.add_argument('--out', type=str, help='directory that will contain the experiment.', default="./")

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')


def build_resume_experiment_argsparser(subparser):
    DESCRIPTION = 'Resume a specific experiment.'

    p = subparser.add_parser("resume",
                             description=DESCRIPTION,
                             help=DESCRIPTION,
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(dest='experiment', type=str, help="experiment's directory")


def buildArgsParser():
    DESCRIPTION = "Script to launch/resume unsupervised experiment using Theano."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--keep', dest='save_frequency', action='store', type=int, help='save model every N epochs. Default=once finished', default=np.inf)
    p.add_argument('--report', dest='report_frequency', action='store', type=int, help="report results every N epochs. Default=once finished", default=np.inf)
    p.add_argument('--gsheet', type=str, metavar="SHEET_ID EMAIL PASSWORD", help="log results into a Google's Spreadsheet.")
    #p.add_argument('--view', action='store_true', help="show filters during training.")
    p.add_argument('--dry', action='store_true', help='only print folder used and quit')

    subparser = p.add_subparsers(title="subcommands", metavar="", dest="subcommand")
    build_launch_experiment_argsparser(subparser)
    build_resume_experiment_argsparser(subparser)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    print args

    if args.subcommand == "launch":
        out_dir = os.path.abspath(args.out)
        if not os.path.isdir(out_dir):
            parser.error('"{0}" must be an existing folder!'.format(out_dir))

        launch_command = " ".join(sys.argv[sys.argv.index('launch'):])

        # If experiment's name was not given generate one by hashing `launch_command`.
        if args.name is None:
            uid = utils.generate_uid_from_string(launch_command)
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            args.name = current_time + "__" + uid

        data_dir = pjoin(out_dir, args.name)
        if args.dry:
            print "Would use:\n" if os.path.isdir(data_dir) else "Would create:\n", data_dir
            return

        if os.path.isdir(data_dir):
            print "Using:\n", data_dir
        else:
            os.mkdir(data_dir)
            print "Creating:\n", data_dir

        # Save launched command to txt file
        pickle.dump(sys.argv[sys.argv.index('launch'):], open(pjoin(data_dir, "command.pkl"), 'w'))

    elif args.subcommand == "resume":
        if not os.path.isdir(args.experiment):
            parser.error("Cannot find specified experiment folder: '{}'".format(args.experiment))

        # Load command to resume
        data_dir = args.experiment
        launch_command = pickle.load(open(pjoin(args.experiment, "command.pkl")))
        command_to_resume = sys.argv[1:sys.argv.index('resume')] + launch_command
        args = parser.parse_args(command_to_resume)

        args.subcommand = "resume"

    with utils.Timer("Loading dataset"):
        dataset = Dataset(args.dataset)

        if 'mnist' in args.dataset:
            # TODO: use the new smartpy framework once merged.
            dataset.image_shape = (28, 28)
            dataset.nb_channels = 1

    with utils.Timer("Building model"):
        model = DeepConvolutionalDeepNADE(image_shape=dataset.image_shape,
                                          nb_channels=dataset.nb_channels,
                                          list_of_nb_kernels=args.list_of_nb_kernels,
                                          list_of_kernel_shapes=map(tuple, args.list_of_kernel_shapes),
                                          list_of_border_modes=args.list_of_border_modes,
                                          hidden_activation=args.hidden_activation,
                                          consider_mask_as_channel=args.consider_mask_as_channel
                                          )

        from smartpy.misc import weights_initializer
        weights_initialization_method = weights_initializer.factory(**vars(args))
        model.initialize(weights_initialization_method)

    with utils.Timer("Building optimizer"):
        ordering_task = DeepNadeOrderingTask(int(np.prod(model.image_shape)), args.batch_size, args.ordering_seed)
        loss = lambda input: model.mean_nll_estimate_loss(input, ordering_task.ordering_mask)

        optimizer = optimizers.factory(args.optimizer, loss=loss, **vars(args))
        optimizer.add_update_rule(*args.update_rules)

    with utils.Timer("Building trainer"):
        trainer = Trainer(model=model, datasets=[dataset.trainset_shared], optimizer=optimizer)

        # Add a task that changes the ordering mask
        trainer.add_task(ordering_task)

        # Print time for one epoch
        trainer.add_task(tasks.PrintEpochDuration())
        trainer.add_task(tasks.AverageObjective(trainer))

        nll_valid = EvaluateDeepNadeNLLEstimate(model, dataset.validset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)
        trainer.add_task(tasks.Print(nll_valid.mean, msg="Average NLL estimate on the validset: {0}"))

        # Add stopping criteria
        if args.max_epoch is not None:
            # Stop when max number of epochs is reached.
            print "Will train Convoluational Deep NADE for a total of {0} epochs.".format(args.max_epoch)
            trainer.add_stopping_criterion(tasks.MaxEpochStopping(args.max_epoch))

        # Do early stopping bywatching the average NLL on the validset.
        if args.lookahead is not None:
            print "Will train Convoluational Deep NADE using early stopping with a lookahead of {0} epochs.".format(args.lookahead)
            save_task = tasks.SaveTraining(trainer, savedir=data_dir)
            early_stopping = tasks.EarlyStopping(nll_valid.mean, args.lookahead, save_task, eps=args.lookahead_eps)
            trainer.add_stopping_criterion(early_stopping)
            trainer.add_task(early_stopping)

        # Add a task to save the whole training process
        if args.save_frequency < np.inf:
            save_task = tasks.SaveTraining(trainer, savedir=data_dir, each_epoch=args.save_frequency)
            trainer.add_task(save_task)

        if args.subcommand == "resume":
            print "Loading existing trainer..."
            trainer.load(data_dir)

    with utils.Timer("Training"):
        trainer.run()
        trainer.status.save(savedir=data_dir)

        if not args.lookahead:
            trainer.save(savedir=data_dir)

    with utils.Timer("Reporting"):
        # Evaluate model on train, valid and test sets
        nll_train = EvaluateDeepNadeNLLEstimate(model, dataset.trainset_shared, batch_size=args.batch_size)
        nll_valid = EvaluateDeepNadeNLLEstimate(model, dataset.validset_shared, batch_size=args.batch_size)
        nll_test = EvaluateDeepNadeNLLEstimate(model, dataset.testset_shared, batch_size=args.batch_size)

        from collections import OrderedDict
        log_entry = OrderedDict()
        log_entry["Nb. kernels"] = model.hyperparams["list_of_nb_kernels"]
        log_entry["Kernel Shapes"] = model.hyperparams["lift_of_kernel_shapes"]
        log_entry["Border Modes"] = model.hyperparams["lift_of_border_modes"]
        log_entry["Mask as channel"] = model.hyperparams["consider_mask_as_channel"]
        log_entry["Activation Function"] = model.hyperparams["hidden_activation"]
        log_entry["Initialization Seed"] = args.initialization_seed
        log_entry["Best Epoch"] = trainer.status.extra["best_epoch"] if args.lookahead else trainer.status.current_epoch
        log_entry["Max Epoch"] = trainer.stopping_criteria[0].nb_epochs_max if args.max_epoch else ''

        if args.max_epoch:
            log_entry["Look Ahead"] = trainer.stopping_criteria[1].lookahead if args.lookahead else ''
            log_entry["Look Ahead eps"] = trainer.stopping_criteria[1].eps if args.lookahead else ''
        else:
            log_entry["Look Ahead"] = trainer.stopping_criteria[0].lookahead if args.lookahead else ''
            log_entry["Look Ahead eps"] = trainer.stopping_criteria[0].eps if args.lookahead else ''

        log_entry["Batch Size"] = trainer.optimizer.batch_size
        log_entry["Update Rule"] = trainer.optimizer.update_rules[0].__class__.__name__
        log_entry["Learning Rate"] = trainer.optimizer.update_rules[0].lr
        log_entry["Weights Initialization"] = args.weights_initialization
        log_entry["Training NLL - Estimate"] = nll_train.mean
        log_entry["Training NLL std"] = nll_train.std
        log_entry["Validation NLL - Estimate"] = nll_valid.mean
        log_entry["Validation NLL std"] = nll_valid.std
        log_entry["Testing NLL - Estimate"] = nll_test.mean
        log_entry["Testing NLL std"] = nll_test.std
        log_entry["Training Time"] = trainer.status.training_time
        log_entry["Experiment"] = os.path.abspath(data_dir)

        formatting = {}
        formatting["Training NLL - Estimate"] = "{:.6f}"
        formatting["Training NLL std"] = "{:.6f}"
        formatting["Validation NLL - Estimate"] = "{:.6f}"
        formatting["Validation NLL std"] = "{:.6f}"
        formatting["Testing NLL - Estimate"] = "{:.6f}"
        formatting["Testing NLL std"] = "{:.6f}"
        formatting["Training Time"] = "{:.4f}"

        from smartpy.trainers import Status
        status = Status()
        logging_task = tasks.LogResultCSV("results_{}_{}.csv".format("ConvDeepNADE", dataset.name), log_entry, formatting)
        logging_task.execute(status)

        if args.gsheet is not None:
            gsheet_id, gsheet_email, gsheet_password = args.gsheet.split()
            logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, "ConvDeepNADE", log_entry, formatting)
            logging_task.execute(status)

if __name__ == '__main__':
    main()

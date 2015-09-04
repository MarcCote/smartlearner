#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack when having multiple clone of smartpy repo like a do.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

from os.path import join as pjoin

import argparse
import datetime
import theano.tensor as T

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

from smartpy.models.convolutional_deepnade import DeepConvNADEBuilder

from smartpy.models.convolutional_deepnade import DeepNadeOrderingTask, DeepNadeTrivialOrderingsTask
from smartpy.models.convolutional_deepnade import EvaluateDeepNadeNLL, EvaluateDeepNadeNLLEstimate
from smartpy.models.convolutional_deepnade import EvaluateDeepNadeNLLEstimateOnTrivial


DATASETS = ['binarized_mnist']


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


def build_launch_experiment_argsparser(subparser):
    DESCRIPTION = "Train a Deep Convolutional NADE model on a specific dataset using Theano."

    p = subparser.add_parser("launch", description=DESCRIPTION, help=DESCRIPTION)

    # General parameters (required)
    p.add_argument('--dataset', type=str, help='dataset to use [{0}].'.format(', '.join(DATASETS)), default=DATASETS[0], choices=DATASETS)

    # NADE-like's hyperparameters
    model = p.add_argument_group("Convolutional Deep NADE")
    model.add_argument('--convnet_blueprint', type=str, help='blueprint of the convolutional layers e.g. "64@3x3(valid)->32@7x7(full)".')
    model.add_argument('--fullnet_blueprint', type=str, help='blueprint of the fully connected layers e.g. "500->784".')
    model.add_argument('--blueprint_seed', type=int, help='seed used to generate random blueprints.')
    model.add_argument('--ordering_seed', type=int, help='seed used to generate new ordering. Default=1234', default=1234)
    model.add_argument('--consider_mask_as_channel', action='store_true', help='consider the ordering mask as a another channel in the convolutional layer.')
    #model.add_argument('--finetune_on_trivial_orderings', action='store_true', help='finetune model using the 8 trivial orderings.')

    model.add_argument('--hidden_activation', type=str, help="Activation functions: {}".format(ACTIVATION_FUNCTIONS.keys()), choices=ACTIVATION_FUNCTIONS.keys(), default=ACTIVATION_FUNCTIONS.keys()[0])
    model.add_argument('--weights_initialization', type=str, help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)), default=WEIGHTS_INITIALIZERS[0], choices=WEIGHTS_INITIALIZERS)
    model.add_argument('--initialization_seed', type=int, help='seed used to generate random numbers. Default=1234', default=1234)

    # Update rules hyperparameters
    utils.create_argument_group_from_hyperparams_registry(p, update_rules.UpdateRule.registry, dest="update_rules", title="Update rules")

    # Optimizer hyperparameters
    optimizer = p.add_argument_group("Optimizer")
    optimizer.add_argument('--optimizer', type=str, help='optimizer to use for training: [{0}]'.format(OPTIMIZERS), default=OPTIMIZERS[0], choices=OPTIMIZERS)
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
    p = subparser.add_parser("resume", description=DESCRIPTION, help=DESCRIPTION)
    p.add_argument(dest='experiment', type=str, help="experiment's directory")


def build_finetune_experiment_argsparser(subparser):
    DESCRIPTION = 'Finetune a model using the 8 trivial orderings.'
    p = subparser.add_parser("finetune", description=DESCRIPTION, help=DESCRIPTION)
    p.add_argument(dest='experiment', type=str, help="experiment's directory")


def buildArgsParser():
    DESCRIPTION = "Script to launch/resume unsupervised experiment using Theano."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--exact_inference', action='store_true', help='Compute the exact NLL on the validset and testset (slower)')
    p.add_argument('--ensemble', type=int, help='Size of the ensemble. Default=1', default=1)
    p.add_argument('--no-train', action='store_true', help='Skip training part of the script and perform reporting')
    p.add_argument('--keep', dest='save_frequency', action='store', type=int, help='save model every N epochs. Default=once finished', default=np.inf)
    p.add_argument('--report', dest='report_frequency', action='store', type=int, help="report results every N epochs. Default=once finished", default=np.inf)
    p.add_argument('--gsheet', type=str, metavar="SHEET_ID EMAIL PASSWORD", help="log results into a Google's Spreadsheet.")
    #p.add_argument('--view', action='store_true', help="show filters during training.")
    p.add_argument('--dry', action='store_true', help='only print folder used and quit')

    subparser = p.add_subparsers(title="subcommands", metavar="", dest="subcommand")
    build_launch_experiment_argsparser(subparser)
    build_resume_experiment_argsparser(subparser)
    build_finetune_experiment_argsparser(subparser)
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

    elif args.subcommand == "finetune":
        if not os.path.isdir(args.experiment):
            parser.error("Cannot find specified experiment folder: '{}'".format(args.experiment))

        # Load command to resume
        data_dir = args.experiment
        launch_command = pickle.load(open(pjoin(args.experiment, "command.pkl")))
        command_to_resume = sys.argv[1:sys.argv.index('finetune')] + launch_command
        args = parser.parse_args(command_to_resume)

        args.subcommand = "finetune"

    with utils.Timer("Loading dataset"):
        dataset = Dataset(args.dataset)

        if 'mnist' in args.dataset:
            # TODO: use the new smartpy framework once merged.
            dataset.image_shape = (28, 28)
            dataset.nb_channels = 1

    with utils.Timer("Building model"):
        builder = DeepConvNADEBuilder(image_shape=dataset.image_shape,
                                      nb_channels=dataset.nb_channels,
                                      ordering_seed=args.ordering_seed,
                                      consider_mask_as_channel=args.consider_mask_as_channel,
                                      hidden_activation=args.hidden_activation)

        if args.blueprint_seed is not None:
            convnet_blueprint, fullnet_blueprint = generate_blueprints(args.blueprint_seed, dataset.image_shape[0])
            print convnet_blueprint
            print fullnet_blueprint
            builder.build_convnet_from_blueprint(convnet_blueprint)
            builder.build_fullnet_from_blueprint(fullnet_blueprint)
        else:
            if args.convnet_blueprint is not None:
                builder.build_convnet_from_blueprint(args.convnet_blueprint)

            if args.fullnet_blueprint is not None:
                builder.build_fullnet_from_blueprint(args.fullnet_blueprint)

        model = builder.build()

        from smartpy.misc import weights_initializer
        weights_initialization_method = weights_initializer.factory(**vars(args))
        model.initialize(weights_initialization_method)

    # Print structure of the model for debugging
    print model

    with utils.Timer("Building optimizer"):
        if args.subcommand == "finetune":
            ordering_task = DeepNadeTrivialOrderingsTask(model.image_shape, args.batch_size, model.ordering_seed)
            loss = lambda input: T.mean(-model.lnp_x_o_d_given_x_o_lt_d(input, ordering_task.mask_o_d, ordering_task.mask_o_lt_d))
        else:
            ordering_task = DeepNadeOrderingTask(int(np.prod(model.image_shape)), args.batch_size, model.ordering_seed)
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

        if args.subcommand == "finetune":
            nll_valid = EvaluateDeepNadeNLLEstimateOnTrivial(model, dataset.validset_shared, batch_size=args.batch_size)
        else:
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
            early_stopping = tasks.EarlyStopping(nll_valid.mean, args.lookahead, save_task, eps=args.lookahead_eps, skip_epoch0=True)
            trainer.add_stopping_criterion(early_stopping)
            trainer.add_task(early_stopping)

        # Add a task to save the whole training process
        if args.save_frequency < np.inf:
            save_task = tasks.SaveTraining(trainer, savedir=data_dir, each_epoch=args.save_frequency)
            trainer.add_task(save_task)

        if args.subcommand == "resume" or args.subcommand == "finetune":
            print "Loading existing trainer..."
            trainer.load(data_dir)

        if args.subcommand == "finetune":
            trainer.status.extra['best_epoch'] = trainer.status.current_epoch

        trainer._build_learn()

    if not args.no_train:
        with utils.Timer("Training"):
            trainer.run()
            trainer.status.save(savedir=data_dir)

            if not args.lookahead:
                trainer.save(savedir=data_dir)

    with utils.Timer("Reporting"):
        # Evaluate model on train, valid and test sets
        nll_train = EvaluateDeepNadeNLLEstimate(model, dataset.trainset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)
        nll_valid = EvaluateDeepNadeNLLEstimate(model, dataset.validset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)
        nll_test = EvaluateDeepNadeNLLEstimate(model, dataset.testset_shared, ordering_task.ordering_mask, batch_size=args.batch_size)

        if args.exact_inference:
            nll_valid = EvaluateDeepNadeNLL(model, dataset.validset_shared, batch_size=args.batch_size, nb_orderings=args.ensemble)
            nll_test = EvaluateDeepNadeNLL(model, dataset.testset_shared, batch_size=args.batch_size, nb_orderings=args.ensemble)

        print "Training NLL - Estimate:", nll_train.mean.view(trainer.status)
        print "Training NLL std:", nll_train.std.view(trainer.status)
        print "Validation NLL - Estimate:", nll_valid.mean.view(trainer.status)
        print "Validation NLL std:", nll_valid.std.view(trainer.status)
        print "Testing NLL - Estimate:", nll_test.mean.view(trainer.status)
        print "Testing NLL std:", nll_test.std.view(trainer.status)

        from collections import OrderedDict
        log_entry = OrderedDict()
        log_entry["Convnet Blueprint"] = args.convnet_blueprint
        log_entry["Fullnet Blueprint"] = args.fullnet_blueprint
        log_entry["Mask as channel"] = model.hyperparams["consider_mask_as_channel"]
        log_entry["Activation Function"] = args.hidden_activation
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
        update_rule = trainer.optimizer.update_rules[0]
        log_entry["Learning Rate"] = "; ".join(["{0}={1}".format(name, getattr(update_rule, name)) for name in update_rule.__hyperparams__.keys()])

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

        logging_task = tasks.LogResultCSV("results_{}_{}.csv".format("ConvDeepNADE", dataset.name), log_entry, formatting)
        logging_task.execute(trainer.status)

        if args.gsheet is not None:
            gsheet_id, gsheet_email, gsheet_password = args.gsheet.split()
            logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, "ConvDeepNADE", log_entry, formatting)
            logging_task.execute(trainer.status)

if __name__ == '__main__':
    main()

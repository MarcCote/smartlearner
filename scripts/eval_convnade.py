#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
# Hack when having multiple clone of smartpy repo like a do.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import re
import pickle
from collections import OrderedDict
import argparse
import numpy as np
from os.path import join as pjoin


from smartpy.misc import utils
from smartpy.trainers import tasks, Status
from smartpy.models.convolutional_deepnade import DeepConvNADE as ConvNADE
from smartpy.misc.dataset import UnsupervisedDataset as Dataset
from smartpy.misc.utils import load_dict_from_json_file

from smartpy.models.convolutional_deepnade import generate_blueprints, DeepConvNADEBuilder
from smartpy.models.convolutional_deepnade import EvaluateDeepNadeNLLParallel


DATASETS = ['binarized_mnist']


def build_eval_argsparser(subparser):
    DESCRIPTION = "Evaluate a Convolutional NADE model."

    p = subparser.add_parser("eval", description=DESCRIPTION, help=DESCRIPTION)

    p.add_argument('dataset', type=str, help='evaluate on a specific dataset [{0}].'.format(', '.join(DATASETS)), choices=DATASETS)
    p.add_argument('--batch_size', type=int, help='size of the batch to use when evaluating the model.', default=64)
    p.add_argument('--subset', type=str, choices=['valid', 'test'], help='evaluate only a specific subset (either "testset" or "validset") {0}]. Default: evaluate both subsets.')
    p.add_argument('--part', metavar="<no_part>/<nb_parts>", type=str, help='evaluate only a specific part of the dataset (e.g. 5/10). Default: evaluate the whole dataset.')
    p.add_argument('--ordering', type=int, help='evaluate only a specific input ordering. Default: evaluate all orderings.')
    p.add_argument('--nb_orderings', type=int, help='evaluate that many input orderings. Default: 8.', default=8)
    p.add_argument('--orderings_seed', type=int, help='evaluate random ordering(s) generate using this seed. Default: will use the 8 trivial orderings.')

    # Optional parameters
    p.add_argument('-f', '--force',  action='store_true', help='overwrite evaluation results')


def build_report_argsparser(subparser):
    DESCRIPTION = 'Report results of the evaluation.'
    p = subparser.add_parser("report", description=DESCRIPTION, help=DESCRIPTION)
    p.add_argument('--ignore',  action='store_true', help='ignore missing orderings')
    p.add_argument('-v', '--verbose', action='store_true', help='print average NLL per ordering')
    # TODO: add stuff to upload results in a CSV file.
    # TODO: add stuff to upload results in a GSheet.
    #p.add_argument('--gsheet', type=str, metavar="SHEET_ID EMAIL PASSWORD", help="log results into a Google's Spreadsheet.")


def buildArgsParser():
    DESCRIPTION = "Script to evaluate a ConvNADE experiment and report the results."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('experiment', type=str, help="experiment's directory")

    subparser = p.add_subparsers(title="subcommands", metavar="", dest="subcommand")
    build_eval_argsparser(subparser)
    build_report_argsparser(subparser)
    return p


def evaluate(args):
    evaluation_folder = pjoin(args.experiment, "evaluation")
    if not os.path.isdir(evaluation_folder):
        os.mkdir(evaluation_folder)

    with utils.Timer("Loading model"):
        hyperparams = load_dict_from_json_file(pjoin(args.experiment, "hyperparams.json"))

        image_shape = tuple(hyperparams["image_shape"])
        hidden_activation = [v for k, v in hyperparams.items() if "activation" in k][0]
        builder = DeepConvNADEBuilder(image_shape=image_shape,
                                      nb_channels=hyperparams["nb_channels"],
                                      ordering_seed=hyperparams["ordering_seed"],
                                      consider_mask_as_channel=hyperparams["consider_mask_as_channel"],
                                      hidden_activation=hidden_activation)

        # Read infos from "command.pkl"

        command = pickle.load(open(pjoin(args.experiment, "command.pkl")))

        blueprint_seed = None
        if "--blueprint_seed" in command:
            blueprint_seed = int(command[command.index("--blueprint_seed") + 1])

        convnet_blueprint = None
        if "--convnet_blueprint" in command:
            convnet_blueprint = int(command[command.index("--convnet_blueprint") + 1])

        fullnet_blueprint = None
        if "--fullnet_blueprint" in command:
            fullnet_blueprint = int(command[command.index("--fullnet_blueprint") + 1])

        if blueprint_seed is not None:
            convnet_blueprint, fullnet_blueprint = generate_blueprints(blueprint_seed, image_shape[0])
            print convnet_blueprint
            print fullnet_blueprint
            builder.build_convnet_from_blueprint(convnet_blueprint)
            builder.build_fullnet_from_blueprint(fullnet_blueprint)
        else:
            if convnet_blueprint is not None:
                builder.build_convnet_from_blueprint(convnet_blueprint)

            if fullnet_blueprint is not None:
                builder.build_fullnet_from_blueprint(fullnet_blueprint)

        convnade = builder.build()
        convnade.load(args.experiment)

    with utils.Timer("Loading dataset"):
        dataset = Dataset(args.dataset)

    no_part, nb_parts = 1, 1
    if args.part is not None:
        no_part, nb_parts = map(int, args.part.split("/"))

    def _compute_nll(subset):
        #eval model_name/ eval --dataset testset --part [1:11]/10 --ordering [0:8]
        nll_evaluation = EvaluateDeepNadeNLLParallel(convnade, subset,
                                                     batch_size=args.batch_size, no_part=no_part, nb_parts=nb_parts,
                                                     no_ordering=args.ordering, nb_orderings=args.nb_orderings, orderings_seed=args.orderings_seed)
        nlls = nll_evaluation.view()

        # Save [partial] evaluation results.
        name = "{subset}_part{no_part}of{nb_parts}_ordering{no_ordering}of{nb_orderings}"
        name = name.format(subset=subset,
                           no_part=no_part,
                           nb_parts=nb_parts,
                           no_ordering=args.ordering,
                           nb_orderings=args.nb_orderings)
        filename = pjoin(evaluation_folder, name + ".npy")
        np.save(filename, nlls)

    if args.subset == "valid" or args.subset is None:
        _compute_nll(dataset.validset_shared)

    if args.subset == "test" or args.subset is None:
        _compute_nll(dataset.testset_shared)


def report(args):
    evaluation_folder = pjoin(args.experiment, "evaluation")
    evaluation_files = os.listdir(evaluation_folder)
    _, nb_parts, _, nb_orderings = map(int, re.findall("part([0-9]+)of([0-9]+)_ordering([0-9]+)of([0-9]+)", evaluation_files[0])[0])

    template = "{subset}_part{no_part}of{nb_parts}_ordering{no_ordering}of{nb_orderings}.npy"

    # Check if we miss some evaluation results
    orderings = {"validset": [], "testset": []}
    nb_results_missing = 0
    for subset in ["validset", "testset"]:
        for no_ordering in range(nb_orderings):
            is_missing_part = False
            for no_part in range(1, nb_parts+1):
                name = template.format(subset=subset,
                                       no_part=no_part, nb_parts=nb_parts,
                                       no_ordering=no_ordering, nb_orderings=nb_orderings)

                if name not in evaluation_files:
                    is_missing_part = True
                    nb_results_missing += 1
                    print "Missing: ", name

            if not is_missing_part:
                orderings[subset].append(no_ordering)

    if nb_results_missing > 0:
        print "Missing {} result(s). Terminating...".format(nb_results_missing)
        if not args.ignore:
            return

    # Merge results
    def _nll_mean_stderr(subset):
        nlls = []
        # Examples have been split in multiple parts.
        for no_part in range(1, nb_parts+1):
            nlls_part = []
            for no_ordering in orderings[subset]:
                name = template.format(subset=subset,
                                       no_part=no_part, nb_parts=nb_parts,
                                       no_ordering=no_ordering, nb_orderings=nb_orderings)

                # Load the NLLs for a given part and a given ordering.
                nlls_part.append(np.load(pjoin(evaluation_folder, name)))

            # Average the probabilities across the orderings independently for each example.
            nlls_part = -np.logaddexp.reduce(-np.array(nlls_part), axis=0)
            nlls_part += np.log(len(orderings[subset]))  # Average across all orderings
            nlls.append(nlls_part)

        # Concatenate every part together.
        nlls = np.hstack(nlls)
        return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

    def _nll_per_ordering(subset, no_ordering):
        nlls = []
        # Examples have been split in multiple parts.
        for no_part in range(1, nb_parts+1):
            name = template.format(subset=subset,
                                   no_part=no_part, nb_parts=nb_parts,
                                   no_ordering=no_ordering, nb_orderings=nb_orderings)

            # Load the NLLs for a given part and a given ordering.
            nlls_part = np.load(pjoin(evaluation_folder, name))
            nlls.append(nlls_part)

        # Concatenate every part together.
        nlls = np.hstack(nlls)
        return nlls.mean()

    # Compute NLL mean and NLL stderror on validset and testset.
    validset_mean, validset_stderr = _nll_mean_stderr("validset")
    print "\nUsing {} orderings.".format(len(orderings["validset"]))
    print "Validation NLL:", validset_mean
    print "Validation NLL std:", validset_stderr
    if args.verbose:
        nlls_mean_per_ordering = map(lambda o: _nll_per_ordering("validset", o), orderings["validset"])
        print "NLLs mean per ordering (validset):"
        print "\n".join(["#{}: {}".format(o, nll_mean) for nll_mean, o in zip(nlls_mean_per_ordering, orderings["validset"])])

    testset_mean, testset_stderr = _nll_mean_stderr("testset")
    print "\nUsing {} orderings.".format(len(orderings["testset"]))
    print "Testing NLL:", testset_mean
    print "Testing NLL std:", testset_stderr
    if args.verbose:
        nlls_mean_per_ordering = map(lambda o: _nll_per_ordering("testset", o), orderings["testset"])
        print "NLLs mean per ordering (testset):"
        print "\n".join(["#{}: {}".format(o, nll_mean) for nll_mean, o in zip(nlls_mean_per_ordering, orderings["testset"])])


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if args.subcommand == "eval":
        with utils.Timer("Evaluating"):
            evaluate(args)

    elif args.subcommand == "report":
        with utils.Timer("Reporting"):
            report(args)

    # # with utils.Timer("Loading experiment"):
    # #     trainer = trainers.load(args.experiment)
    # #     #nade = trainer.model

    # ### Temporary patch ###
    # import pickle
    # from os.path import join as pjoin
    # from smartpy.misc.utils import load_dict_from_json_file
    # status = load_dict_from_json_file(pjoin(args.experiment, "status.json"))
    # best_epoch = status["extra"]["best_epoch"]
    # command = pickle.load(open(pjoin(args.experiment, "command.pkl")))
    # lr = float(command[command.index("--AdamV1") + 1])
    # training_time = status["training_time"]
    # ######

    # nll_train = tasks.EvaluateNLL(nade.get_nll, dataset.trainset, batch_size=100)
    # nll_valid = tasks.EvaluateNLL(nade.get_nll, dataset.validset, batch_size=100)
    # nll_test = tasks.EvaluateNLL(nade.get_nll, dataset.testset, batch_size=100)

    # log_entry = OrderedDict()
    # log_entry["Learning Rate"] = lr  # trainer.optimizer.update_rules[0].lr
    # log_entry["Learning Rate NADE"] = lr
    # log_entry["Random Seed"] = 1234
    # log_entry["Hidden Size"] = nade.hyperparams["hidden_size"]
    # log_entry["Activation Function"] = nade.hyperparams["hidden_activation"]
    # log_entry["Gamma"] = 0.
    # log_entry["Tied Weights"] = nade.hyperparams["tied_weights"]
    # log_entry["Best Epoch"] = best_epoch  # trainer.status.extra["best_epoch"]
    # log_entry["Max Epoch"] = ''
    # log_entry["Look Ahead"] = 10  # trainer.stopping_criteria[0].lookahead
    # log_entry["Batch Size"] = 100  # trainer.optimizer.batch_size
    # log_entry["Update Rule"] = "AdamV1"  # trainer.optimizer.update_rules[0].__class__.__name__
    # log_entry["Weights Initialization"] = "Uniform"
    # log_entry["Training NLL"] = nll_train.mean
    # log_entry["Training NLL std"] = nll_train.std
    # log_entry["Validation NLL"] = nll_valid.mean
    # log_entry["Validation NLL std"] = nll_valid.std
    # log_entry["Testing NLL"] = nll_test.mean
    # log_entry["Testing NLL std"] = nll_test.std
    # log_entry["Training Time"] = training_time  # trainer.status.training_time
    # log_entry["Experiment"] = os.path.abspath(args.experiment)
    # log_entry["NADE"] = "./"

    # formatting = {}
    # formatting["Training NLL"] = "{:.6f}"
    # formatting["Training NLL std"] = "{:.6f}"
    # formatting["Validation NLL"] = "{:.6f}"
    # formatting["Validation NLL std"] = "{:.6f}"
    # formatting["Testing NLL"] = "{:.6f}"
    # formatting["Testing NLL std"] = "{:.6f}"
    # formatting["Training Time"] = "{:.4f}"

    # status = Status()
    # with utils.Timer("Evaluating"):
    #     logging_task = tasks.LogResultCSV("results_{}_{}.csv".format("NADE", dataset.name), log_entry, formatting)
    #     logging_task.execute(status)

    #     if args.gsheet is not None:
    #         gsheet_id, gsheet_email, gsheet_password = args.gsheet.split()
    #         logging_task = tasks.LogResultGSheet(gsheet_id, gsheet_email, gsheet_password, "NADE", log_entry, formatting)
    #         logging_task.execute(status)

if __name__ == '__main__':
    main()

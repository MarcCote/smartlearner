#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin
import argparse

import pickle
import numpy as np

from smartpy.misc import utils
from smartpy.models.convolutional_deepnade import generate_blueprints, DeepConvNADEBuilder

from smartpy.misc.utils import load_dict_from_json_file


def buildArgsParser():
    DESCRIPTION = "Generate samples from a Conv Deep NADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('experiment', type=str, help='folder where to find a trained ConvDeepNADE model')
    p.add_argument('count', type=int, help='number of samples to generate.')
    p.add_argument('--out', type=str, help='name of the samples file')

    # General parameters (optional)
    p.add_argument('--seed', type=int, help='seed used to generate random numbers.')
    p.add_argument('--view', action='store_true', help="show samples.")

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')

    return p


def load_model(args):
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
            builder.build_convnet_from_blueprint(convnet_blueprint)
            builder.build_fullnet_from_blueprint(fullnet_blueprint)
        else:
            if convnet_blueprint is not None:
                builder.build_convnet_from_blueprint(convnet_blueprint)

            if fullnet_blueprint is not None:
                builder.build_fullnet_from_blueprint(fullnet_blueprint)

        print convnet_blueprint
        print fullnet_blueprint
        convnade = builder.build()
        convnade.load(args.experiment)

    return convnade


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    model = load_model(args)

    with utils.Timer("Generating {} samples from Conv Deep NADE".format(args.count)):
        sample = model.build_sampling_function(seed=args.seed)
        samples = sample(args.count)

    if args.out is not None:
        outfile = pjoin(args.experiment, args.out)
        with utils.Timer("Saving {0} samples to '{1}'".format(args.count, outfile)):
            np.save(outfile, samples)

    if args.view:
        import pylab as plt
        from mlpython.misc.utils import show_samples
        show_samples(samples, title="Uniform samples")
        plt.show()

if __name__ == '__main__':
    main()

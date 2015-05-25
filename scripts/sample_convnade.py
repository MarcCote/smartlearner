#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin
import argparse

import numpy as np

from smartpy.misc import utils
from smartpy.models.deep_convolutional_deep_nade import DeepConvolutionalDeepNADE


def buildArgsParser():
    DESCRIPTION = "Generate samples from a Conv Deep NADE model."
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('model', type=str, help='folder where to find a trained ConvDeepNADE model')
    p.add_argument('count', type=int, help='number of samples to generate.')
    p.add_argument('--out', type=str, help='name of the samples file')

    # General parameters (optional)
    p.add_argument('--seed', type=int, help='seed used to generate random numbers.')
    p.add_argument('--view', action='store_true', help="show samples.")

    p.add_argument('-v', '--verbose', action='store_true', help='produce verbose output')
    p.add_argument('-f', '--force',  action='store_true', help='permit overwriting')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    with utils.Timer("Loading model"):
        model = DeepConvolutionalDeepNADE.create(args.model)

    with utils.Timer("Generating {} samples from Conv Deep NADE".format(args.count)):
        sample = model.build_sampling_function(seed=args.seed)
        samples = sample(args.count)

    if args.out is not None:
        outfile = pjoin(args.model, args.out)
        with utils.Timer("Saving {0} samples to '{1}'".format(args.count, outfile)):
            np.save(outfile, samples)

    if args.view:
        import pylab as plt
        from mlpython.misc.utils import show_samples
        show_samples(samples, title="Uniform samples")
        plt.show()

if __name__ == '__main__':
    main()

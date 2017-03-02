#! /usr/bin/env python

import itertools
import argparse
import random
import os

import noscope
import keras
import numpy as np
from noscope import Learner, HyperBand

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', type=int, help='Number of frames')
    # Regression or not
    parser.add_argument('--regression', dest='regression', action='store_true')
    parser.add_argument('--no-regression', dest='regression', action='store_false')
    parser.set_defaults(regression=False)
    # Binary or not (for classification)
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--no-binary', dest='binary', action='store_false')
    parser.set_defaults(binary=True)
    args = parser.parse_args()

    def check_args(args):
        if args.regression:
            if args.binary:
                print 'WARNING: Setting args.binary to False'
                args.binary = False
        else:
            # Check here?
            pass
        assert args.objects is not None
    check_args(args)

    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    print 'Preparing data....'
    data, nb_classes = noscope.DataUtils.get_data(
            args.csv_in, args.video_in,
            binary=args.binary,
            num_frames=args.num_frames,
            OBJECTS=objects,
            regression=args.regression,
            resol=(50, 50))
    X_train, Y_train, X_test, Y_test = data

    def get_random_learner():
        lr_mults = [0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 10.0]
        if random.randint(0, 1) == 0:
            name = 'cifar10'
            model_gen = noscope.Models.generate_cifar10
            param0s = [32, 64, 128, 256]
            param1s = [0, 1, 2]
        else:
            name = 'mnist'
            model_gen = noscope.Models.generate_mnist
            param0s = [32, 64, 128, 256]
            param1s = [16, 32, 64]

        param0 = np.random.choice(param0s)
        param1 = np.random.choice(param1s)
        lr_mult = np.random.choice(lr_mults)
        model = model_gen(X_train.shape[1:], nb_classes,
                          param0,
                          param1,
                          regression=args.regression,
                          lr_mult=lr_mult)
        fname = os.path.join(
            args.output_dir,
            '%s_%s_%2.1f_%d_%d.hdf5' % (args.base_name, name, lr_mult, param0, param1))

        return Learner(model, data, fname,
                       regression=args.regression,
                       nb_examples=10000)


    hyperband = HyperBand(100)
    learners = hyperband.run(get_random_learner)
    print len(learners)
    for learner in learners:
        print learner.fname
        print learner.test_eval()
        learner.save_model()


if __name__ == '__main__':
    main()

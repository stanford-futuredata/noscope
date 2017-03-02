#! /usr/bin/env python

import itertools
import argparse
import noscope
import numpy as np
import glob
from skimage.io import imread
from keras.utils import np_utils

def metrics_names(metrics):
    return sorted(metrics.keys())

def metrics_to_list(metrics):
    return map(lambda key: metrics[key], metrics_names(metrics))

def read_images_from_dir(dir_name):
    for image_path in glob.iglob(dir_name + '/*.jpg'):
        img = imread(image_path)


def get_data(object, image_dir):
    X_train_positive = read_images_from_dir('%s/%s_images' % (image_dir, object))
    X_train_negative = read_images_from_dir('%s/%s_images' % (image_dir, 'no_' + object))

    Y_train_positive = np.ones(len(X_train_positive))
    Y_train_negative = np.zeros(len(X_train_negative))

    X_train = np.vstack(X_train_positive, X_train_negative)
    Y_train = np.vstack(Y_train_positive, Y_train_negative)

    data = (X_train, Y_train)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--image_dir', required=True, help='Directory with positive and negative images')
    args = parser.parse_args()

    print 'Preparing data....'
    data = get_data(args.object, args.image_dir)
    nb_classes = 2
    X_train, Y_train, X_test, Y_test = data

    print 'Trying CIFAR10....'
    # CIFAR10 based architectures
    noscope.Models.try_params(
            noscope.Models.generate_cifar10,
            list(itertools.product(
                    *[[X_train.shape[1:]], [nb_classes],
                      [32, 64, 128, 256], [0, 1, 2]])),
            data,
            args.csv_out_base + '_cifar10.csv',
            regression=args.regression,
            nb_epoch=4)
    print 'Trying MNIST....'
    noscope.Models.try_params(
            noscope.Models.generate_mnist,
            list(itertools.product(
                    *[[X_train.shape[1:]], [nb_classes],
                      [32, 64, 128, 256], [16, 32, 64]])),
            data,
            args.csv_out_base + '_mnist.csv',
            regression=args.regression,
            nb_epoch=4)

if __name__ == '__main__':
    main()

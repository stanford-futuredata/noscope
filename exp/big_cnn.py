#! /usr/bin/env python

import itertools
import argparse
import noscope

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', type=int, help='Number of frames')
    args = parser.parse_args()

    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    print 'Preparing data....'
    data, nb_classes = noscope.DataUtils.get_data(
            args.csv_in, args.video_in,
            binary=True,
            num_frames=args.num_frames,
            OBJECTS=objects,
            regression=False,
            resol=(50, 50))
    X_train, Y_train, X_test, Y_test = data

    print 'Trying VGG16....'
    # CIFAR10 based architectures
    noscope.Models.try_params(
            noscope.Models.generate_vgg16,
            list(itertools.product(
                    *[[X_train.shape[1:]], [nb_classes],
                      [False, True]])),
            data,
            args.output_dir,
            args.base_name,
            'vgg16',
            objects[0],
            regression=False,
            nb_epoch=20)

if __name__ == '__main__':
    main()

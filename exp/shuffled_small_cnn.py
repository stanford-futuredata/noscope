import itertools
import argparse
import numpy as np
import pandas as pd
import noscope
from noscope import np_utils

def to_test_train(avg_fname, all_frames, all_counts, train_ratio=0.6):
    print all_frames.shape
    assert len(all_frames) == len(all_counts), 'Frame length should equal counts length'

    nb_classes = all_counts.max() + 1
    X = all_frames

    mean = np.mean(X, axis=0)
    np.save(avg_fname, mean)

    N = 150000
    # N = 500000

    '''pos_inds = np.random.permutation(np.where(all_counts.ravel() == 0))
    pos_inds = pos_inds[0, 0 : N/2]
    neg_inds = np.random.permutation(np.where(all_counts.ravel() == 1))
    neg_inds = neg_inds[0, 0 : N/2]
    print pos_inds.shape
    print neg_inds.shape
    p = np.concatenate([pos_inds, neg_inds])
    np.random.shuffle(p)'''

    p = np.random.permutation(len(all_counts))
    p = p[0:N]

    Y = np_utils.to_categorical(all_counts, nb_classes)
    X, Y = X[p], Y[p]
    X -= mean

    def split(arr):
        # 250 -> 100, 50, 100
        ind = int(len(arr) * train_ratio)
        if ind > 50000:
            ind = len(arr) - 50000
        return arr[:ind], arr[ind:]

    X_train, X_test = split(X)
    Y_train, Y_test = split(Y)

    return X_train, X_test, Y_train, Y_test


def get_data(csv_fname, video_fname, avg_fname,
             num_frames=None, start_frame=0,
             OBJECTS=['person'], resol=(50, 50),
             center=True, dtype='float32', train_ratio=0.6):
    def print_class_numbers(Y, nb_classes):
        classes = np_utils.probas_to_classes(Y)
        for i in xrange(nb_classes):
            print 'class %d: %d' % (i, np.sum(classes == i))

    print '\tParsing %s, extracting %s' % (csv_fname, str(OBJECTS))
    all_counts = noscope.DataUtils.get_binary(csv_fname, limit=num_frames, OBJECTS=OBJECTS, start=start_frame)
    print '\tRetrieving all frames from %s' % video_fname
    all_frames = noscope.VideoUtils.get_all_frames(
            len(all_counts), video_fname, scale=resol, start=start_frame)
    print '\tSplitting data into training and test sets'
    X_train, X_test, Y_train, Y_test = to_test_train(avg_fname, all_frames, all_counts)

    nb_classes = all_counts.max() + 1
    print '(train) positive examples: %d, total examples: %d' % \
        (np.count_nonzero(np_utils.probas_to_classes(Y_train)),
         len(Y_train))
    print_class_numbers(Y_train, nb_classes)
    print '(test) positive examples: %d, total examples: %d' % \
        (np.count_nonzero(np_utils.probas_to_classes(Y_test)),
         len(Y_test))
    print_class_numbers(Y_test, nb_classes)

    print 'shape of image: ' + str(all_frames[0].shape)
    print 'number of classes: %d' % (nb_classes)

    data = (X_train, Y_train, X_test, Y_test)
    return data, nb_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--avg_fname', required=True)
    parser.add_argument('--num_frames', type=int, help='Number of frames')
    parser.add_argument('--start_frame', type=int)
    args = parser.parse_args()

    def check_args(args):
        assert args.objects is not None
    check_args(args)

    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    print 'Preparing data....'
    data, nb_classes = get_data(
            args.csv_in, args.video_in, args.avg_fname,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            OBJECTS=objects,
            resol=(50, 50))
    X_train, Y_train, X_test, Y_test = data

    nb_epoch = 1
    noscope.Models.try_params(
            noscope.Models.generate_conv_net_base,
            list(itertools.product(
                    *[[X_train.shape[1:]], [nb_classes],
                      [32, 64, 128, 256], [32], [0, 1, 2]])),
            data,
            args.output_dir,
            args.base_name,
            'convnet',
            objects[0],
            regression=False,
            nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()

import ast
import h5py
import numpy as np
import pandas as pd
from keras.utils import np_utils
from itertools import tee

import VideoUtils

# NOTE WARNING USERS BEWARE
# REGRESSION WILL NOT DO WHAT YOU EXPECT IT TO

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def nth_elem(list, n):
    return np.array([list[i] for i in xrange(0, len(list), n)])

def get_labels(csv_fname, limit=None, interval=1, start=0, labels=['person', 'bus', 'car']):
    df = pd.read_csv(csv_fname)
    df = df[df['frame'] >= start]
    df = df[df['frame'] < start + limit]
    df['frame'] -= start
    df = df[df['object_name'].isin(labels)]
    groups = df.set_index('frame')
    return groups

def get_raw_counts(csv_fname, OBJECTS=['person'], limit=None, interval=1, start=0):
    labels = get_labels(csv_fname, interval=interval, limit=limit, start=start)
    counts = np.zeros( (len(labels), len(OBJECTS)), dtype='uint8' )
    for i, label in enumerate(labels):
        for j, obj in enumerate(OBJECTS):
            counts[i, j] = sum(map(lambda x: 1 if x['object_name'] == obj else 0, label))
    return counts

# FIXME: efficiency
def get_counts(csv_fname, OBJECTS=['person'], limit=None, interval=1, start=0):
    labels = get_labels(csv_fname, interval=interval, limit=limit, start=start)
    counts = np.zeros( (len(labels), len(OBJECTS)), dtype='float' )
    for i, label in enumerate(labels):
        for j, obj in enumerate(OBJECTS):
            counts[i, j] = max([0] + \
                    map(lambda x: x['confidence'] if x['object_name'] == obj else 0, label))
    return counts

def get_differences(csv_fname, OBJECT, limit=None, interval=1, delay=1):
    def sym_diff(first, second):
        first_objs = set(x['object_name'] for x in first if x['object_name'] == OBJECT)
        second_objs = set(x['object_name'] for x in second if x['object_name'] == OBJECT)
        return len(first_objs.symmetric_difference(second_objs)) > 0

    labels = get_labels(csv_fname, limit=limit, interval=interval, start=delay)
    return np.array([1 if sym_diff(labels[i], labels[i-delay]) else 0 for i in xrange(delay, limit, interval)])

def get_binary(csv_fname, OBJECTS=['person'], limit=None, start=0, WINDOW=30):
    df = pd.read_csv(csv_fname)
    df = df[df['object_name'].isin(OBJECTS)]
    groups = df.set_index('frame')
    counts = map(lambda i: i in groups.index, range(start, limit + start))
    counts = np.array(counts)

    smoothed_counts = np.convolve(np.ones(WINDOW), np.ravel(counts), mode='same') > WINDOW * 0.7
    print np.sum(smoothed_counts != counts), np.sum(smoothed_counts)
    smoothed_counts = smoothed_counts.reshape(len(counts), 1)
    counts = smoothed_counts
    return counts

def smooth_binary(counts):
    for i in xrange(1, len(counts) - 1):
        if counts[i][0] > 0:
            continue
        if counts[i - 1][0] > 0 and counts[i + 1][0] > 0:
            counts[i][0] = 1
    return counts

# Given X_train, X_test, center both by the X_train mean
def center_data(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    return X_train - mean, X_test - mean

# Convert (frames, counts) into test, train
def to_test_train(all_frames, all_counts,
                  regression=False, center=True, dtype='float32', train_ratio=0.6):
    assert len(all_frames) == len(all_counts), 'Frame length should equal counts length'

    def split(arr):
        # 250 -> 100, 50, 100
        ind = int(len(arr) * train_ratio)
        if ind > 100000:
            ind = len(arr) - 100000
        return arr[:ind], arr[ind:]

    nb_classes = all_counts.max() + 1
    X = all_frames
    if regression:
        Y = np.array(all_counts)
    else:
        Y = np_utils.to_categorical(all_counts, nb_classes)

    if center:
        X_train, X_test = center_data(*split(X))
        X_train = X_train.astype(dtype)
        X_test = X_test.astype(dtype)
    else:
        X_train, X_test = split(X)
    Y_train, Y_test = split(Y)
    return X_train, X_test, Y_train, Y_test

def read_coco_dataset(coco_dir, object, resol=50):
    def read_hdf5_file(coco_dir, object, resol, data_type):
        fname = '%s/%s_%d_%s2014.h5' % (coco_dir, object, resol, data_type)
        h5f = h5py.File(fname, 'r')
        X = h5f['images'][:]
        Y = h5f['labels'][:].astype('uint8')
        # shuffle X and Y in unison
        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)
        h5f.close()
        return X, Y

    X_train, Y_train = read_hdf5_file(coco_dir, object, resol, 'train')
    X_val, Y_val = read_hdf5_file(coco_dir, object, resol, 'val')

    assert np.max(Y_train) == np.max(Y_val)
    nb_classes = np.max(Y_train) + 1
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_val = np_utils.to_categorical(Y_val, nb_classes)

    return X_train, Y_train, X_val, Y_val

def get_data(csv_fname, video_fname, binary=False, num_frames=None,
             regression=False, OBJECTS=['person'], resol=(50, 50),
             center=True, dtype='float32', train_ratio=0.6):
    def print_class_numbers(Y, nb_classes):
        classes = np_utils.probas_to_classes(Y)
        for i in xrange(nb_classes):
            print 'class %d: %d' % (i, np.sum(classes == i))

    print '\tParsing %s, extracting %s' % (csv_fname, str(OBJECTS))
    if binary:
        all_counts = get_binary(csv_fname, limit=num_frames, OBJECTS=OBJECTS)
    else:
        all_counts = get_counts(csv_fname, limit=num_frames, OBJECTS=OBJECTS)
    print '\tRetrieving all frames from %s' % video_fname
    all_frames = VideoUtils.get_all_frames(
            len(all_counts), video_fname, scale=resol, dtype=dtype)
    print '\tSplitting data into training and test sets'
    X_train, X_test, Y_train, Y_test = to_test_train(
            all_frames, all_counts, regression=regression,
            center=center, dtype=dtype, train_ratio=train_ratio)
    if regression:
        nb_classes = 1
        print '(train) mean, std: %f, %f' % \
            (np.mean(Y_train), np.std(Y_train))
        print '(test) mean, std: %f %f' % \
            (np.mean(Y_test), np.std(Y_test))
    else:
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

def get_class_weights(Y_train, class_weight_factor=1.0):
    n_classes = max(Y_train) + 1
    class_multiplier = np.array([1.0*class_weight_factor, 1.0/class_weight_factor])
    class_weights = float(len(Y_train)) / (n_classes*np.bincount(Y_train)*class_multiplier)
    return dict(zip(range(n_classes), class_weights))

def output_csv(csv_fname, stats, headers):
    df = pd.DataFrame(stats, columns=headers)
    df.to_csv(csv_fname, index=False)

def confidences_to_csv(csv_fname, confidences, OBJECT):
    col_names = ['frame', 'labels']
    labels = map(lambda conf: [{'confidence': conf, 'object_name': OBJECT}],
                 confidences)
    # because past fuccboi DK make yolo_standalone 1-indexed
    frames = range(1, len(confidences) + 1)
    output_csv(csv_fname, zip(frames, labels), col_names)


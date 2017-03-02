#! /usr/bin/env python

import argparse
import numpy as np
from noscope import VideoUtils, DataUtils
from noscope.StatsUtils import output_csv
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def get_frames(video_fname, num_frames, frame_interval, scale):
    print 'Retrieving %d frames from %s' % (num_frames, video_fname)
    return VideoUtils.get_all_frames(num_frames, video_fname, frame_interval, scale=scale)

def get_features(frames):
    return VideoUtils.get_hog_features(frames)

def evaluate_model(model, X, y_true):
    y_preds = model.predict(X)
    confusion = confusion_matrix(y_true, y_preds)

    TN = float(confusion[0][0])
    FN = float(confusion[1][0])
    TP = float(confusion[1][1])
    FP = float(confusion[0][1])
    metrics = [
               ('TP', TP),
               ('TN', TN),
               ('FP', FP),
               ('FN', FN),
               ('accuracy', (TP + TN) / (TP + FP + TN + FN)),
               ('f1', (2 * TP) / (2 * TP + FP + FN))
              ]
    return metrics

def get_model(model_type, X_train, Y_train, class_weights):
    if model_type == 'lr':
        lr = LogisticRegression(class_weight=class_weights)
        model = lr.fit(X_train, Y_train)
    elif model_type == 'lsvm':
        from sklearn.svm import LinearSVC
        svm = LinearSVC(class_weight=class_weights)
        model = svm.fit(X_train, Y_train)
    elif model_type == 'rsvm':
        from sklearn.svm import SVC
        svm = SVC(class_weight=class_weights, kernel='rbf')
        model = svm.fit(X_train, Y_train)
    elif model_type == 'psvm':
        from sklearn.svm import SVC
        svm = SVC(class_weight=class_weights, kernel='poly')
        model = svm.fit(X_train, Y_train)
    else:
        import sys
        print 'Invalid model type: %s' % model_type
        sys.exit(1)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--csv_out', required=True, help='CSV output filename--do NOT confuse with csv_in')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames to use to form training and test set. \
                    Default: 1000')
    parser.add_argument('--frame_interval', type=int, default=30, help='Number of frames to skip when creating training and \
                    test sets. Must be greater than 0. Default: 30')
    parser.add_argument('--object', required=True, help='Object(s) to classify. Multiple values should be separated by a comma \
                    (e.g., person,car)')
    parser.add_argument('--scale', type=float, default=0.1, help='Scale factor applied to each frame. Default: 0.1')
    parser.add_argument('--sample', dest='sample_data', action='store_true')
    parser.add_argument('--no_sample', dest='sample_data', action='store_false')
    parser.set_defaults(sample_data=True)
    parser.add_argument('--test_ratio', type=float, default=0.5, help='Ratio between test and training set. Default: 0.5')
    parser.add_argument('--models', default='lsvm',
        help='Type of model: Logistic Regression (lr), Linear SVM (lsvm), rbf SVM (rsvm), or polynomial SVM (psvm). \
                    Multiple values must be separated by comma (e.g., lr,lsvm). Default: lsvm')
    parser.add_argument('--class_weight_factor', type=float, default=1.0,
        help='How much to increase weight of positive training examples, and decrease weight of negative examples. \
                    Higher -> fewer false negatives, more false positives. Default: 1.0')
    args = parser.parse_args()
    csv_out = args.csv_out
    if args.frame_interval <= 0:
        import sys
        print '--frame_interval must be greater than 0'
        sys.exit(1)
    print args
    models_to_try = args.models.strip().split(',')
    args_dict = args.__dict__
    del(args_dict['models'])
    del(args_dict['csv_out'])
    init_header, init_row = zip(*sorted(list(args_dict.iteritems())))
    init_header, init_row = list(init_header), list(init_row)

    print 'Retrieving %d frames from %s' % (args.num_frames, args.video_in)
    video_frames = VideoUtils.get_all_frames(args.num_frames, args.video_in, scale=args.scale,
            interval=args.frame_interval)

    print 'Retrieving %d labels from %s' % (args.num_frames, args.csv_in)
    Y = DataUtils.get_binary(args.csv_in, [args.object], limit=args.num_frames, interval=args.frame_interval)
    Y = Y.flatten()

    if args.sample_data:
        print 'Partitioning training and test sets using random sampling'
        train_ind, test_ind, Y_train, Y_test = train_test_split(np.arange(len(Y)), Y, test_size=args.test_ratio)
    else:
        print 'Partitioning training and test sets using simple strategy: first \
        %0.2f are training, remaining %0.2f are test' % (1-args.test_ratio, args.test_ratio)
        split_index = int(len(Y)*(1 - args.test_ratio))
        inds = np.arange(len(Y))
        train_ind, test_ind, Y_train, Y_test = inds[:split_index], inds[split_index:], Y[:split_index], Y[split_index:]
    print '(train) positive examples: %d, total examples: %d' % \
        (np.count_nonzero(np_utils.probas_to_classes(Y_train)),
         len(Y_train))
    print '(test) positive examples: %d, total examples: %d' % \
        (np.count_nonzero(np_utils.probas_to_classes(Y_test)),
         len(Y_test))
    class_weights = DataUtils.get_class_weights(Y_train, args.class_weight_factor)
    print 'Class weights:', class_weights

    rows = []
    print 'Getting features....'
    X = get_features(video_frames)
    X_train, X_test = X[train_ind], X[test_ind]
    for model_type in models_to_try:
        headers = init_header[:]
        row = init_row[:]
        headers.append('model')
        row.append(model_type)
        print model_type
        model = get_model(model_type, X_train, Y_train, class_weights)
        print 'evaluating on training set'
        train_metrics = evaluate_model(model, X_train, Y_train)
        for key, val in train_metrics:
            headers.append('train ' + key)
            row.append(val)
        print train_metrics
        print 'evaluating on test set'
        test_metrics =  evaluate_model(model, X_test, Y_test)
        for key, val in train_metrics:
            headers.append('test ' + key)
            row.append(val)
        print test_metrics
        rows.append(row)
    output_csv(csv_out, np.array(rows), np.array(headers))

if __name__ == '__main__':
    main()

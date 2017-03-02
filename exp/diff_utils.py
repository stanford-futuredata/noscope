#! /usr/bin/env python

import numpy as np
from noscope import VideoUtils, DataUtils, ColorHistogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def get_color_histograms(frames):
    return np.array([ColorHistogram.compute_histogram(frame) for frame in frames])

def pairwise_hstack(features):
    return np.array([np.hstack((f1, f2)) for f1, f2 in DataUtils.pairwise(features)])

def hstack(features):
    return np.array([np.hstack((f1, f2)) for f1, f2 in features])

def pairwise_hstack_multi(features):
    return np.array([np.hstack((np.hstack(f1), np.hstack(f2))) for f1, f2 in DataUtils.pairwise(features)])

def get_frames(video_fname, num_frames, frame_interval, scale):
    print 'Retrieving %d frames from %s' % (num_frames, video_fname)
    return VideoUtils.get_all_frames(num_frames, video_fname, frame_interval, scale=scale)

def get_training_labels(csv_in_fname, object, num_frames, frame_interval):
    print 'Retrieving %d labels from %s' % (num_frames, csv_in_fname)
    return DataUtils.get_binary(csv_in_fname, object, frame_interval, limit=num_frames)

def get_features(feature_type, frames):
    if feature_type == 'sift':
        return VideoUtils.get_sift_features(frames)
    elif feature_type == 'ch':
        return get_color_histograms(frames)
    elif feature_type == 'hog':
        return VideoUtils.get_hog_features(frames)
    elif feature_type == 'hog+ch':
        ch_features = get_color_histograms(frames)
        hog_features = VideoUtils.get_hog_features(frames)
        return hstack(zip(ch_features, hog_features))
    else:
        import sys
        print 'Invalid feature type: %s' % feature_type
        sys.exit(1)

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

def get_class_weights(Y_train, class_weight_factor=1.0):
    n_classes = max(Y_train) + 1
    class_multiplier = np.array([1.0*class_weight_factor, 1.0/class_weight_factor])
    class_weights = float(len(Y_train)) / (n_classes*np.bincount(Y_train)*class_multiplier)
    return dict(zip(range(n_classes), class_weights))

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


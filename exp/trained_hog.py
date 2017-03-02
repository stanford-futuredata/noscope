import noscope
from noscope.filters import HOG
import argparse
import os
import time
import numpy as np


def get_confidences(model, X, DELAY):
    X_delta = get_deltas(X, DELAY)
    probs = model.predict_proba(X_delta)
    return np.concatenate(
            (np.zeros(DELAY),
            probs[:, 1])
        )

def get_deltas(X, DELAY):
    return X[DELAY:] - X[:-DELAY]

def train_model(model_type, X_train, Y_train, DELAY):
    class_weights = noscope.DataUtils.get_class_weights(Y_train)
    X_delta = get_deltas(X_train, DELAY)
    Y_delta = get_deltas(Y_train, DELAY)
    Y_delta[Y_delta != 0] = 1
    if model_type == 'lr':
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(class_weight=class_weights)
        model = lr.fit(X_delta, Y_delta)
    elif model_type == 'svm':
        from sklearn.svm import SVC
        svm = SVC(kernel='linear', class_weight=class_weights, probability=True)
        model = svm.fit(X_delta, Y_delta)
    else:
        import sys
        print 'Invalid model type: %s' % model_type
        sys.exit(1)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV with labels')
    parser.add_argument('--video_in', required=True, help='Video input')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', type=int, required=True, help='Number of frames')
    parser.add_argument('--resol', type=int, required=True, help='Resolution. Square')
    parser.add_argument('--delay', type=int, required=True, help='Delay')
    args = parser.parse_args()

    DELAY = args.delay
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
            regression=True,
            resol=(args.resol, args.resol))
    X_train, Y_train, X_test, _ = data
    Y_train = Y_train.flatten()
    Y_train = Y_train.astype('uint8')

    base_fname = os.path.join(args.output_dir, args.base_name)
    print 'Computing features....'
    for feature_name, feature_fn, metrics in [('hog', HOG.compute_feature, HOG.DIST_METRICS)]:
        print feature_name
        X_train_feats = np.array([feature_fn(X) for X in X_train])
        X_test_feats = np.array([feature_fn(X) for X in X_test])
        X_all_feats = np.concatenate([X_train_feats, X_test_feats])

        for model_type in ['lr', 'svm']:
            model = train_model(model_type, X_train_feats, Y_train, DELAY)
            csv_fname = '%s_%s_delay%d_resol%d.csv' % (base_fname, feature_name + '-' +
                    model_type, DELAY, args.resol)
            print csv_fname

            begin = time.time()
            confidences = get_confidences(model, X_all_feats, DELAY)
            end = time.time()
            print end - begin
            noscope.DataUtils.confidences_to_csv(csv_fname, confidences, objects[0])

if __name__ == '__main__':
    main()

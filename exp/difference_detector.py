#! /usr/bin/env python

import argparse
import numpy as np
from noscope import VideoUtils, DataUtils, StatsUtils
from noscope.filters import HOG, RawImage, ColorHistogram, SIFT
from sklearn.metrics import confusion_matrix


def get_features(feature_fn, frames):
    return np.array([feature_fn(frame) for frame in frames])

def get_distances(dist_fn, features, delay):
    return np.array([dist_fn(features[i], features[i-delay]) for i in
        xrange(delay, len(features))])

def get_feature_and_dist_fns(feature_type):
    if feature_type == 'hog':
        return (HOG.compute_feature, HOG.get_distance_fn, HOG.DIST_METRICS)
    elif feature_type == 'sift':
        return (SIFT.compute_feature, SIFT.get_distance_fn, SIFT.DIST_METRICS)
    elif feature_type == 'ch':
        return (ColorHistogram.compute_feature, ColorHistogram.get_distance_fn, ColorHistogram.DIST_METRICS)
    elif feature_type == 'raw':
        return (RawImage.compute_feature, RawImage.get_distance_fn, RawImage.DIST_METRICS)
    else:
        import sys
        print 'Invalid feature type: %s' % feature_type
        sys.exit(1)

def evaluate_model(y_preds, y_true):
    confusion = confusion_matrix(y_true, y_preds)

    TN = float(confusion[0][0])
    FN = float(confusion[1][0])
    TP = float(confusion[1][1])
    FP = float(confusion[0][1])
    metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'false negative ratio': FN / (FN + TP),
        'filtration': TN / (TN + FP),
        'true positive ratio': TP / (TP + FP),
        'accuracy': (TP + TN) / (TP + FP + TN + FN),
        'f1': (2 * TP) / (2 * TP + FP + FN)
    }
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--csv_out_base', required=True, help='CSV output filename--do NOT confuse with csv_in')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames to use to form training and test set. \
                    Default: 1000')
    parser.add_argument('--frame_delay', type=int, default=15, help='Delta between current frame and previous frame to compare \
                    against. Must be greater than 0. Default: 15')
    parser.add_argument('--object', required=True, help='Object to detect.')
    parser.add_argument('--scale', type=float, default=0.1, help='Scale factor applied to each frame. Default: 0.1')
    parser.add_argument('--features', default='hog', help='Type of features: HOG (hog), SIFT (sift), Color Histogram (ch), \
                    or raw images (raw). Multiple values must be separated by comma (e.g., hog,ch). Default: hog')
    args = parser.parse_args()
    csv_out_base = args.csv_out_base
    video_in = args.video_in
    csv_in = args.csv_in
    if args.frame_delay <= 0:
        import sys
        print '--frame_delay must be greater than 0'
        sys.exit(1)
    print args
    features_to_try = args.features.strip().split(',')
    args_dict = args.__dict__
    del(args_dict['features'])
    del(args_dict['csv_out_base'])
    del(args_dict['video_in'])
    del(args_dict['csv_in'])
    init_header, init_row = zip(*sorted(list(args_dict.iteritems())))
    init_header, init_row = list(init_header), list(init_row)

    print 'Retrieving %d frames from %s' % (args.num_frames, video_in)
    video_frames = VideoUtils.get_all_frames(args.num_frames, video_in, scale=args.scale, interval=1)

    print 'Retrieving %d labels from %s' % (args.num_frames, csv_in)
    Y_truth = DataUtils.get_differences(csv_in, args.object, limit=args.num_frames, interval=1, delay=args.frame_delay)

    header = init_header + ['feature', 'distance metric', 'threshold', 'filtration', 'true positive ratio']
    rows = []
    for feature_type in features_to_try:
        row_with_feat = init_row[:]
        row_with_feat.append(feature_type)
        print feature_type
        feature_fn, get_distance_fn, dist_metrics_to_try = get_feature_and_dist_fns(feature_type)
        features = get_features(feature_fn, video_frames)
        for dist_metric in dist_metrics_to_try:
            recorder = StatsUtils.OutputRecorder('%s_%s_%s.csv' % (csv_out_base, feature_type, dist_metric))
            row = row_with_feat[:]
            row.append(dist_metric)
            print dist_metric
            dists = get_distances(get_distance_fn(dist_metric), features, args.frame_delay)
            prev_thresh = None
            prev_metrics = None
            best_Y_preds = None

            thresholds_to_try = np.linspace(np.min(dists), np.max(dists), 250)
            for thresh in thresholds_to_try[1:]:
                Y_preds = dists > thresh
                metrics = evaluate_model(Y_preds, Y_truth)
                if metrics['false negative ratio']  > 0.01:
                    break
                prev_metrics = metrics
                prev_thresh = thresh
                best_Y_preds = Y_preds

            if not prev_metrics:
                prev_thresh = 0.0
                prev_metrics = {
                    'filtration': 0.0,
                    'true positive ratio': 0.0
                }

            print prev_thresh, prev_metrics['filtration'], prev_metrics['true positive ratio']
            _row = row[:]
            _row.append(prev_thresh)
            for key in ['filtration', 'true positive ratio']:
                val = prev_metrics[key]
                _row.append(val)
            rows.append(_row)
            for i in xrange(args.frame_delay):
                recorder.add_row(False, args.object)
            if best_Y_preds is not None:
                for pred in best_Y_preds:
                    recorder.add_row(pred, args.object)
                recorder.output_csv()
    StatsUtils.output_csv(csv_out_base + '_summary.csv', np.array(rows), np.array(header))

if __name__ == '__main__':
    main()

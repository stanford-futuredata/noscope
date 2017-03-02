import noscope
import cv2
import itertools
import argparse
import os
import time
import numpy as np


# NOTE: assume metric is commutative
def get_confidences(X, DELAY, metric):
    confidences = [0 for i in xrange(DELAY)] + \
        map(metric, X[DELAY:], X[:-DELAY])
    confidences = np.array(confidences, dtype=float)

    # Normalize to 0-1
    confidences -= np.min(confidences)
    if np.max(confidences) > 0:
        confidences /= np.max(confidences)

    return confidences


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
            regression=False,
            resol=(args.resol, args.resol))
    X_train, Y_train, X_test, Y_test = data

    # For generating the CSVs, we don't really need to "learn" anything
    X_all = np.concatenate([X_train, X_test])
    Y_all = np.concatenate([Y_train, Y_test])

    metrics = noscope.Metrics()
    base_fname = os.path.join(args.output_dir, args.base_name)
    for name, metric in metrics.iteritems():
        csv_fname = '%s_%s_delay%d_resol%d.csv' % (base_fname, name, DELAY, args.resol)
        print csv_fname

        begin = time.time()
        confidences = get_confidences(X_all, DELAY, metric)
        end = time.time()
        print end - begin
        noscope.DataUtils.confidences_to_csv(csv_fname, confidences, objects[0])


if __name__ == '__main__':
    main()

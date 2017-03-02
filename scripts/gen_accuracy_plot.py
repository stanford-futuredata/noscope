#!/usr/bin/env python

from __future__ import division
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import multiprocessing as mp
import numpy as np
import os
import sys

DIRNAME = os.path.dirname(os.path.realpath(__file__))
WINDOW_SIZE = 30
WINDOW_THRES = 28

PRESENT = 1
NOT_PRESENT = 0

def parse_csv_file(f):
    f.readline() # skip first line (header)

    frame_objects = []
    for line in f:
        frame_objects.append( eval(line.split(',', 1)[1].replace('"', '').replace("nan", "0.0")) )

    return frame_objects

def truth_label_indicator(frame_objects, label):
    for frame_object in frame_objects:
        if label in frame_object.values():
            return PRESENT
    return NOT_PRESENT

def filter_label_confidence(frame_objects, label):
    for frame_object in frame_objects:
        if label in frame_object.values():
            return frame_object['confidence']
    return NOT_PRESENT

def filter_label_indicator(bundle):
    confidences, thres = bundle[0], bundle[1]

    indicator = []
    for i in xrange(0, len(confidences), WINDOW_SIZE):
        c = confidences[i:i+WINDOW_SIZE].max()
        indicator.append( int(c >= thres) )
    return np.asarray(indicator)

def smooth_and_decimate_indicator(indicator_array):
    rolling_sum = np.cumsum(indicator_array, dtype=int)
    rolling_sum[WINDOW_SIZE:] = rolling_sum[WINDOW_SIZE:] - rolling_sum[:-WINDOW_SIZE]

    shift = int((WINDOW_SIZE-1) / 2)
    rolling_sum = np.append(rolling_sum, np.zeros((shift,), dtype=int))
    rolling_sum = rolling_sum[shift:]

    # overlapping windows
    # smoothed_indicator = map(lambda x: int(x >= WINDOW_THRES), rolling_sum)

    # non-overlaping window
    smoothed_indicator = map(lambda x: int(x >= WINDOW_THRES), rolling_sum[0::WINDOW_SIZE])
    #smoothed_indicator = reduce(lambda x,y: x+y, map(lambda x: [x for _ in xrange(WINDOW_SIZE)], smoothed_indicator) )
    #smoothed_indicator = smoothed_indicator[:len(indicator_array)]

    return smoothed_indicator

################################################################################
# begin the script
################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write('Usage:\n\t' + os.path.basename(sys.argv[0]) + ' LABEL TRUTH_CSV OTHER_CSV\n\n')
        sys.exit(1)

    label = sys.argv[1]

    truth_frames = None
    other_frames = None
    with open(sys.argv[2], 'r') as truth_csv_file:
        with open(sys.argv[3], 'r') as other_csv_file:
            truth_frames = parse_csv_file(truth_csv_file)
            other_frames = parse_csv_file(other_csv_file)

    truth_indicator = np.asarray( map(lambda x: truth_label_indicator(x, label), truth_frames) )
    windowed_truth_indicator = smooth_and_decimate_indicator(truth_indicator)

    N = 400
    pool = mp.Pool(2)
    confidence_thres = [ i / N for i in xrange(N) ]
    other_confidence = np.asarray( map(lambda x: filter_label_confidence(x, label), other_frames) )

    bundle = [ (other_confidence, thres) for thres in confidence_thres ]
    other_indicators = pool.map(filter_label_indicator, bundle)

    #print other_indicators

    fp = []
    fn = []
    for i in xrange(N):
        diff = other_indicators[i] - windowed_truth_indicator
        diff_fp = diff == 1
        diff_fn = diff == -1

        fn.append( sum(diff_fn) / sum(windowed_truth_indicator) )
        fp.append( sum(diff_fp) / (len(windowed_truth_indicator) - sum(windowed_truth_indicator)) )

    # plt.plot(fp, fn, marker='x',markersize=10, color='k')
    # plt.plot(fp, [0.01 for _ in xrange(len(fn))], linestyle='--', color='r')

    # red_patch = mpatches.Patch(color='red', label='1% false negative line')
    # #black_patch = mpatches.Patch(color='black', label='gap length')
    # plt.legend(handles=[red_patch])

    # plt.title(sys.argv[3])
    # plt.xlabel('false positive (%)')
    # plt.ylabel('false negative (%)')
    # plt.savefig('plots/plot-' + os.path.basename(sys.argv[3]) + '.pdf')
    # plt.savefig('plots/plot-' + os.path.basename(sys.argv[3]) + '.png')
    # plt.clf()

    # output csv result
    idx = len(fn) - len(filter(lambda x: x > 0.01, fn)) - 1

    # choose the closest
    #print fn
    #print fp
    if (0.01 - fn[idx] > fn[idx+1] - 0.01):
        idx = idx + 1

    thres = confidence_thres[idx]
    fn_rate = fn[idx]
    fp_rate = fp[idx]

    filen = os.path.basename(sys.argv[3])
    eles = filen.split('_')

    videon = eles[0]
    feature = eles[1]
    try:
        methodn = eles[2]
        learned = eles[3]
        delay = eles[4].split('delay')[1]
        resolution = eles[5].split('resol')[1]
        ref = eles[6].split('ref-index')[1]
        block_size = eles[7].split('num-blocks')[1].split('.')[0]
    except:
        methodn = '_'.join(eles[2:4])
        learned = eles[4]
        delay = eles[5].split('delay')[1]
        resolution = eles[6].split('resol')[1]
        ref = eles[7].split('ref-index')[1]
        block_size = eles[8].split('num-blocks')[1].split('.')[0]

    result = "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(filen,
                                                               videon,
                                                               feature,
                                                               methodn,
                                                               learned,
                                                               delay,
                                                               resolution,
                                                               ref,
                                                               block_size,
                                                               fn_rate,
                                                               fp_rate,
                                                               thres)
    sys.stdout.write(result)
    sys.stderr.write(result)

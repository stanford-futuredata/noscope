#!/usr/bin/env python

from __future__ import division
import noscope_accuracy as v
import multiprocessing as mp
import numpy as np
import json
import sys
import os

DIRNAME = os.path.dirname(os.path.realpath(__file__))
POOL_SIZE = 48

GRID_SIZE = 1000

def cnn_param_grid_search(p):
    # unpack the params 
    yolo_indicator, video_stats, diff_thres, diff_mask, cnn_confidences, cnn_confidences_range, \
    TARGET_CNN_FALSE_NEGATIVE_RATE, TARGET_CNN_FALSE_POSITIVE_RATE, MAX_DIFF_FALSE_NEGATIVE_RATE, \
    TRAIN_START_IDX, TRAIN_END_IDX = p

    # grid search over the cnn_confidences and get the accruacy dict from v.*
    thresholds = np.linspace(cnn_confidences_range[0], cnn_confidences_range[1], GRID_SIZE)
    accuracies = []
    diff_sum = np.sum(diff_mask)
    base_stats = {
            'threshold_skip_distance': video_stats['skip_distance'],
            'num_diff_evals': video_stats['num_frames'] - video_stats['num_skipped_frames'],
            'num_cnn_evals': diff_sum * (v.WINDOW_SIZE // video_stats['skip_distance']) - \
                (v.WINDOW_SIZE - ((TRAIN_END_IDX - TRAIN_START_IDX) % v.WINDOW_SIZE)),
            'skip_dd': False,
            'skip_cnn': False,
    }
    for thres in thresholds:
        cnn_indicator = cnn_confidences >= thres
        noscope_indicator = diff_mask & cnn_indicator

        acc = v.accuracy(yolo_indicator, noscope_indicator)
        acc.update(base_stats)
        accuracies.append( acc )

    # identify the sets of params that most closely match the target fp and fn rates
    fn = map(lambda x: x['false_negative'], accuracies)
    fp = map(lambda x: x['false_positive'], accuracies)

    lower_idx = max( GRID_SIZE - len( filter(lambda x: x > TARGET_CNN_FALSE_NEGATIVE_RATE, fn) ) - 1, 0 )
    upper_idx = len( filter(lambda x: x > TARGET_CNN_FALSE_POSITIVE_RATE, fp) ) - 1

    # be conservative with the thresholds
    if( lower_idx != 0 ):
        lower_idx = lower_idx - 1

    if( upper_idx != len(fp)-1 ):
        upper_idx = upper_idx + 1

    # if the lower and upper bound cross, pick a point in the middle
    if (lower_idx > upper_idx):
        middle = int( (lower_idx + upper_idx) / 2 )
        lower_idx = middle
        upper_idx = middle
    
    lower_thres = thresholds[lower_idx]
    upper_thres = thresholds[upper_idx]

    # output the result
    # print lower_idx, upper_idx 
    # print lower_thres, upper_thres
    # print fn[lower_idx], fp[upper_idx]
    # print fn, fp
    
    # assume YOLO will catch everything that gets passed through
    params = accuracies[lower_idx]
    params['threshold_diff'] = diff_thres
    params['threshold_lower_cnn'] = lower_thres
    params['threshold_upper_cnn'] = upper_thres
    
    positive_passthrough = fn[upper_idx] - fn[lower_idx]
    negative_passthrough = fp[lower_idx] - fp[upper_idx]

    positive_count = sum(map(lambda x: x==1, yolo_indicator))
    negative_count = sum(map(lambda x: x==0, yolo_indicator))
    total_passthrough = (positive_passthrough*positive_count + negative_passthrough*negative_count) / (positive_count + negative_count)
    _total_passthrough = (positive_passthrough*params['num_true_positives'] + negative_passthrough*params['num_true_negatives']) / params['num_windows']
    assert(total_passthrough == _total_passthrough)
    
    params['false_negative'] = fn[lower_idx]
    params['false_positive'] = fp[upper_idx]
    params['accuracy'] = 1 - (params['num_true_positives']*fn[lower_idx] + params['num_true_negatives']*fp[upper_idx]) / params['num_windows']
    # params['passthrough_positive'] = positive_passthrough
    # params['passthrough_negative'] = negative_passthrough
    # params['passthrough_total'] = total_passthrough

    params['num_yolo_evals'] = np.sum( 
        np.repeat(
            diff_mask & 
            (cnn_confidences < upper_thres) & 
            (cnn_confidences > lower_thres),
            v.WINDOW_SIZE)[:(TRAIN_END_IDX - TRAIN_START_IDX):params['threshold_skip_distance']]
    )

    return params

def runtime_estimator(params, TARGET_CNN_FALSE_NEGATIVE_RATE):

    # cost of various components of pipeline
    COST_CONST = 1
    COST_DIFF = 10
    COST_CNN = 400
    COST_YOLO = 60000

    runtime_cost = 0
    runtime_cost += COST_CONST * params['num_windows']
    runtime_cost += COST_DIFF * params['num_diff_evals']
    runtime_cost += COST_CNN * params['num_cnn_evals']
    runtime_cost += COST_YOLO * params['num_yolo_evals']

    if (params['skip_cnn'] == False):
        runtime_cost += 1080000

    if (params['false_negative'] > 1.5*TARGET_CNN_FALSE_NEGATIVE_RATE ):
        runtime_cost = float('inf')
    
    params['optimizer_cost'] = runtime_cost
    params['optimizer_predicted_speedup'] = v.WINDOW_SIZE * params['num_windows'] * COST_YOLO / runtime_cost
    
    return params
    
def param_search(yolo_frames, noscope_stats, noscope_frames,
                 TRAIN_START_IDX, TRAIN_END_IDX,
                 TARGET_CNN_FALSE_NEGATIVE_RATE, TARGET_CNN_FALSE_POSITIVE_RATE, MAX_DIFF_FALSE_NEGATIVE_RATE, 
                 diff_confidence_range, cnn_confidence_range):
    # find all the thresholds for the diff filter that are less than the max
    yolo_indicator = np.asarray(v.window_yolo(yolo_frames))

    nodd_results = []
    nocnn_results = []
    cnn_grid_search_params = []
    for skip_distance in [10, 30]:

        diff_confidences = np.asarray( map(lambda x: x['diff_confidence'], noscope_frames) )

        # zero out some confidences to simulate skipping
        tmp = np.copy( diff_confidences[::skip_distance] )
        diff_confidences[:] = 0
        diff_confidences[::skip_distance] = tmp

        stats = dict.copy(noscope_stats)
        stats['skip_distance'] = skip_distance
        stats['num_skipped_frames'] = len(diff_confidences) - len( diff_confidences[::skip_distance] )
        
        windowed_diff_confidences = []
        for i in xrange(0, len(diff_confidences), v.WINDOW_SIZE):
            windowed_diff_confidences.append( diff_confidences[i:i+v.WINDOW_SIZE].max() )
        windowed_diff_confidences = np.asarray(windowed_diff_confidences)

        assert(len(windowed_diff_confidences) == len(yolo_indicator)) # are the inputs the same size?
        combined = np.transpose( np.vstack((windowed_diff_confidences, yolo_indicator)) )
        combined_sorted = combined[combined[:,0].argsort()]

        cumulative_false_negatives = np.cumsum( combined_sorted, axis=0 )[:, 1] 
        threshold = MAX_DIFF_FALSE_NEGATIVE_RATE * np.sum(yolo_indicator)
        max_confidence_idx = np.sum( cumulative_false_negatives <= threshold ) - 1
        max_confidence = combined_sorted[max_confidence_idx,0]

        diff_min = diff_confidence_range[0]
        diff_max = max_confidence

        # collects all the params for parallel optimization
        diff_confidences_grid = np.linspace(diff_min, diff_max, GRID_SIZE)
    
        cnn_confidences = np.asarray( map(lambda x: x['cnn_confidence'], noscope_frames) )

        # zero out some confidences to simulate skipping
        tmp = np.copy( cnn_confidences[::skip_distance] )
        cnn_confidences[:] = 0
        cnn_confidences[::skip_distance] = tmp

        windowed_cnn_confidences = []
        for i in xrange(0, len(cnn_confidences), v.WINDOW_SIZE):
            windowed_cnn_confidences.append( cnn_confidences[i:i+v.WINDOW_SIZE].max() )
        windowed_cnn_confidences = np.asarray(windowed_cnn_confidences)

        args = [ (yolo_indicator, 
                  stats,
                  diff_confidences_grid[i],
                  windowed_diff_confidences >= diff_confidences_grid[i],
                  windowed_cnn_confidences,
                  cnn_confidence_range,
                  TARGET_CNN_FALSE_NEGATIVE_RATE,
                  TARGET_CNN_FALSE_POSITIVE_RATE,
                  MAX_DIFF_FALSE_NEGATIVE_RATE,
                  TRAIN_START_IDX,
                  TRAIN_END_IDX,
              ) for i in xrange(GRID_SIZE) ]
        
        cnn_grid_search_params += args

        # try skipping the DD
        # diff_mask = windowed_diff_confidences >= windowed_diff_confidences.min()
        # nodd_args = (yolo_indicator, 
        #              stats,
        #              0, 
        #              diff_mask, 
        #              windowed_cnn_confidences,
        #              cnn_confidence_range,
        #              TARGET_CNN_FALSE_NEGATIVE_RATE, 
        #              TARGET_CNN_FALSE_POSITIVE_RATE, 
        #              MAX_DIFF_FALSE_NEGATIVE_RATE)
        
        # nodd_acc = cnn_param_grid_search( nodd_args[:]  )
        # nodd_acc['threshold_skip_distance'] = stats['skip_distance']
        # nodd_acc['num_diff_evals'] = 0
        # nodd_acc['threshold_diff'] = 0
        # nodd_acc['skip_dd'] = True
        # nodd_acc['skip_cnn'] = False
        # nodd_results.append(nodd_acc)

        # try skipping the CNN 
        # for i in xrange(GRID_SIZE):
        #     diff_indicator = windowed_diff_confidences >= diff_confidences_grid[i]
        #     nocnn_acc = v.accuracy(yolo_indicator, diff_indicator)
        
        #     nocnn_acc['threshold_diff'] = diff_confidences_grid[i]
        #     nocnn_acc['threshold_lower_cnn'] = 0
        #     nocnn_acc['threshold_upper_cnn'] = 1
        #     nocnn_acc['threshold_skip_distance'] = stats['skip_distance']
        #     nocnn_acc['num_diff_evals'] = stats['num_frames'] - stats['num_skipped_frames']
        #     nocnn_acc['skip_dd'] = False
        #     nocnn_acc['skip_cnn'] = True
        #     nocnn_acc['num_cnn_evals'] = 0
        #     nocnn_acc['num_yolo_evals'] = np.sum( diff_indicator )

        #     nocnn_results.append(nocnn_acc)
            
    pool = mp.Pool(POOL_SIZE)
    
    # launch all the parallel workers
    results = pool.map(cnn_param_grid_search, cnn_grid_search_params)

    # add the no DD results
    results += nodd_results
    
    # add the skip CNN results
    results += nocnn_results

    # compute the expected runtime of each
    runtimes = map(runtime_estimator, results, [TARGET_CNN_FALSE_NEGATIVE_RATE for _ in xrange(len(results))])
    
    # pick the "best" set of params
    runtimes.sort(key=lambda x: x['optimizer_cost'])

    return runtimes

################################################################################
# begin the script
################################################################################
def main(object_name,
         yolo_csv_filename, noscope_csv_filename,
         target_fn, target_fp,
         TRAIN_START_IDX, TRAIN_END_IDX):
    TARGET_CNN_FALSE_NEGATIVE_RATE = target_fn / 2.0
    TARGET_CNN_FALSE_POSITIVE_RATE = target_fp / 2.0
    MAX_DIFF_FALSE_NEGATIVE_RATE = target_fn / 2.0

    with open(yolo_csv_filename, 'r') as yolo:
        with open(noscope_csv_filename, 'r') as noscope:
            yolo_frames = v.parse_yolo_csv(yolo, object_name, TRAIN_START_IDX, TRAIN_END_IDX)
            noscope_frames, noscope_stats = v.parse_noscope_csv(noscope, object_name, 0, TRAIN_END_IDX-TRAIN_START_IDX)
            
    #print len(yolo_frames), len(noscope_frames)
    assert(len(yolo_frames) == len(noscope_frames)) # were the same number of frames read?
    
    runtimes = param_search(yolo_frames, noscope_stats, noscope_frames,
                            TRAIN_START_IDX, TRAIN_END_IDX,
                            TARGET_CNN_FALSE_NEGATIVE_RATE, TARGET_CNN_FALSE_POSITIVE_RATE, MAX_DIFF_FALSE_NEGATIVE_RATE,
                            (noscope_stats['min_diff_confidence'], noscope_stats['max_diff_confidence']),
                            (noscope_stats['min_cnn_confidence'], noscope_stats['max_cnn_confidence']))
    
    optimal_params = runtimes[0]
    return optimal_params

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.stderr.write('Usage:\n\t' + os.path.basename(sys.argv[0]) + 
                         ' LABEL TRUTH_CSV OTHER_CSV TARGET_FN TARGET_FP\n\n')

        sys.stderr.write('Description:\n\t Operates on frames {} to {} in TRUTH.csv.\n\t (assumes OTHER.csv is {}) frames long.\n\n'.format(TRAIN_START_IDX, TRAIN_END_IDX, TRAIN_END_IDX-TRAIN_START_IDX))

        sys.exit(1)
                
    object_name = sys.argv[1]
    yolo_csv_filename = sys.argv[2]
    noscope_csv_filename = sys.argv[3]
    target_fn = float(sys.argv[4]) / 2.0 # be conservative
    target_fp = float(sys.argv[5]) / 2.0
    
    optimal_params = main(object_name, yolo_csv_filename, noscope_csv_filename, target_fn, target_fp)

    print json.dumps( optimal_params, indent=True, sort_keys=True )
    print "---best configuration shown---"


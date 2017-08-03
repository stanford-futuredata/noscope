#!/usr/bin/env python

from __future__ import division
import numpy as np
import hashlib
import json
import sys
import os
import timeit
import csv
from collections import namedtuple

DIRNAME = os.path.dirname(os.path.realpath(__file__))

YOLO_CONFIDENCE = 0.20
SKIP_FRAME_STATUS = 0

WINDOW_SIZE = 30
WINDOW_THRES = 28

def strip_comment_lines(lines):
    lines = map(lambda x: x.split('#')[0], lines)
    lines = filter(lambda x: x != '', lines)
    return lines

def parse_yolo_csv(f, label, start, end):
    # Pandas crashes with multiprocessing so use the CSV
    f.readline() # Skip the header
    reader = csv.reader(f)
    rows = []
    for row in reader:
        # frame,object_name,confidence,xmin,ymin,xmax,ymax
        rows.append( (int(row[0]), row[1]) )

    rows = filter(lambda row: row[0] >= start and row[0] < end, rows)
    rows = filter(lambda row: row[1] == label, rows)
    frames = set(map(lambda row: row[0], rows))

    nt = namedtuple('YoloObj', ['frame', 'object', 'confidence'])
    frame_objs = []
    for i in xrange(start, end):
        if i in frames:
            frame_objs.append(nt(i, label, 1))
        else:
            frame_objs.append(nt(i, label, 0))
    return frame_objs
    
def parse_noscope_csv(f, label, start, end):
    lines = f.read().strip().split('\n')
    metadata = lines[0]

    file_lines = strip_comment_lines( lines )
    file_lines = file_lines[start:end]
    
    # assume the following columns
    # frame,status,diff_confidence,cnn_confidence,label
    max_diff_confidence = float("-inf") 
    min_diff_confidence = float("inf")
    max_cnn_confidence = float("-inf") 
    min_cnn_confidence = float("inf")

    num_skipped_frames = 0
    frame_objects_final = []
    for line in file_lines:
        frame_num, status, diff_confidence, cnn_confidence, yolo_confidence, decision = line.split(',')
        
        frame_num = int(frame_num)
        status = int(status)
        diff_confidence = float(diff_confidence)
        cnn_confidence = float(cnn_confidence)
        decision = int(decision)
        
        if max_diff_confidence < diff_confidence:
            max_diff_confidence = diff_confidence
            
        if min_diff_confidence > diff_confidence:
            min_diff_confidence = diff_confidence

        if max_cnn_confidence < cnn_confidence:
            max_cnn_confidence = cnn_confidence
            
        if min_cnn_confidence > cnn_confidence:
            min_cnn_confidence = cnn_confidence

        obj = dict()
        obj['frame_num'] = frame_num
        obj['status'] = status
        obj['object_name'] = label
        obj['diff_confidence'] = diff_confidence
        obj['cnn_confidence'] = cnn_confidence
        obj['final_decision'] = decision
        
        if( status == SKIP_FRAME_STATUS ):
            num_skipped_frames += 1
            assert(obj['diff_confidence'] == 0)
            assert(obj['cnn_confidence'] == 0)
            
            #obj['diff_confidence'] = 0
            #obj['cnn_confidence'] = 0

        frame_objects_final.append(obj)
    
    stats = dict()
    elements = metadata.split()
    for i in xrange(len(elements)):
        if elements[i] in ['diff_thresh:', 'distill_thresh_lower:', 'distill_thresh_upper:', 'skip:']:
            key = elements[i].split(':')[0]
            try:
                val = float(elements[i+1].split(',')[0])
                stats[key] = val
            except:
                stats[key] = None                

        if elements[i] == 'skip_cnn:':
            key = elements[i].split(':')[0]
            val = bool( int(elements[i+1].split(',')[0]) )
            stats[key] = val
            
        if elements[i] == 'runtime:':
            key = elements[i].split(':')[0]
            val = float(elements[-1])
            stats[key] = val

    for key in ['diff_thresh', 'distill_thresh_lower', 'distill_thresh_upper', 'skip', 'skip_cnn', 'runtime']:
        if key not in stats.keys():
            print 'missing the following header:', key
            sys.exit(-1)

    stats['max_diff_confidence'] = max_diff_confidence
    stats['min_diff_confidence'] = min_diff_confidence
    stats['max_cnn_confidence'] = max_cnn_confidence
    stats['min_cnn_confidence'] = min_cnn_confidence
    stats['num_frames'] = len(frame_objects_final)
    stats['num_skipped_frames'] = num_skipped_frames 

    # print '\n'.join(map(str, frame_objects_final[1600:1630]))
    return frame_objects_final, stats

def smooth_indicator(indicator):
    rolling_sum = np.cumsum( np.asarray(indicator) )
    rolling_sum[WINDOW_SIZE:] = rolling_sum[WINDOW_SIZE:] - rolling_sum[:-WINDOW_SIZE]

    shift = int((WINDOW_SIZE-1) / 2)
    rolling_sum = np.append(rolling_sum, np.zeros((shift,), dtype=int))
    rolling_sum = rolling_sum[shift:]
    indicator_smooth = map(lambda x: int(x >= WINDOW_THRES), rolling_sum[0::WINDOW_SIZE])

    return np.asarray( indicator_smooth )

def window_yolo(frames):
    true_indicator = np.asarray( map(lambda x: int(x.confidence > YOLO_CONFIDENCE), frames) )
    
    # smooth and window the yolo labels
    return smooth_indicator(true_indicator)
    
def window_noscope(frames):
    noscope_indicator = np.asarray( map(lambda x: x['final_decision'], frames) )
    noscope_status = np.asarray( map(lambda x: x['status'], frames) )
    
    # fill in skipped frames
    assert(len(noscope_indicator) == len(noscope_status))
    prev_status = noscope_status[0]
    for i in xrange(1, len(noscope_indicator)):
        if ( noscope_status[i] == SKIP_FRAME_STATUS ):
            noscope_indicator[i] = noscope_indicator[i-1]

    # smooth and window the noscope labels
    return smooth_indicator(noscope_indicator)

def _OLD_window_noscope(frames, noscope_stats, yolo_indicator, DIFF_THRES, CNN_LOWER_THRES, CNN_UPPER_THRES):
    diff_indicator = []
    diff_confidences = np.asarray( map(lambda x: x['diff_confidence'], frames) ) 
    for i in xrange(0, len(diff_confidences), WINDOW_SIZE):
        c = diff_confidences[i:i+WINDOW_SIZE].max()
        diff_indicator.append( int(c >= DIFF_THRES) )
    diff_indicator = np.asarray( diff_indicator )

    cnn_indicator = []
    cnn_confidences = np.asarray( map(lambda x: x['cnn_confidence'], frames) )
    for i in xrange(0, len(cnn_confidences), WINDOW_SIZE):
        c = cnn_confidences[i:i+WINDOW_SIZE].max()
        if(c > CNN_UPPER_THRES):
            cnn_indicator.append(1)
        elif(c < CNN_LOWER_THRES):
            cnn_indicator.append(0)
        else:
            cnn_indicator.append(yolo_indicator[i//WINDOW_SIZE])
    cnn_indicator = np.asarray( cnn_indicator )

    # print len(diff_indicator), len(cnn_indicator)

    if( not noscope_stats['skip_cnn'] ):
        return diff_indicator & cnn_indicator
    else:
        return diff_indicator

def accuracy(yolo_indicator, noscope_indicator):
    # compute the accuracy 
    difference_indicator = noscope_indicator - yolo_indicator 
    false_positives = difference_indicator == 1
    false_negatives = difference_indicator == -1

    yolo_sum = np.sum(yolo_indicator)
    nb_fp = np.sum(false_positives)
    nb_fn = np.sum(false_negatives)

    error_rate = (nb_fp + nb_fp) / len(difference_indicator)
    false_positive_rate = nb_fp / (len(yolo_indicator) - yolo_sum)
    false_negative_rate = nb_fn / yolo_sum
    
    # report the results
    results = dict()
    results['accuracy'] = 1 - error_rate
    results['false_positive'] = false_positive_rate
    results['false_negative'] = false_negative_rate
    results['num_true_positives'] = yolo_sum
    results['num_true_negatives'] = len(yolo_indicator) - yolo_sum
    results['num_windows'] = len(yolo_indicator)
    results['window_size'] = WINDOW_SIZE
    results['window_thres'] = WINDOW_THRES

    return results 

def accuracy2str(results):
    return json.dumps(results, indent=4, sort_keys=True)

################################################################################
# begin the script
################################################################################
def main(object_name,
         TEST_START_IDX, TEST_END_IDX,
         yolo_csv_filename, noscope_csv_filename):

    with open(yolo_csv_filename, 'r') as yolo:
        with open(noscope_csv_filename, 'r') as noscope:
            yolo_frames = parse_yolo_csv(yolo, object_name, TEST_START_IDX, TEST_END_IDX)
            noscope_frames, noscope_stats = parse_noscope_csv(noscope, object_name, 0, TEST_END_IDX-TEST_START_IDX)
            
    # print len(noscope_frames), TEST_END_IDX-TEST_START_IDX
    assert(len(noscope_frames) == TEST_END_IDX-TEST_START_IDX)
    assert(len(yolo_frames) == TEST_END_IDX-TEST_START_IDX)

    # window yolo labels
    print len(yolo_frames)
    yolo_indicator = window_yolo(yolo_frames) 
    print len(yolo_indicator)

    # window the noscope labels
    DIFF_THRES = noscope_stats['diff_thresh']
    CNN_LOWER_THRES = noscope_stats['distill_thresh_lower']
    CNN_UPPER_THRES = noscope_stats['distill_thresh_upper']
    if noscope_stats['skip_cnn']:
        print 'Note: cnn was skipped?'
        # print DIFF_THRES, CNN_LOWER_THRES, CNN_UPPER_THRES

    noscope_indicator = _OLD_window_noscope(noscope_frames, noscope_stats, yolo_indicator, DIFF_THRES, CNN_LOWER_THRES, CNN_UPPER_THRES)

    acc = accuracy(yolo_indicator, noscope_indicator)
    acc['runtime'] = noscope_stats['runtime']
    return acc

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write('Usage:\n\t' + os.path.basename(sys.argv[0]) + 
                         ' LABEL TRUTH_CSV OTHER_CSV\n\n')
        sys.stderr.write('Description:\n\t Operates on frames {} to {} in TRUTH.csv.\n\t (assumes OTHER.csv is {} frames long.)\n\n'.format(TEST_START_IDX, TEST_END_IDX, TEST_END_IDX-TEST_START_IDX))

        sys.exit(1)
                
    object_name = sys.argv[1]
    yolo_csv_filename = sys.argv[2]
    noscope_csv_filename = sys.argv[3]
    
    print accuracy2str( main(object_name, yolo_csv_filename, noscope_csv_filename) )

#!/usr/bin/env python

import noscope_accuracy as acc
import noscope_optimizer as opt
import datetime
import subprocess
import uuid
import stat
import json
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--code_dir', required=True, help='Do not use relative paths')
parser.add_argument('--data_dir', required=True, help='Do not use relative paths')
parser.add_argument('--video_name', required=True)
parser.add_argument('--gpu_num', required=True, type=int)
args = parser.parse_args()

# CODE_DIR = '/home/daniel_d_kang/code/'
# DATA_DIR = '/home/daniel_d_kang/data/'
CODE_DIR = args.code_dir
DATA_DIR = args.data_dir

RUN_BASH_SCRIPT = """
#!/bin/bash

# {date}

if [[ -z $1 ]]; then
    echo "Usage: $0 GPU_ID"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"

START_FRAME={start}
LEN={length}
END_FRAME=$((START_FRAME + LEN))

time {tf_prefix}/bazel-bin/tensorflow/noscope/noscope \\
    --diff_thresh={dd_thres} \\
    --distill_thresh_lower={cnn_lower_thres} \\
    --distill_thresh_upper={cnn_upper_thres} \\
    --skip_small_cnn={skip_cnn} \\
    --skip_diff_detection={skip_dd} \\
    --skip={kskip} \\
    --avg_fname={cnn_avg_path} \\
    --graph={cnn_path} \\
    --video={video_path} \\
    --yolo_cfg=%s/darknet/cfg/yolo.cfg \\
    --yolo_weights=%s/darknet/weights/yolo.weights \\
    --yolo_class={yolo_class} \\
    --confidence_csv={output_csv} \\
    --start_from=${{START_FRAME}} \\
    --nb_frames=$LEN \\
    --dumped_videos={dumped_videos} \\
    --diff_detection_weights={diff_detection_weights} \\
    --use_blocked={use_blocked} \\
    --ref_image={ref_image} \\
    &> {output_log}
""" % (CODE_DIR, CODE_DIR)

LABELS = dict()
LABELS[0] = 'person'
LABELS[2] = 'car'
LABELS[5] = 'bus'
LABELS[8] = 'boat'

YOLO_LABELS = dict()
YOLO_LABELS["coral-reef-long"] = (
        0,
        [("coral-reef-long_convnet_32_32_1.pb", None),
         ("coral-reef-long_convnet_32_32_2.pb", None),],
        648000 + 1188000,
        1188000,
        648000 + 1188000 * 2,
        1188000
)
YOLO_LABELS['jackson-town-square'] = (
        2,
        [('jackson-town-square_convnet_128_32_2.pb', None),
         ('jackson-town-square_convnet_128_32_0.pb', None)],
        918000,
        918000,
        918000 * 2,
        918000,
)
NO_CACHING = False

# TARGET_ERROR_RATES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25]
# TARGET_ERROR_RATES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25]
TARGET_ERROR_RATES = [0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]

TF_DIRPREFIX = os.path.join(CODE_DIR, 'tensorflow-noscope/')
DATA_DIR_PREFIX = DATA_DIR
EXPERIMENTS_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'experiments/')
VIDEO_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'videos')
VIDEO_CACHE_PREFIX = os.path.join(DATA_DIR_PREFIX, 'video-cache')
TRUTH_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'csv')

DD_MEAN_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'dd-means')
CNN_MODEL_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'cnn-models')
CNN_AVG_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'cnn-avg')

TRAIN_DIRNAME = 'train'

RUN_SCRIPT_FILENAME = 'run_testset.sh'
RUN_OPTIMIZER_SCRIPT_FILENAME = 'run_optimizerset.sh'

OUTPUT_SUMMARY_CSV = 'summary.csv'

################################################################################
# Begin script
################################################################################
video_name = args.video_name
GPU_NUM = args.gpu_num
if( video_name not in YOLO_LABELS.keys() ):
    print "video must be from the following list"
    print
    print "Possible videos:"
    for v in sorted(YOLO_LABELS.keys()):
        print "\t", v
    print
    sys.exit(1)

experiment_dir = os.path.join(EXPERIMENTS_DIR_PREFIX, video_name)
if( os.path.exists(experiment_dir) ):
    print experiment_dir, "already exists."
    print "WARNING. (remove the dir if you want to rerun)"
    print

try:
    os.mkdir( experiment_dir )
except:
    print experiment_dir, "already exists"

os.chdir( experiment_dir )

try:
    os.mkdir( TRAIN_DIRNAME )
except:
    print TRAIN_DIRNAME, "already exists"

################################################################################
# get training data for the optimizer and get ground truth for accuracy
################################################################################
print "preparing the training data (for optimizer) and getting ground truth"
yolo_label_num, pipelines, \
    TRAIN_START_IDX, TRAIN_LEN, \
    TEST_START_IDX, TEST_LEN = YOLO_LABELS[video_name]
TRAIN_END_IDX = TRAIN_START_IDX + TRAIN_LEN
TEST_END_IDX = TEST_START_IDX + TEST_LEN


train_csv_filename =  "train_" + str(TRAIN_START_IDX) + "_" + str(TRAIN_START_IDX+TRAIN_LEN) + ".csv"
train_csv_str = 'train_${START_FRAME}_${END_FRAME}.csv'

train_log_filename = "train_" + str(TRAIN_START_IDX) + "_" + str(TRAIN_START_IDX+TRAIN_LEN) + ".log"
train_log_str = 'train_${START_FRAME}_${END_FRAME}.log'

video_path = os.path.join(VIDEO_DIR_PREFIX, video_name+'.mp4')

cnn_avg_path = os.path.join(CNN_AVG_DIR_PREFIX, video_name+'.txt')

pipeline_paths = []
for cnn, dd in pipelines:
    
    dd_name = 'non_blocked_mse.src'
    if( dd is not None ):
        dd_name = dd[0]

    pipeline_path = os.path.join(EXPERIMENTS_DIR_PREFIX,
                                 video_name,
                                 TRAIN_DIRNAME,
                                 cnn+'-'+dd_name)

    cnn_path = os.path.join(CNN_MODEL_DIR_PREFIX, cnn)

    pipeline_paths.append((pipeline_path, cnn_path, dd))
    try:
        os.mkdir(pipeline_path)
    except:
        print pipeline_path, "already exists"

    train_csv_path_str = os.path.join(
        pipeline_path,
        train_csv_str
    )
    train_csv_path = os.path.join(
        pipeline_path,
        train_csv_filename
    )

    train_log_path = os.path.join(
        pipeline_path,
        train_log_str
    )

    use_blocked = 0
    diff_detection_weights = '/dev/null'
    ref_image = 0
    if dd is not None:
        use_blocked = 1
        diff_detection_weights = os.path.join( DD_MEAN_DIR_PREFIX, video_name, dd[0]  )
        ref_image = dd[1]

    run_optimizer_script = os.path.join(pipeline_path, RUN_OPTIMIZER_SCRIPT_FILENAME)

    video_cache_filename = os.path.join(
            VIDEO_CACHE_PREFIX,
            '%s_%d_%d_%d.bin' % (video_name, TRAIN_START_IDX, TRAIN_LEN, 1))

    with open(run_optimizer_script, 'w') as f:
        script = RUN_BASH_SCRIPT.format(
            date=str(datetime.datetime.now()),
            tf_prefix=TF_DIRPREFIX,
            dd_thres=0,
            cnn_lower_thres=0,
            cnn_upper_thres=0,
            skip_dd=0,
            skip_cnn=0,
            kskip=30,
            diff_detection_weights=diff_detection_weights,
            use_blocked=use_blocked,
            ref_image=ref_image,
            cnn_path=cnn_path,
            cnn_avg_path=cnn_avg_path,
            video_path=video_path,
            yolo_class=yolo_label_num,
            start=TRAIN_START_IDX,
            length=TRAIN_LEN,
            output_csv=train_csv_path_str,
            output_log=train_log_path,
            dumped_videos=video_cache_filename
        )

        f.write(script)

    st = os.stat(run_optimizer_script)
    os.chmod(run_optimizer_script, st.st_mode | stat.S_IEXEC)
    print 'obtaining the optimizer data for', pipeline_path

    if( not os.path.exists(train_csv_path) or NO_CACHING ):
        print "GPU_NUM:", GPU_NUM
        os.system( 'bash ' + run_optimizer_script + ' ' + str(GPU_NUM) ) 
    else:
        print 'WARNING: using cached results! Skipping computation.'

################################################################################
# find the best pipeline for each error rate
################################################################################
summary_file = open(OUTPUT_SUMMARY_CSV, 'w')
summary_file.write('target_fn, target_fp, skip_dd, skip_cnn, dd, dd_thres, cnn, cnn_upper_thres, cnn_lower_thres, accuracy, fn, fp, num_tp, num_tn, runtime\n')
summary_file.flush()

label_name = LABELS[yolo_label_num]
truth_csv = os.path.join(TRUTH_DIR_PREFIX, video_name+'.csv')

test_csv_filename =  "test_" + str(TEST_START_IDX) + "_" + str(TEST_START_IDX+TEST_LEN) + ".csv"
test_csv_str =  'test_${START_FRAME}_${END_FRAME}.csv'

# test_log_filename = "test_" + str(TEST_START_IDX) + "_" + str(TEST_START_IDX+TEST_LEN) + ".log"
test_log_str = 'test_${START_FRAME}_${END_FRAME}.log'

video_cache_id = uuid.uuid4()
for error_rate in TARGET_ERROR_RATES:        
    test_path = os.path.join(EXPERIMENTS_DIR_PREFIX, video_name, str(error_rate))
    try:
        os.mkdir(test_path)
    except:
        pass

    # find the best configuration (find the optimal params for all of them)
    params_list = []
    for pipeline_path, cnn_path, dd in pipeline_paths:
        params = opt.main(
            label_name, 
            truth_csv, os.path.join(pipeline_path, train_csv_filename),
            error_rate, error_rate,
            TRAIN_START_IDX, TRAIN_END_IDX
        )

        use_blocked = 0
        diff_detection_weights = '/dev/null'
        ref_image = 0
        if dd is not None:
            use_blocked = 1
            diff_detection_weights = os.path.join( DD_MEAN_DIR_PREFIX, video_name, dd[0]  )
            ref_image = dd[1]

        params['pipeline_path'] = pipeline_path
        params['cnn_path'] = cnn_path
        params['dd_path'] = diff_detection_weights
        params['dd_ref_index'] = ref_image
        params['use_blocked'] = use_blocked
        params_list.append(params)

    best_params = sorted(params_list, key=lambda x: x['optimizer_cost'])[0]
    # NOTE FIXME TODO
    # THIS IS A HACK
    best_params['threshold_skip_distance'] = 30
    
    video_cache_filename = os.path.join(
            VIDEO_CACHE_PREFIX,
            '%s_%d_%d_%d.bin' % (video_name, TEST_START_IDX, TEST_LEN,
                                 best_params['threshold_skip_distance']))
    
    # run the actual experiment
    test_csv_path_str = os.path.join(
        test_path, 
        test_csv_str
    )
    test_csv_path = os.path.join(
        test_path,
        test_csv_filename
    )

    test_log_path = os.path.join(
        test_path, 
        test_log_str
    )

    with open( os.path.join(test_path, 'params.json'), 'w') as f:
        f.write( json.dumps(best_params, sort_keys=True, indent=4) ) 

    
    run_script = os.path.join(test_path, RUN_SCRIPT_FILENAME)
    with open(run_script, 'w') as f:
        script = RUN_BASH_SCRIPT.format(
            date=str(datetime.datetime.now()),
            tf_prefix=TF_DIRPREFIX,
            dd_thres=best_params['threshold_diff'],
            cnn_lower_thres=best_params['threshold_lower_cnn'],
            cnn_upper_thres=best_params['threshold_upper_cnn'],
            skip_dd=int(best_params['skip_dd']),
            skip_cnn=int(best_params['skip_cnn']),
            kskip=best_params['threshold_skip_distance'],
            ref_image=int(best_params['dd_ref_index'] / best_params['threshold_skip_distance']),
            diff_detection_weights=best_params['dd_path'],
            use_blocked=best_params['use_blocked'],
            cnn_path=best_params['cnn_path'],
            cnn_avg_path=cnn_avg_path,
            video_path=video_path,
            yolo_class=yolo_label_num,
            start=TEST_START_IDX,
            length=TEST_LEN,
            output_csv=test_csv_path_str,
            output_log=test_log_path,
            dumped_videos=video_cache_filename
        )
    
        f.write(script)

    st = os.stat(run_script)
    os.chmod(run_script, st.st_mode | stat.S_IEXEC)
    print 'running experiment: {} {}'.format(error_rate, test_path)
    if( not os.path.exists(test_csv_path) or NO_CACHING ):
        print "GPU_NUM:", GPU_NUM
        os.system( 'bash ' + run_script + ' ' + str(GPU_NUM) ) 
    else:
        print 'WARNING: using cached results! Skipping computation.'

    # compute the actual accuracy
    accuracy = acc.main(label_name,
                        TEST_START_IDX, TEST_END_IDX,
                        truth_csv, test_csv_path)
    with open( os.path.join(test_path, 'accuracy.json'), 'w') as f:
        f.write( json.dumps(accuracy, sort_keys=True, indent=4) ) 

    summary_file.write('{target_fn}, {target_fp}, {skip_dd}, {skip_cnn}, {dd}, {dd_thres}, {cnn}, {cnn_upper_thres}, {cnn_lower_thres}, {acc}, {fn}, {fp}, {tp}, {tn}, {runtime}\n'.format(
        target_fn=error_rate,
        target_fp=error_rate,
        skip_dd=best_params['skip_dd'],
        skip_cnn=best_params['skip_cnn'],
        dd=best_params['dd_path'],
        dd_thres=best_params['threshold_diff'],
        cnn=best_params['cnn_path'],
        cnn_upper_thres=best_params['threshold_upper_cnn'],
        cnn_lower_thres=best_params['threshold_lower_cnn'],
        acc=accuracy['accuracy'],
        fn=accuracy['false_negative'],
        fp=accuracy['false_positive'],
        tp=accuracy['num_true_positives'],
        tn=accuracy['num_true_negatives'],
        runtime=accuracy['runtime']
    ))
    summary_file.flush()
        
summary_file.close()

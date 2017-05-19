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

time {tf_prefxix}/bazel-bin/tensorflow/noscope/noscope \\
    --diff_thresh={dd_thres} \\
    --distill_thresh_lower={cnn_lower_thres} \\
    --distill_thresh_upper={cnn_upper_thres} \\
    --skip_small_cnn={skip_cnn} \\
    --skip_diff_detection={skip_dd} \\
    --skip={kskip} \\
    --avg_fname={cnn_avg_path} \\
    --graph={cnn_path} \\
    --video={video_path} \\
    --yolo_cfg=/lfs/1/ddkang/noscope/darknet/cfg/yolo.cfg \\
    --yolo_weights=/lfs/1/ddkang/noscope/darknet/weights/yolo.weights \\
    --yolo_class={yolo_class} \\
    --confidence_csv={output_csv} \\
    --start_from=${{START_FRAME}} \\
    --nb_frames=$LEN \\
    --dumped_videos={dumped_videos} \\
    --diff_detection_weights={diff_detection_weights} \\
    --use_blocked={use_blocked} \\
    --ref_image={ref_image} \\
    &> {output_log}
"""

LABELS = dict()
LABELS[0] = "person"
LABELS[2] = "car"
LABELS[5] = "bus" 

YOLO_LABELS = dict()
# YOLO_LABELS["broadway-jackson-hole"] = (2, [("broadway-jackson-hole_mnist_128_64.pb", None), ("broadway-jackson-hole_mnist_128_64.pb", ("broadway-jackson-hole_raw_mse_lr_l2_delay10_resol50_ref-index59_num-blocks10.model", 60)), ("broadway-jackson-hole_mnist_128_16.pb", None), ("broadway-jackson-hole_mnist_128_16.pb", ("broadway-jackson-hole_raw_mse_lr_l2_delay10_resol50_ref-index59_num-blocks10.model", 60))])
YOLO_LABELS["buffalo-meat"] = (
        0,
        [("buffalo-meat_mnist_128_32.pb", None),
         #("buffalo-meat_mnist_128_32.pb", ("buffalo-meat_raw_mse_lr_l2_delay10_resol100_ref-index59_num-blocks10.model", 60)),
         ("buffalo-meat_mnist_256_16.pb", None),],
         #("buffalo-meat_mnist_256_16.pb", ("buffalo-meat_raw_mse_lr_l2_delay10_resol100_ref-index59_num-blocks10.model", 60))],
        150000,
        100000,
        250000 + 54000,
        559020,
)
YOLO_LABELS["buses-cars-archie-europe-rotonde"] = (
        2,
        [("buses-cars-archie-europe-rotonde_cifar10_32_2.pb", None),
         ("buses-cars-archie-europe-rotonde_cifar10_32_2.pb", ("buses-cars-archie-europe-rotonde_raw_mse_lr_l2_delay10_resol50_ref-index369_num-blocks10.model", 360)),
         ("buses-cars-archie-europe-rotonde_mnist_32_16.pb", None),
         ("buses-cars-archie-europe-rotonde_mnist_32_16.pb", ("buses-cars-archie-europe-rotonde_raw_mse_lr_l2_delay10_resol50_ref-index369_num-blocks10.model", 360))],
        150000,
        100000,
        250000 + 54000,
        731010,
)
# YOLO_LABELS["coral-reef"] = (0, [("coral-reef_cifar10_256_1.pb", None), ("coral-reef_cifar10_256_1.pb", ("coral-reef_raw_mse_lr_l2_delay10_resol100_ref-index508_num-blocks10.model", 510)), ("coral-reef_mnist_128_16.pb", None), ("coral-reef_mnist_128_16.pb", ("coral-reef_raw_mse_lr_l2_delay10_resol100_ref-index508_num-blocks10.model", 510))])
YOLO_LABELS["elevator"] = (
        0,
        [("elevator_mnist_256_32.pb", None),
         ("elevator_mnist_256_32.pb", ("elevator_raw_mse_lr_l2_delay10_resol100_ref-index59_num-blocks10.model", 60)),
         ("elevator_mnist_64_16.pb", None),
         ("elevator_mnist_64_16.pb", ("elevator_raw_mse_lr_l2_delay10_resol100_ref-index59_num-blocks10.model", 60))],
        150000,
        100000,
        250000 + 54000,
        592020,
)
# YOLO_LABELS["live-zicht-binnenhaven"] = (2, [("live-zicht-binnenhaven_mnist_256_64.pb", None), ("live-zicht-binnenhaven_mnist_256_64.pb", ("live-zicht-binnenhaven_raw_mse_lr_l2_delay10_resol50_ref-index141_num-blocks10.model", 150)), ("live-zicht-binnenhaven_mnist_128_16.pb", None), ("live-zicht-binnenhaven_mnist_128_16.pb", ("live-zicht-binnenhaven_raw_mse_lr_l2_delay10_resol50_ref-index141_num-blocks10.model", 150))])
YOLO_LABELS["shibuya-halloween"] = (
        2,
        [("shibuya-halloween_mnist_64_16.pb", None),
         ("shibuya-halloween_mnist_64_16.pb", ("shibuya-halloween_raw_mse_lr_l2_delay10_resol50_ref-index59_num-blocks10.model", 60)),
         ("shibuya-halloween_cifar10_128_1.pb", None),
         ("shibuya-halloween_cifar10_128_1.pb", ("shibuya-halloween_raw_mse_lr_l2_delay10_resol50_ref-index59_num-blocks10.model", 60))]
)
# YOLO_LABELS["taipei"] = (5, [("taipei_cifar10_256_1.pb", None), ("taipei_cifar10_256_1.pb", ("taipei_raw_mse_lr_l2_delay10_resol50_ref-index59_num-blocks10.model", 60)), ("taipei_mnist_32_64.pb", None), ("taipei_mnist_32_64.pb", ("taipei_raw_mse_lr_l2_delay10_resol50_ref-index59_num-blocks10.model", 60))])
YOLO_LABELS["taipei-long"] = (
        5,
        [("taipei-long_cifar10_128_2.pb", None),
         ("taipei-long_cifar10_256_0.pb", None)],
        409000,
        100000,
        1296000,
        1296000
)
YOLO_LABELS["live-zicht-long"] = (
        2,
        [("live-zicht-long_cifar10_32_0.pb", None),
         ("live-zicht-long_cifar10_32_2.pb", None),
         ("live-zicht-long_cifar10_64_1.pb", None),],
        409000,
        100000,
        1296000,
        1296000
)
YOLO_LABELS["jackson-crop2"] = (
        2,
        [("jackson-crop2_cifar10_32_1.pb", None),
         ("jackson-crop2_cifar10_32_0.pb", None),],
        409000,
        100000,
        918000,
        918000
)
YOLO_LABELS["coral-reef-long"] = (
        0,
        [("coral-reef-long_cifar10_32_1.pb", None),
         ("coral-reef-long_cifar10_32_2.pb", None),],
        648000 + 200000,
        100000,
        1188000 + 648000,
        1188000
)
YOLO_LABELS["whitewater"] = (
        0,
        [("whitewater_cifar10_32_1.pb", None),
         ("whitewater_cifar10_32_2.pb", None),],
        800000,
        100000,
        1080000,
        1080000,
)

NO_CACHING = False

# TARGET_ERROR_RATES = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25]
# TARGET_ERROR_RATES = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25]
TARGET_ERROR_RATES = [0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]

# TRAIN_START_IDX = 150000
# TRAIN_LEN = 100000
# TRAIN_END_IDX = TRAIN_START_IDX + TRAIN_LEN

# TEST_START_IDX = 250000
# TEST_LEN = 300000
# TEST_END_IDX = TEST_START_IDX + TEST_LEN

TF_DIRPREFIX = '/lfs/1/ddkang/noscope/tensorflow-noscope/'
DATA_DIR_PREFIX = '/lfs/1/ddkang/noscope/data/'
EXPERIMENTS_DIR_PREFIX = '/lfs/1/ddkang/noscope/experiments/'
VIDEO_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'videos')
# VIDEO_CACHE_PREFIX = os.path.join(DATA_DIR_PREFIX, 'video-cache')
VIDEO_CACHE_PREFIX = '/lfs/0/ddkang/noscope/data/video-cache/'
TRUTH_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'csv')

DD_MEAN_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'dd-means')
CNN_MODEL_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'cnn-models')
CNN_AVG_DIR_PREFIX = os.path.join(DATA_DIR_PREFIX, 'cnn-avg')

TRAIN_DIRNAME = 'train'

RUN_SCRIPT_FILENAME = 'run_testset.sh'
RUN_OPTIMIZER_SCRIPT_FILENAME = 'run_optimizerset.sh'

OUTPUT_SUMMARY_CSV = 'summary.csv'

GPU_NUM = '-1'

################################################################################
# Begin script
################################################################################
if len(sys.argv) != 3:
    print "USAGE:", sys.argv[0], "VIDEO GPU_NUM"
    print 

    print "Possible videos:"
    for v in sorted(YOLO_LABELS.keys()):
        print "\t", v
    print
    sys.exit(1)

video_name = sys.argv[1]
GPU_NUM = sys.argv[2]
if( video_name not in YOLO_LABELS.keys() ):
    print "video must be from the following list"
    print
    print "Possible videos:"
    for v in sorted(YOLO_LABELS.keys()):
        print "\t", v
    print
    sys.exit(1)

# FIXME
if 'long' in video_name or 'crop2' in video_name or \
        'whitewater' in video_name:
    CNN_MODEL_DIR_PREFIX = '/lfs/1/ddkang/noscope/model-search/'

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
            tf_prefxix=TF_DIRPREFIX,
            dd_thres=0,
            cnn_lower_thres=0,
            cnn_upper_thres=0,
            skip_dd=0,
            skip_cnn=0,
            kskip=1,
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
        os.system( 'bash ' + run_optimizer_script + ' ' + GPU_NUM ) 
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
    
    # video_cache_filename = os.path.join(VIDEO_CACHE_PREFIX, str(video_cache_id) + "_" + str(best_params['threshold_skip_distance']) + ".mp4.raw")
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
            tf_prefxix=TF_DIRPREFIX,
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
        os.system( 'bash ' + run_script + ' ' + GPU_NUM ) 
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

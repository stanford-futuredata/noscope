#!/usr/bin/env python

import argparse
import glob
import os
import subprocess
from pathos import multiprocessing

def fn(arg):
    i, OBJECT, dir_name, truth_csv_name, test_csv_name, output_csv, runner = arg
    if not os.path.isfile(truth_csv_name):
        print 'Can\'t find %s' % (truth_csv_name)

    cmd = ['python', runner,
           OBJECT,
           truth_csv_name,
           test_csv_name
          ]
    print 'Running ' + str(cmd)
    with open(output_csv + '_' + str(i), 'a') as f:
        process = subprocess.Popen(cmd, stdout=f)
        process.wait()
        if process.returncode == 1:
            print str(cmd) + ' failed'
    return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_csv', required=True, help='File for outputting results')
    args = parser.parse_args()

    to_run = {
            'aoerc': 'person',
            'broadway-jackson-hole': 'car',
            'buffalo-meat': 'person',
            'buses-cars-archie-europe-rotonde': 'car',
            'bytes': 'person',
            'coral-reef': 'person',
            'coupa': 'person',
            'elevator': 'person',
            'huang': 'person',
            'lady-in-the-corner': 'person',
            'live-zicht-binnenhaven': 'car',
            'shibuya-halloween': 'car',
            'taipei': 'bus',
            'town-square-shootout': 'person'
        }
    input_csv_base = '/dfs/scratch1/jemmons/labels-truncated/%s.mp4.csv.truncated'
    runner = 'gen_accuracy_plot.py'

    i = 0
    run_args = []
    for dir_name in to_run:
        for filename in glob.iglob('%s/*.csv' % dir_name):
            OBJECT = to_run[dir_name]
            truth_csv_name = input_csv_base % dir_name
            test_csv_name = filename
            run_args.append(
                    (i, OBJECT, dir_name, truth_csv_name,
                        test_csv_name, args.output_csv, runner)
            )
            i += 1

    pool = multiprocessing.Pool(48)
    results = pool.map(fn, run_args)
    print results
    print len(results)


if __name__ == '__main__':
    main()

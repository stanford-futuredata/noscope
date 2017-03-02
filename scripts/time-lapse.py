#! /usr/bin/env python

from glob import glob
from utils import mkdir_p, parallelize
from noscope.TimeLapse import create_time_lapse
import argparse
import os

DONE_DIR = 'time-lapse-done'
OUTPUT_DIR = 'time-lapse'
NEW_FILE_SUFFIX = '.time-lapse'

def time_lapse_single_file(frames_to_skip):
    def _time_lapse(movie_full_path):
        filename_only = movie_full_path[movie_full_path.rindex('/') + 1:]
        new_filename = filename_only.replace('.mp4', '%s.mp4' % NEW_FILE_SUFFIX)
        new_full_path = movie_full_path.replace(filename_only,
                '/' + OUTPUT_DIR + '/' + new_filename)
        create_time_lapse(movie_full_path, new_full_path, frames_to_skip)
        new_path_for_old_file = movie_full_path.replace(filename_only,
                '/' + DONE_DIR + '/' + filename_only)
        os.rename(movie_full_path, new_path_for_old_file)
    return _time_lapse

def time_lapse_all_in_dir(input_dir, frames_to_skip):
    raw_movie_files = glob('%s/*.mp4' % input_dir)
    mkdir_p('%s/%s' % (input_dir, DONE_DIR))
    mkdir_p('%s/%s' % (input_dir, OUTPUT_DIR))
    time_lapse_fn = time_lapse_single_file(frames_to_skip)
    parallelize(time_lapse_fn, raw_movie_files, num_pools=64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True,
            help='directory with mp4 files that need to time-lapsed')
    parser.add_argument('--frames_to_skip', type=int, default=10,
            help='Number of frames to skip in input video')
    args = parser.parse_args()

    time_lapse_all_in_dir(args.dir, args.frames_to_skip)

if __name__ == '__main__':
    main()


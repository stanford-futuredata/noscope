#! /usr/bin/env python

from glob import glob
from utils import execute_command, mkdir_p, parallelize
import argparse
import os

AVCONV_TEMPLATE = 'avconv -i %s -c:a copy -c:v libx264 -preset veryfast -tune fastdecode -crf 15 -crf_max 20 %s'
DONE_DIR = 'processed'
OUTPUT_DIR = 'fixed'
NEW_FILE_SUFFIX = '.fixed'

def reencode_single_file(movie_full_path):
    filename_only = movie_full_path[movie_full_path.rindex('/') + 1:]
    new_filename = filename_only.replace('.mp4', '%s.mp4' % NEW_FILE_SUFFIX)
    new_full_path = movie_full_path.replace(filename_only, '/' + OUTPUT_DIR + '/' + new_filename)
    execute_command(AVCONV_TEMPLATE % (movie_full_path, new_full_path))
    new_path_for_old_file = movie_full_path.replace(filename_only, '/' + \
            DONE_DIR + '/' + filename_only)
    os.rename(movie_full_path, new_path_for_old_file)

def reencode_dir(input_dir, num_pools):
    raw_movie_files = glob('%s/*.mp4' % input_dir)
    mkdir_p('%s/%s' % (input_dir, DONE_DIR))
    mkdir_p('%s/%s' % (input_dir, OUTPUT_DIR))
    parallelize(reencode_single_file, raw_movie_files, num_pools=num_pools)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True,
            help='directory with mp4 files that need to be reencoded')
    parser.add_argument('--num_pools', type=int, default=48,
            help='number of concurrent videos to reencode (default: 48)')
    args = parser.parse_args()

    reencode_dir(args.dir, args.num_pools)

if __name__ == '__main__':
    main()


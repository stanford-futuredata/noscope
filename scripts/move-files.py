#! /usr/bin/env python

from glob import glob
from utils import mkdir_p
from subprocess import check_call
import argparse

def move_files(input_dir, output_dir):
    raw_movie_files = glob('%s/*.*[0-9].mp4' % input_dir)
    # remove absolute path, '.mp4' from filename
    filenames_only = [full_path[full_path.rindex('/') + 1:-4] for full_path in raw_movie_files]
    filenames_only.sort(key=lambda x: (x[:x.rindex('.')], int(x[x.rindex('.') + 1:])), reverse=True)

    mkdir_p(output_dir)
    prev = ''
    for filename in filenames_only:
        filename_no_num = filename[:filename.rindex('.')]
        if filename_no_num != prev:
            prev = filename_no_num
            continue
        prev = filename_no_num
        filename_with_ext = filename + '.mp4'
        print 'copying %s...' % filename_with_ext
        exit_code = check_call('cp %s %s' % ('%s/%s' % (input_dir, filename_with_ext),
            '%s/%s' % (output_dir, filename_with_ext)), shell=True)
        if exit_code != 0:
            print 'error copying over %s; do not remove' % filename_with_ext
            continue
        else:
            print 'removing %s from %s' % (filename_with_ext, input_dir)
            exit_code = check_call('rm %s' % ('%s/%s' % (input_dir, filename_with_ext)), shell=True)
        if exit_code != 0:
            print 'error removing %s' % filename_with_ext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
            help='directory with mp4 files that have finished scraping. Use absolute paths.')
    parser.add_argument('--output_dir', required=True,
            help='directory to copy files to; do NOT confused with input_dir. Use absolute paths.')
    args = parser.parse_args()

    if args.input_dir[0] != '/':
        print '--input_dir argument must be an absolute path'
        import sys; sys.exit(1)
    elif args.output_dir[0] != '/':
        print '--output_dir argument must be an absolute path'
        import sys; sys.exit(1)

    move_files(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()


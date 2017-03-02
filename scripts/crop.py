#! /usr/bin/env python

from noscope.CommonUtils import execute_command, replace_filename_suffix
import argparse

# -vf "crop=out_w:out_h:x:y"'
CROP_TEMPLATE = 'avconv -i %s -vf "crop=%d:%d:%d:%d" %s'
NEW_FILE_SUFFIX = '.cropped'

def crop_video(movie_full_path, x, y, width, height):
    new_full_path = replace_filename_suffix(movie_full_path, NEW_FILE_SUFFIX)
    execute_command(CROP_TEMPLATE % (movie_full_path, width, height, x, y, new_full_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in', required=True,
            help='mp4 video to crop')
    parser.add_argument('--x', type=int, required=True,
            help='X coordinate')
    parser.add_argument('--y', type=int, required=True,
            help='Y coordinate')
    parser.add_argument('--width', type=int, required=True,
            help='width')
    parser.add_argument('--height', type=int, required=True,
            help='height')
    args = parser.parse_args()

    crop_video(args.video_in, args.x, args.y, args.width, args.height)

if __name__ == '__main__':
    main()


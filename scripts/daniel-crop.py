
import argparse
import os
import glob
import subprocess

from pathos import multiprocessing

def fn(args):
    crop_args, vid_regex, vid_out_dir, vid_in_fname = args

    vid_in_path, base_fname = os.path.split(vid_in_fname)
    base, ext = os.path.splitext(base_fname)

    vid_out_fname = os.path.join(vid_out_dir, base_fname)
    cmd = ['./ffmpeg.linux64',
           '-i', vid_in_fname,
           '-filter:v', '"crop=%s"' % (crop_args),
           vid_out_fname]
    cmd = './ffmpeg.linux64 -i ' + vid_in_fname + \
        ' -filter:v "crop=' + crop_args + '" ' + \
        vid_out_fname

    print 'Running ' + str(cmd)
    process = subprocess.Popen(cmd, universal_newlines=True, shell=True)
    process.wait()
    if process.returncode == 1:
        print 'Something failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_regex', required=True, help='Video files')
    parser.add_argument('--vid_out_dir', required=True, help='Output directory for videos')
    parser.add_argument('--crop_args', required=True, help='Passed to ffmpeg')
    args = parser.parse_args()

    vid_regex = args.vid_regex
    vid_out_dir = args.vid_out_dir
    crop_args = args.crop_args

    run_args = []
    for vid_in_fname in glob.glob(vid_regex):
        run_args.append( (crop_args, vid_regex, vid_out_dir, vid_in_fname) )

    pool = multiprocessing.Pool(20)
    results = pool.map(fn, run_args)
    print results


if __name__ == '__main__':
    main()

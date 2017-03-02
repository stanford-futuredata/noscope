import os
import subprocess
from pathos import multiprocessing


def fn(arg):
    RESOL, DELAY, OBJECT, BASE_NAME, csv_in_name, vid_in_name, output_dir, runner = arg
    if not os.path.isfile(vid_in_name):
        print 'Can\'t find %s' % (vid_in_name)
    if not os.path.isfile(csv_in_name):
        print 'Can\'t find %s' % (csv_in_name)

    cmd = ['python', runner,
           '--csv_in', csv_in_name,
           '--video_in', vid_in_name,
           '--output_dir', output_dir,
           '--base_name', BASE_NAME,
           '--objects', OBJECT,
           '--num_frames', str(250000),
           '--resol', str(RESOL),
           '--delay', str(DELAY)]
    print 'Running ' + str(cmd)
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print 'Something failed'
    return process.returncode


'''def fn(arg):
    RESOL, DELAY, input_csv_base, input_vid_base, output_dir, to_run, runner = arg

    for fname in to_run:
        OBJECT = to_run[fname]
        csv_in_name = input_csv_base % fname
        vid_in_name = input_vid_base % fname

        if not os.path.isfile(vid_in_name):
            print 'Can\'t find %s' % (vid_in_name)
        if not os.path.isfile(csv_in_name):
            print 'Can\'t find %s' % (csv_in_name)

        cmd = ['python', runner,
               '--csv_in', csv_in_name,
               '--video_in', vid_in_name,
               '--output_dir', output_dir,
               '--base_name', fname,
               '--objects', OBJECT,
               '--num_frames', str(250),
               '--resol', str(RESOL),
               '--delay', str(DELAY)]
    print 'Running ' + str(cmd)
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print 'Something failed'
    return process.returncode'''


def main():
    to_run = {'aoerc': 'person',
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
              'taipei': 'bus'}
    input_csv_base = '/dfs/scratch1/fabuzaid/noscope-datasets-completed/csv/%s.mp4.csv'
    input_vid_base = '/dfs/scratch1/fabuzaid/noscope-datasets-completed/videos/%s.mp4'
    output_dir = '/dfs/scratch1/fabuzaid/noscope-experiments/hog/metrics_window30_all'
    runner = 'hog.py'

    run_args = []
    #for RESOL in [50, 100, 200, 400]:
    for RESOL in [50, 100]:
        for DELAY in [10, 30]:
            for fname in to_run:
                OBJECT = to_run[fname]
                csv_in_name = input_csv_base % fname
                vid_in_name = input_vid_base % fname
                run_args.append(
                        (RESOL, DELAY, OBJECT, fname,
                         csv_in_name, vid_in_name,
                         output_dir,
                         runner))

    pool = multiprocessing.Pool(16)
    results = pool.map(fn, run_args)
    print results
    print len(results)


if __name__ == '__main__':
    main()

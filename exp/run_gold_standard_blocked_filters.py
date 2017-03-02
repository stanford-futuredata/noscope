import os
import subprocess
from pathos import multiprocessing


def fn(arg):
    RESOL, DELAY, NUM_BLOCKS, REF_INDEX, REG, MODEL, METRIC, OBJECT, BASE_NAME, csv_in_name, vid_in_name, output_dir, runner = arg
    if not os.path.isfile(vid_in_name):
        print 'Can\'t find %s' % (vid_in_name)
    if not os.path.isfile(csv_in_name):
        print 'Can\'t find %s' % (csv_in_name)

    cmd = [
           'python', runner,
           '--csv_in', csv_in_name,
           '--video_in', vid_in_name,
           '--output_dir', output_dir,
           '--base_name', BASE_NAME,
           '--objects', OBJECT,
           '--filters', 'raw',
           '--model', MODEL,
           '--metric', METRIC,
           '--reg', REG,
           '--num_frames', str(250000),
           '--resol', str(RESOL),
           '--delay', str(DELAY),
           '--num_blocks', str(NUM_BLOCKS),
           '--ref_index', str(REF_INDEX)
          ]
    print 'Running ' + str(cmd)
    process = subprocess.Popen(cmd)
    process.wait()
    if process.returncode == 1:
        print str(cmd) + ' failed'
    return process.returncode


def main():
    to_run = {
            # video, object, metric, model, reg, delay, resolution, ref-index, num-blocks
            'aoerc': ('person', 'mse', 'lr', 'l2', 10, 100, -1, 10),
            'broadway-jackson-hole': ('car', 'mse', 'lr', 'l2', 10, 50, 100, 10),
            'buffalo-meat': ('person', 'mse', 'lr', 'l2', 10, 100, 100, 10),
            'buses-cars-archie-europe-rotonde': ('car', 'mse', 'lr', 'l2', 10, 50, 350, 10),
            'bytes': ('person', 'mse', 'lr', 'l2', 30, 50, -1, 10),
            'coral-reef': ('person', 'mse', 'lr', 'l2', 10, 100, 500, 10),
            'coupa': ('person', 'mse', 'lr', 'l2', 10, 50, 100, 10),
            'elevator': ('person', 'mse', 'lr', 'l2', 10, 100, 100, 10),
            'huang': ('person', 'mse', 'lr', 'l2', 10, 50, 110, 10),
            'lady-in-the-corner': ('person', 'mse', 'lr', 'l2', 10, 100, -1, 10),
            'live-zicht-binnenhaven': ('car', 'mse', 'lr', 'l2', 10, 50, 100, 10),
            'shibuya-halloween': ('car', 'mse', 'lr', 'l2', 10, 50, 50, 10),
            'taipei': ('bus', 'mse', 'lr', 'l2', 10, 50, 50, 10),
            'town-square-shootout': ('bus', 'mse', 'lr', 'l2', 10, 50, 50, 10)
        }
    input_csv_base = '/dfs/scratch1/fabuzaid/noscope-datasets-completed/csv/%s.mp4.csv'
    input_vid_base = '/dfs/scratch1/fabuzaid/noscope-datasets-completed/videos/%s.mp4'
    output_dir = '/dfs/scratch1/fabuzaid/noscope-experiments/blocked_filters-gold-standard-no-mean/'
    runner = 'blocked_filters.py'

    run_args = []
    for fname in to_run:
        OBJECT, METRIC, MODEL, REG, DELAY, RESOL, REF_INDEX, NUM_BLOCKS = to_run[fname]
        csv_in_name = input_csv_base % fname
        vid_in_name = input_vid_base % fname
        # no ref-index
        run_args.append(
                (RESOL, DELAY, NUM_BLOCKS, REF_INDEX, REG,
                 MODEL, METRIC, OBJECT, fname, csv_in_name,
                 vid_in_name, output_dir + fname, runner)
        )
        # with ref-index

    pool = multiprocessing.Pool(13)
    results = pool.map(fn, run_args)
    print results
    print len(results)


if __name__ == '__main__':
    main()

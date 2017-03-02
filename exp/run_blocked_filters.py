import os
import subprocess
from pathos import multiprocessing


def fn(arg):
    RESOL, DELAY, NUM_BLOCKS, REF_INDEX, REG, MODEL, OBJECT, BASE_NAME, csv_in_name, vid_in_name, output_dir, runner = arg
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
            'aoerc': ('person', 100),
            'broadway-jackson-hole': ('car', 100),
            'buffalo-meat': ('person', 100),
            'buses-cars-archie-europe-rotonde': ('car', 350),
            'bytes': ('person', 200),
            'coral-reef': ('person', 500),
            'coupa': ('person', 100),
            'elevator': ('person', 100),
            'huang': ('person', 110),
            'lady-in-the-corner': ('person', 34945),
            'live-zicht-binnenhaven': ('car', 100),
            'shibuya-halloween': ('car', 50),
            'taipei': ('bus', 50),
            'town-square-shootout': ('person', 50)
        }
    input_csv_base = '/dfs/scratch1/fabuzaid/noscope-datasets-completed/csv/%s.mp4.csv'
    input_vid_base = '/dfs/scratch1/fabuzaid/noscope-datasets-completed/videos/%s.mp4'
    output_dir = '/dfs/scratch1/fabuzaid/noscope-experiments/blocked_filters_stacked_features_reg/'
    runner = 'blocked_filters.py'

    run_args = []
    for NUM_BLOCKS in [5, 10]:
        for RESOL in [50, 100]:
            for DELAY in [10, 30]:
                for REG in ['l1', 'l2']:
                    for fname in to_run:
                        OBJECT, REF_INDEX = to_run[fname]
                        csv_in_name = input_csv_base % fname
                        vid_in_name = input_vid_base % fname
                        # no ref-index
                        run_args.append(
                                (RESOL, DELAY, NUM_BLOCKS, -1, REG,
                                 'lr', OBJECT, fname, csv_in_name, vid_in_name,
                                 output_dir + fname, runner)
                        )
                        # with ref-index
                        run_args.append(
                                (RESOL, DELAY, NUM_BLOCKS, REF_INDEX, REG,
                                 'lr', OBJECT, fname, csv_in_name, vid_in_name,
                                 output_dir + fname, runner)
                        )

    pool = multiprocessing.Pool(18)
    results = pool.map(fn, run_args)
    print results
    print len(results)


if __name__ == '__main__':
    main()

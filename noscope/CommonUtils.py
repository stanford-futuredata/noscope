from subprocess import check_output
from pathos import multiprocessing
import os
import time

NUM_POOLS=48

def mkdir_p(dir):
  if not os.path.exists(dir):
      os.makedirs(dir)

def execute_command(command):
    return check_output(command, shell=True)

def parallelize(fn, list_args, num_pools=NUM_POOLS):
    pool = multiprocessing.ProcessPool(num_pools)
    results = pool.amap(fn, list_args)
    while not results.ready():
        time.sleep(5)
    pool.close()

# Assumes that the file is always a *.mp4 file
def replace_filename_suffix(orig_full_path, new_suffix):
    filename_only = orig_full_path[orig_full_path.rindex('/') + 1:]
    new_filename = filename_only.replace('.mp4', '%s.mp4' % new_suffix)
    return orig_full_path.replace(filename_only, new_filename)


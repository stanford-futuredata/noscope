import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--npy_fname', required=True)
parser.add_argument('--txt_fname', required=True)
args = parser.parse_args()

avg = np.load(args.npy_fname)
print avg.shape
with open(args.txt_fname, 'wb') as f:
    for x in avg.ravel():
        f.write(str(x))
        f.write('\n')


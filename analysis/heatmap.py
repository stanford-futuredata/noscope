import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import itertools
import cv2
import numpy as np
import pandas as pd

def get_labels(csv_fname, limit=None, interval=1, start=0, label='bus'):
    df = pd.read_csv(csv_fname)
    df = df[df['frame'] >= start]
    df = df[df['frame'] < start + limit]
    df['frame'] -= start
    df = df[df['object_name'] == label]
    return df

def draw_rect(arr, label, resol):
    r = np.zeros(resol)
    h, w = resol
    c1 = (int(label.xmin * w), int(label.ymin * h))
    c2 = (int(label.xmax * w), int(label.ymax * h))
    '''print label
    print c1, c2
    print h, w
    print arr.shape
    print
    # r[c1[0]:c2[0], c1[1]:c2[1]] = 1'''
    r[c1[1]:c2[1], c1[0]:c2[0]] = 1
    arr += r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True, help='CSVs with YOLO labels')
    parser.add_argument('--objects', default='bus', help='Object to draw')
    parser.add_argument('--range1', required=True)
    parser.add_argument('--range2', required=True)
    parser.add_argument('--resol_xy', required=True)
    args = parser.parse_args()

    objects = args.objects.split(',')
    assert len(objects) == 1
    s1, e1 = map(int, args.range1.split(','))
    s2, e2 = map(int, args.range2.split(','))
    x, y = map(int, args.resol_xy.split(','))

    df1 = get_labels(args.input_csv, start=s1, limit=e1 - s1, label=objects[0])
    df2 = get_labels(args.input_csv, start=s2, limit=e2 - s2, label=objects[0])

    resol = (y, x)
    h1 = np.zeros(resol)
    h2 = np.zeros(resol)
    map(lambda label: draw_rect(h1, label, resol), df1.itertuples())
    map(lambda label: draw_rect(h2, label, resol), df2.itertuples())

    def plot(h, fname):
        plt.clf()
        plt.imshow(h, cmap='hot', interpolation='nearest')
        plt.savefig(fname)

    plot(h1, 'heatmap_train.pdf')
    plot(h2, 'heatmap_test.pdf')

if __name__ == '__main__':
    main()

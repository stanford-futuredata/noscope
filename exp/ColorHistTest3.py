import noscope
import cv2
import itertools
import argparse
import os
import time
import numpy as np

DELAY = 10


def label_fn(a, b):
    if a == 1 and b == 0:
        return 1
    #if a == 0 and b == 1:
    #    return 1
    if a == 0 and b == 0:
        return 0
    return None


def jointly_filter(a, b):
    assert len(a) == len(b)

    c, d = [], []
    for i in xrange(len(a)):
        if a[i] != None:
            c.append(a[i])
            d.append(b[i])
    return c, d


def VidHistLabelIter(video_fname, labels, scale=None):
    it1 = noscope.VideoUtils.VideoHistIterator(video_fname, scale)
    it2 = iter(labels)
    while True:
        frame_ind, frame, hist = it1.next()
        label = it2.next()
        yield frame_ind, frame, hist, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_in_fname', required=True, help='Video input')
    parser.add_argument('--csv_in_fname', required=True, help='CSV with labels')
    parser.add_argument('--num_frames', required=True, type=int,
                        help='Number of frames to consider. Half test, half train')
    parser.add_argument('--scale', type=float, help='Downscale the image to 1/scale')
    args = parser.parse_args()


    nb_frames = args.num_frames
    vid_in_fname = args.vid_in_fname
    csv_in_fname = args.csv_in_fname
    if args.scale != None:
        scale = 1.0 / args.scale
    else:
        scale = None

    counts = noscope.DataUtils.get_binary(csv_in_fname, limit=nb_frames)
    it = VidHistLabelIter(vid_in_fname, counts, scale=scale)
    all_data = list(itertools.islice(it, nb_frames))
    print len(all_data[0])
    print all_data[0][-1]

    def chisq(a, b):
        ha = noscope.filters.ColorHistogram.compute_histogram(a)
        hb = noscope.filters.ColorHistogram.compute_histogram(b)
        return noscope.filters.ColorHistogram.compute_chisq(ha, hb)
    def mse(a, b):
        return np.sum((a - b) ** 2) / float(a.size)
    from skimage.measure import compare_ssim, compare_nrmse, compare_psnr
    def ssim(a, b):
        return compare_ssim(a, b, multichannel=True)
    def nrmse_euc(a, b):
        return compare_nrmse(a, b, norm_type='Euclidean')
    def nrmse_minmax(a, b):
        return compare_nrmse(a, b, norm_type='min-max')
    def nrmse_mean(a, b):
        return compare_nrmse(a, b, norm_type='mean')
    def psnr(a, b):
        return -1 * compare_psnr(a, b)
    def chisq_3d(a, b):
        ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        return noscope.ColorHistogram.compute_chisq(ha, hb)

    metrics = [(mse, 'mse'),
               (chisq, 'chisq'),
               (psnr, 'psnr'),
               (nrmse_euc, 'nrmse_euc'),
               (nrmse_minmax, 'nrmse_minmax'),
               (nrmse_mean, 'nrmse_mean'),
               (chisq_3d, 'chisq_3d')]

    print all_data[0][1].shape
    Y_true = map(lambda i: label_fn(all_data[i][-1][0], all_data[i - DELAY][-1][0]),
                 xrange(DELAY, len(all_data)))
    Y_true, indices = jointly_filter(Y_true, range(DELAY, len(all_data)))
    for metric, metric_name in metrics:
        print 'Running ' + metric_name
        begin = time.time()
        #Y_prob = map(lambda i: noscope.ColorHistogram.compute_chisq(all_data[i][2], all_data[i - DELAY][2]),
        #             xrange(DELAY, len(all_data)))
        Y_prob = map(lambda i: metric(all_data[i][1], all_data[i - DELAY][1]),
                     indices)

        prefix, _ = os.path.splitext(os.path.split(vid_in_fname)[1])
        prefix += '.' + metric_name
        print noscope.StatsUtils.plot_auc_pr(
            Y_true, Y_prob,
            roc_fname=prefix + '.auc.png',
            pr_fname=prefix + '.pr.png',
            fnr_fpr_fname=prefix + '.fnr_fpr.png')
        end = time.time()
        print end - begin


if __name__ == '__main__':
    main()
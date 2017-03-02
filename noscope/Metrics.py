
import cv2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse


is_version3 = cv2.__version__[0] == '3'
class Metrics(dict):
    @staticmethod
    def compute_chisq(hist1, hist2):
        if is_version3:
            return cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CHISQR)
        else:
            return cv2.compareHist(hist1, hist2, method=cv2.cv.CV_COMP_CHISQR)

    @staticmethod
    def chisq(a, b):
        def get_hist(frame, nb_bins=32):
            hist = np.concatenate(map(
                    lambda i: cv2.calcHist([frame], [i], None, [nb_bins], [-1, 1]),
                    xrange(3)))
            hist = cv2.normalize(hist, hist)
            return hist
        ha = get_hist(a)
        hb = get_hist(b)
        return Metrics.compute_chisq(ha, hb)

    @staticmethod
    def absdiff(a, b):
        return np.sum(np.abs(a - b))

    @staticmethod
    def mse(a, b):
        return np.sum((a - b) ** 2) / float(a.size)

    @staticmethod
    def ssim(a, b):
        return compare_ssim(a, b, multichannel=True)

    @staticmethod
    def nrmse_euc(a, b):
        return compare_nrmse(a, b, norm_type='Euclidean')

    @staticmethod
    def nrmse_minmax(a, b):
        return compare_nrmse(a, b, norm_type='min-max')

    @staticmethod
    def nrmse_mean(a, b):
        return compare_nrmse(a, b, norm_type='mean')

    @staticmethod
    def psnr(a, b):
        psnr_val = -1 * compare_psnr(a, b)
        if np.isnan(psnr_val):
            psnr_val = 10
        if np.isinf(psnr_val):
            psnr_val = -10
        return psnr_val

    @staticmethod
    def chisq_3d(a, b):
        def get_hist(frame):
            tmp = frame
            hist = cv2.calcHist([tmp], [0, 1, 2], None, [8, 8, 8], [-1, 1] * 3)
            hist = hist.flatten()
            hist = cv2.normalize(hist, hist)
            return hist
        ha = get_hist(a)
        hb = get_hist(b)
        return Metrics.compute_chisq(ha, hb)

    def __init__(self, *args):
        dict.__init__(self, args)

        self.metrics = \
            [(self.absdiff, 'absdiff'),
             (self.mse, 'mse'),
             (self.chisq, 'chisq'),
             (self.psnr, 'psnr'),
             (self.nrmse_euc, 'nrmse-euc'),
             (self.nrmse_minmax, 'nrmse-minmax'),
             (self.nrmse_mean, 'nrmse-mean'),
             (self.chisq_3d, 'chisq-3d')]

        for metric, name in self.metrics:
            self[name] = metric


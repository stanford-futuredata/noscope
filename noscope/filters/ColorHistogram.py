import cv2
import numpy as np

is_version3 = cv2.__version__[0] == '3'

DIST_METRICS = ['chisqr']

def _compute_histogram(frame, nb_bins=32):
    # hist = cv2.calcHist(frame, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # hist = cv2.normalize(hist, hist).flatten()
    nb_channels = frame.shape[-1]
    # Needs to be the same type as the return of calcHist
    hist = np.zeros( (nb_bins * nb_channels, 1), dtype='float32' )
    for i in xrange(nb_channels):
        hist[i * nb_bins : (i + 1) * nb_bins] = \
            cv2.calcHist(frame, [i], None, [nb_bins], [0, 256])
    hist = cv2.normalize(hist, hist)
    return hist

def compute_feature(frame):
    return cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

def get_distance_fn(dist_metric):
    if dist_metric == 'chisqr':
        return compute_chisq
    else:
        import sys
        print 'Invalid distance metric: %s' % dist_metric
        sys.exit(1)

def compute_chisq(hist1, hist2):
    if is_version3:
        return cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CHISQR)
    else:
        return cv2.compareHist(hist1, hist2, method=cv2.cv.CV_COMP_CHISQR)


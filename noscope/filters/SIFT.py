
import cv2
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine


DIST_METRICS = [
        ('euclidean', euclidean),
        ('manhattan', cityblock),
        ('chebyshev', chebyshev),
        ('cosine', lambda x, y: -1*cosine(x, y)),
        #('chisqr', lambda x, y: cv2.compareHist(x, y, cv2.HISTCMP_CHISQR)),
        #('bhatta', lambda x, y: cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA))
]

def compute_feature(frame):
    sift = cv2.xfeatures2d.SIFT_create()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, des = sift.detectAndCompute(image, None)
    if des is not None:
        return np.mean(des, axis=0).astype('float32')
    else:
        return np.zeros(128)


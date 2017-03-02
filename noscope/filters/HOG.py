
from skimage.feature import hog
from skimage import color
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine


DIST_METRICS = [
        ('euclidean', euclidean),
        ('manhattan', cityblock),
        ('chebyshev', chebyshev),
        ('cosine', cosine),
        #('chisqr', lambda x, y: cv2.compareHist(x, y, cv2.HISTCMP_CHISQR)),
        #('bhatta', lambda x, y: cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA))
]

def compute_feature(frame, orientations=10):
    image = color.rgb2gray(frame)
    return hog(image, orientations=orientations, pixels_per_cell=(10, 10),
            cells_per_block=(2, 2)).astype('float32')


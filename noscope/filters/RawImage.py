
from skimage.measure import compare_ssim, compare_nrmse, compare_psnr
import numpy as np

DIST_METRICS = ['mse', 'ssim', 'nrmse_euc', 'nrmse_minmax', 'nrmse_mean', 'psnr']

# nop, these metrics operate on the images themselves
def compute_feature(frame):
    return frame

def get_distance_fn(dist_metric):
    if dist_metric == 'mse':
        return lambda a, b: np.sum((a - b) ** 2) / float(a.size)
    if dist_metric == 'ssim':
        return lambda a, b: compare_ssim(a, b, multichannel=True)
    if dist_metric == 'nrmse_euc':
        return lambda a, b: compare_nrmse(a, b, norm_type='Euclidean')
    if dist_metric == 'nrmse_minmax':
        return lambda a, b: compare_nrmse(a, b, norm_type='min-max')
    if dist_metric == 'nrmse_mean':
        return lambda a, b: compare_nrmse(a, b, norm_type='mean')
    if dist_metric == 'psnr':
        return lambda a, b: -1 * compare_psnr(a, b)



from BlockedFilter import BlockedFilter
from  noscope.Metrics import Metrics
import numpy as np

class BlockedRawImage(BlockedFilter):
    def __init__(self, frame_shape, num_blocks=10):
        super(BlockedRawImage, self).__init__(frame_shape, num_blocks)
        self.mse_cached = self.cache_dist_metric(Metrics.mse)
        self.psnr_cached = self.cache_dist_metric(Metrics.psnr)
        self.nrmse_euc_cached = self.cache_dist_metric(Metrics.nrmse_euc)
        self.nrmse_minmax_cached = self.cache_dist_metric(Metrics.nrmse_minmax)
        self.nrmse_mean_cached = self.cache_dist_metric(Metrics.nrmse_mean)

    def compute_block(self, frame):
        # Compute distance metric on raw image
        return frame

    def stacked(self, frame_num, block_num, a, b):
        return np.array([fn_cached(frame_num, block_num, a, b) for fn_cached in [self.mse_cached, self.psnr_cached,
            self.nrmse_euc_cached, self.nrmse_minmax_cached, self.nrmse_mean_cached]], dtype='float32')

    def cache_dist_metric(self, metric_fn):
        cache = {}
        def fn(i, j, a, b):
            if (i, j) not in cache:
                cache[(i, j)] = metric_fn(a, b)
            return cache[(i, j)]
        return fn

    def distance_metrics(self):
        return [
            #('stacked', self.stacked),
            ('mse', self.mse_cached),
            #('psnr', self.psnr_cached),
            #('nrmse-euc', self.nrmse_euc_cached),
            #('nrmse-minmax', self.nrmse_minmax_cached),
            #('nrmse-mean', self.nrmse_mean_cached),
        ]



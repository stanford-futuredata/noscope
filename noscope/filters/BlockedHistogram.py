
import numpy as np
from BlockedFilter import BlockedFilter
from noscope.Metrics import Metrics

class BlockedHistogram(BlockedFilter):
    def __init__(self, frame_shape, num_blocks=10, num_bins=20):
        super(BlockedHistogram, self).__init__(frame_shape, num_blocks)
        self.hist_bins = np.linspace(0, 1, num_bins+1)

    def compute_block(self, frame):
        tmp, _ = np.histogram(frame, self.hist_bins)
        return tmp

    def distance_metrics(self):
        return [
            ('chisq', Metrics.compute_chisq),
        ]


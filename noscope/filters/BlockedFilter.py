
import numpy as np

class BlockedFilter(object):
    def __init__(self, frame_shape, num_blocks=10):
        self.H, self.W, self.C = frame_shape
        assert self.H % num_blocks == 0
        assert self.W % num_blocks == 0
        self.height = self.H / num_blocks
        self.width = self.W / num_blocks

    def compute_block(self, frame):
        raise NotImplementedError('subclasses of BlockedFilter must implement a \
                "compute_block" method')

    def compute_feature(self, frame):
        res = []
        for c in xrange(self.C):
            for h in xrange(0, self.H, self.height):
                for w in xrange(0, self.W, self.width):
                    tmp = self.compute_block(frame[h:h+self.height, w:w+self.width, c])
                    res.append(tmp)
        return np.array(res).astype('float32')

    def compute_blocked_distances(self, frame_num, dist_fn, block1, block2, use_max=True):
        assert block1.shape == block2.shape
        res = []
        for block_num in xrange(block1.shape[0]):
            tmp = dist_fn(frame_num, block_num, block1[block_num, :],
                    block2[block_num, :])
            res.append(tmp)
        if use_max:
            return np.max(res)
        else:
            return np.array(res).flatten()

    '''
    return list of tuples: [(str, dist_fn),...]
    '''
    def distance_metrics(self):
        raise NotImplementedError('subclasses of BlockedFilter must implement a \
                "distance_metrics" method')


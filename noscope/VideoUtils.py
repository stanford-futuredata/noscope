import cv2
import numpy as np
from math import ceil

# Pass in None for scale to skip resizing
# FIXME: Change the name of scale
def VideoIterator(video_fname, scale=None, interval=1, start=0):
    cap = cv2.VideoCapture(video_fname)
    # Seeks to the Nth frame. The next read is the N+1th frame
    # In OpenCV 2.4, this is cv2.cv.CAP_PROP_POS_FRAMES (I think)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    frame = 0
    frame_ind = -1
    if scale is not None:
        try:
            len(scale)
            resol = scale
            scale = None
        except:
            resol = None
    while frame is not None:
        frame_ind += 1
        _, frame = cap.read()
        if frame_ind % interval != 0:
            continue
        if scale is not None:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        elif resol is not None:
            frame = cv2.resize(frame, resol, interpolation=cv2.INTER_NEAREST)
        yield frame_ind, frame

def VideoHistIterator(video_fname, scale=None, start=0):
    from noscope.filters import ColorHistogram
    vid_it = VideoIterator(video_fname, scale=scale, start=start)
    frame = 0
    while frame is not None:
        frame_ind, frame = vid_it.next()
        hist = ColorHistogram.compute_histogram(frame)
        yield frame_ind, frame, hist

def get_all_frames(num_frames, video_fname, scale=None, interval=1, start=0, dtype='float32'):
    if video_fname[-4:] == '.bin':
        RESOL = (50, 50) # FIXME
        FRAME_SIZE = RESOL[0] * RESOL[0] * 3
        f = open(video_fname, 'rb')
        f.seek(start * FRAME_SIZE)
        frames = np.fromfile(f, dtype='uint8', count=num_frames * FRAME_SIZE)
        frames = frames.reshape((num_frames, RESOL[0], RESOL[1], 3))
        return frames.astype('float32') / 255.

    true_num_frames = int(ceil((num_frames + 0.0) / interval))
    print '%d total frames / %d frame interval = %d actual frames' % (num_frames, interval, true_num_frames)
    vid_it = VideoIterator(video_fname, scale=scale, interval=interval, start=start)

    _, frame = vid_it.next()
    frames = np.zeros( tuple([true_num_frames] + list(frame.shape)), dtype=dtype )
    frames[0, :] = frame

    for i in xrange(1, true_num_frames):
        _, frame = vid_it.next()
        frames[i, :] = frame

    if dtype == 'float32':
        frames /= 255.0

    return frames


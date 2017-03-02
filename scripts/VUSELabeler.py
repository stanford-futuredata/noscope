import noscope
import cv2
import csv
import argparse
import time
import numpy as np

# Example use:
# python Labeler.py --yolo_dir ".." \
#   --output_csv_fname ~/tmp/del.csv --input_video_fname \
#   ~/tmp/video.mp4 --start_from 1

class YOLOLabeler():
    def __init__(self, config_filename, weights_filename, data_config):
        self.YOLO = noscope.YOLO(config_filename, weights_filename, data_config)

    def label_frame(self, image):
        return self.YOLO.label_frame(image)

def time_labeling(vid_fname, csv_fname, yolo_dir='.', start_from=0):
    YOLO_labeler = YOLOLabeler(
            yolo_dir + '/cfg/yolo.cfg',
            yolo_dir + '/weights/yolo.weights',
            yolo_dir + '/cfg/coco.data')

    all_frames = noscope.VideoUtils.get_all_frames(3000, vid_fname, scale=None)

    begin = time.time()
    labels = map(lambda frame: YOLO_labeler.label_frame(frame), all_frames)
    end = time.time()
    print 'time to label: %f' % (end - begin)


def label_video(vid_fname, csv_fname, yolo_dir='.', start_from=0):
    YOLO_labeler = YOLOLabeler(
            yolo_dir + '/cfg/yolo.cfg',
            yolo_dir + '/weights/yolo.weights',
            yolo_dir + '/cfg/coco.data')

    begin = time.time()
    with open(csv_fname, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['frame', 'labels'])
        count = 0
        cap = cv2.VideoCapture(vid_fname)
        while cap.isOpened():
            count += 1
            ret, frame = cap.read()
            if frame is None:
                break
            if count < start_from:
                continue

            detections = YOLO_labeler.label_frame(frame)
            if detections is None:
                detections = []
            writer.writerow([count, detections])

            if count % 500 == 0:
                print count, detections
                csv_file.flush()

    print 'finished processing', count, 'frames'
    print 'time: ' + str(time.time() - begin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_dir', help='YOLO directory with cfg/, weights/, data/labels/')
    parser.add_argument('--output_csv_fname', help='Output CSV filename')
    parser.add_argument('--input_video_fname', help='Input video filename')
    parser.add_argument('--start_from', type=int, help='Which frame to start from')
    args = parser.parse_args()
    if args.yolo_dir is None:
        args.yolo_dir = '.'
    if args.start_from is None:
        args.start_from = 0
    label_video(args.input_video_fname, args.output_csv_fname,
                args.yolo_dir, args.start_from)
    #time_labeling(args.input_video_fname, args.output_csv_fname,
    #              args.yolo_dir, args.start_from)


if __name__ == '__main__':
    main()

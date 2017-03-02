#! /usr/bin/env python

import argparse
import cv2

def create_time_lapse(video_in_fname, video_out_fname, frames_to_skip=10):
    cap = cv2.VideoCapture(video_in_fname)
    ret, frame = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    res = frame.shape[0:2][::-1]
    vout = cv2.VideoWriter(video_out_fname, fourcc, 23.17, res)
    vout.write(frame)

    count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if count % frames_to_skip == 0:
            print 'Adding frame %d' % count
            vout.write(frame)
        count += 1

    vout.release()
    cap.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in', required=True,
            help='Video input filename. DO NOT confuse with video_out')
    parser.add_argument('--video_out', required=True,
            help='Video output filename.')
    parser.add_argument('--frames_to_skip', type=int, default=10,
            help='Number of frames to skip in input video')
    args = parser.parse_args()

    create_time_lapse(args.video_in, args.video_out, args.frames_to_skip)

if __name__ == '__main__':
    main()

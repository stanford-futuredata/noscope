import imageio
import argparse

def reencode(video_in_list, video_out_fname):
    reader = imageio.get_reader(video_in_list[0])
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(video_out_fname, fps=fps, macro_block_size=None)

    for video_in_fname in video_in_list:
        print 'Processing %s' % video_in_fname
        reader = imageio.get_reader(video_in_fname)
        num_frames = reader.get_meta_data()['nframes']
        print 're-encoding %d frames' % num_frames
        count = 0
        for im in reader:
            writer.append_data(im)
            count += 1
            if count % 100 == 0:
                print 'num frames processed: %d' % count
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in_list', required=True, help='List of video input filenames. DO NOT confuse with'
                        'video_out. Comma separated')
    parser.add_argument('--video_out', required=True, help='Video output filename.')
    args = parser.parse_args()

    video_in_list = args.video_in_list.split(',')
    reencode(video_in_list, args.video_out)

if __name__ == '__main__':
    main()

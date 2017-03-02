
from utils import execute_command
import argparse

# -vf "color=black:out_wxout_h [over]; [in][over] overlay=x:y [out]"
OVERLAY_TEMPLATE = 'avconv -i %s -vf "color=black:%dx%d [over]; [in][over]overlay=%d:%d [out]" %s'
NEW_FILE_SUFFIX = '.overlay'

def overlay_video(movie_full_path, x, y, width, height):
    filename_only = movie_full_path[movie_full_path.rindex('/') + 1:]
    new_filename = filename_only.replace('.mp4', '%s.mp4' % NEW_FILE_SUFFIX)
    new_full_path = movie_full_path.replace(filename_only, new_filename)
    execute_command(OVERLAY_TEMPLATE % (movie_full_path, width, height, x, y, new_full_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in', required=True,
            help='mp4 video to crop')
    parser.add_argument('--x', type=int, required=True,
            help='X coordinate')
    parser.add_argument('--y', type=int, required=True,
            help='Y coordinate')
    parser.add_argument('--width', type=int, required=True,
            help='width')
    parser.add_argument('--height', type=int, required=True,
            help='height')
    args = parser.parse_args()

    overlay_video(args.video_in, args.x, args.y, args.width, args.height)

if __name__ == '__main__':
    main()


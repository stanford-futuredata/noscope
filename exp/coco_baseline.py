
import itertools
import argparse
import noscope

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--coco_dir', required=True, help='Directory containing \
            MSCOCO datasets')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', type=int, help='Number of frames')
    # Regression or not
    parser.add_argument('--regression', dest='regression', action='store_true')
    parser.add_argument('--no-regression', dest='regression', action='store_false')
    parser.set_defaults(regression=False)
    # Binary or not (for classification)
    parser.add_argument('--binary', dest='binary', action='store_true')
    parser.add_argument('--no-binary', dest='binary', action='store_false')
    parser.set_defaults(binary=True)
    args = parser.parse_args()

    def check_args(args):
        if args.regression:
            if args.binary:
                print 'WARNING: Setting args.binary to False'
                args.binary = False
        else:
            # Check here?
            pass
        assert args.objects is not None
    check_args(args)

    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    X_train, Y_train, X_val, Y_val = noscope.DataUtils.read_coco_dataset(args.coco_dir,
            objects[0], resol=50)

    #print 'Preparing data....'
    data, nb_classes = noscope.DataUtils.get_data(
            args.csv_in, args.video_in,
            binary=args.binary,
            num_frames=args.num_frames,
            OBJECTS=objects,
            regression=args.regression,
            resol=(50, 50))
    _, _, X_test, Y_test = data

    data = [X_train, Y_train, X_test, Y_test]

    # Generally regression requires more iterations to converge.
    # Or so sleep deprived DK thinks
    nb_epoch = 15 + 15 * args.regression
    print 'Trying VGG-style nets....'
    # CIFAR10 based architectures
    noscope.Models.try_params(
            noscope.Models.generate_conv_net,
            list(itertools.product(
                    *[[X_train.shape[1:]], [nb_classes],
                      [64, 128, 256], [32, 64], [1, 2],
                      [0.2, 0.5, 1]])),
            data,
            args.output_dir,
            args.base_name,
            'convnet',
            objects[0],
            regression=args.regression,
            nb_epoch=nb_epoch,
            validation_data=(X_val, Y_val))

if __name__ == '__main__':
    main()

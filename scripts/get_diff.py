import itertools
import csv
import argparse
import noscope
import numpy as np
import sklearn
import keras
from keras.utils import np_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--model_in', required=True, help='Model output')
    parser.add_argument('--csv_out', required=True, help='DO NOT CONFUSE WITH CSV IN')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', type=int, help='Number of frames')
    args = parser.parse_args()

    model = keras.models.load_model(args.model_in)

    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    print 'Preparing data....'
    data, nb_classes = noscope.DataUtils.get_data(
            args.csv_in, args.video_in,
            binary=True,
            num_frames=args.num_frames,
            OBJECTS=objects,
            regression=False)
    X_train, Y_train, X_test, Y_test = data

    proba = model.predict(X_test, batch_size=256, verbose=0)

    if len(Y_test.shape) == 1:
        Y_test = np.transpose(np.array([1 - Y_test, Y_test]))
    predicted_labels = np_utils.probas_to_classes(proba)
    true_labels = np_utils.probas_to_classes(Y_test)
    print sklearn.metrics.accuracy_score(predicted_labels, true_labels)


    print true_labels.shape
    print predicted_labels.shape
    with open(args.csv_out, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['frame', 'yolo_label', 'small_cnn_label'])
        for ind in np.where(predicted_labels != true_labels)[0]:
            writer.writerow([int(ind), true_labels[ind], predicted_labels[ind]])


if __name__ == '__main__':
    main()

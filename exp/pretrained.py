import argparse
import os
import random
import h5py
import noscope
import numpy as np
from keras import optimizers
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils


# Weights are located at
# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
top_model_weights_path = 'bottleneck_fc_model.h5'


def save_bottleneck_features(X_train, X_test, weights_file, resol):
    # build the VGG16 network
    model = noscope.Models.generate_vgg16_conv(
            (resol, resol, 3),
            full_16=True, dropout=False)

    assert os.path.exists(weights_file), \
        'Model weights not found (see "weights_file" variable in script).'
    model.load_weights(weights_file)

    bottleneck_features_train = model.predict(X_train)
    # np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    bottleneck_features_validation = model.predict(X_test)
    # np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    return model, bottleneck_features_train, bottleneck_features_validation


def train_top_model(data, nb_epoch=50, regression=False):
    X_train, Y_train, X_test, Y_test = data
    nb_classes = len(Y_train[0])

    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(256, activation='relu'))#, W_constraint = maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    if not regression:
        model.add(Activation('softmax'))

    print X_train.shape
    print Y_train.shape
    print Y_train[0]

    loss = noscope.Models.get_loss(regression)
    print loss
    optimizer = optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=noscope.Models.computed_metrics)

    '''model.fit(X_train, Y_train,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(X_test, Y_test))'''
    noscope.Models.run_model(model, data, nb_epoch=nb_epoch, patience=10)
    if regression:
        print noscope.Models.evaluate_model_regression(model, X_test, Y_test)
    else:
        print noscope.Models.evaluate_model_multiclass(model, X_test, Y_test)
    # model.save_weights(top_model_weights_path)
    return model


def fine_tune(model, data, nb_epoch=5, regression=False):
    X_train, Y_train, X_test, Y_test = data

    for layer in model.layers[:25]:
        layer.trainable = False
    model.compile(loss=noscope.Models.get_loss(regression),
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=noscope.Models.computed_metrics)

    noscope.Models.run_model(model, data, nb_epoch=nb_epoch, patience=3)
    if regression:
        print noscope.Models.evaluate_model_regression(model, X_test, Y_test)
    else:
        print noscope.Models.evaluate_model_multiclass(model, X_test, Y_test)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_file', required=True, help='VGG16 weights')
    parser.add_argument('--csv_in', required=True, help='CSV input filename')
    parser.add_argument('--video_in', required=True, help='Video input filename')
    parser.add_argument('--csv_out_base', required=True, help='CSV output base. DO NOT confuse with csv_in')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', type=int, required=True, help='Number of frames')
    parser.add_argument('--resol', type=int, required=True, help='Resolution of image. Square')
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

    weights_file = args.weights_file

    objects = args.objects.split(',')

    data, nb_classes = noscope.DataUtils.get_data(
            args.csv_in, args.video_in,
            binary=args.binary,
            num_frames=args.num_frames,
            OBJECTS=objects,
            regression=args.regression,
            resol=(args.resol, args.resol),
            center=False)
    X_train, Y_train, X_test, Y_test = data
    # Who knows what's going on with VGG
    def to_vgg(X):
        X *= 255
        # X = X[:, :, :, ::-1]
        X[:, :, :, 0] -= 103.939
        X[:, :, :, 1] -= 116.779
        X[:, :, :, 2] -= 123.68
        return X
    X_train = to_vgg(X_train)
    X_test = to_vgg(X_test)
    # Flatten Y
    #Y_train = np.ravel(Y_train[:, 1])
    #Y_test = np.ravel(Y_test[:, 1])
    data = (X_train, Y_train, X_test, Y_test)

    # Train the FC parts
    conv_model, Xconv_train, Xconv_test = \
        save_bottleneck_features(X_train, X_test, weights_file, args.resol)
    data = (Xconv_train, Y_train, Xconv_test, Y_test)
    fc_model = train_top_model(data, nb_epoch=60, regression=args.regression)

    # Fine-tune
    model = conv_model
    model.add(fc_model)
    data = (X_train, Y_train, X_test, Y_test)
    fine_tune(model, data, nb_epoch=30, regression=args.regression)


if __name__ == '__main__':
    main()

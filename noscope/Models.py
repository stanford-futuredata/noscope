import sklearn
import sklearn.metrics
import time
import os
import tempfile
import StatsUtils
import DataUtils
import numpy as np
import keras.optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import np_utils

computed_metrics = ['accuracy', 'mean_squared_error']

# In case we want more callbacks
def get_callbacks(model_fname, patience=2):
    return [ModelCheckpoint(model_fname)]
    return [EarlyStopping(monitor='loss',     patience=patience, min_delta=0.00001),
            EarlyStopping(monitor='val_loss', patience=patience + 2, min_delta=0.0001),
            ModelCheckpoint(model_fname, save_best_only=True)]

def get_loss(regression):
    if regression:
        return 'mean_squared_error'
    else:
        return 'categorical_crossentropy'

def get_optimizer(regression, nb_layers, lr_mult=1):
    if regression:
        return keras.optimizers.RMSprop(lr=0.001 / (1.5 * nb_layers) * lr_mult)
    else:
        return keras.optimizers.RMSprop(lr=0.001 * lr_mult)# / (5 * nb_layers))


def generate_conv_net_base(
        input_shape, nb_classes,
        nb_dense=128, nb_filters=32, nb_layers=1, lr_mult=1,
        kernel_size=(3, 3), stride=(1, 1),
        regression=False):
    assert nb_layers >= 0
    assert nb_layers <= 3
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape,
                            subsample=stride,
                            activation='relu'))
    model.add(Convolution2D(nb_filters, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    if nb_layers > 1:
        model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    if nb_layers > 2:
        model.add(Convolution2D(nb_filters * 4, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(nb_filters * 4, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    if not regression:
        model.add(Activation('softmax'))

    loss = get_loss(regression)
    model.compile(loss=loss,
                  optimizer=get_optimizer(regression, nb_layers, lr_mult=lr_mult),
                  metrics=computed_metrics)
    return model


def generate_conv_net(input_shape, nb_classes,
                      nb_dense=128, nb_filters=32, nb_layers=1, lr_mult=1,
                      regression=False):
    return generate_conv_net_base(
            input_shape, nb_classes,
            nb_dense=nb_dense, nb_filters=nb_filters, nb_layers=nb_layers, lr_mult=lr_mult,
            regression=regression)


def generate_vgg16_conv(input_shape, full_16=False, dropout=True):
    border_mode = 'same'
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode=border_mode,
                            input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode=border_mode))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode=border_mode))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode=border_mode))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    if full_16:
        for i in xrange(2):
            model.add(Convolution2D(512, 3, 3, activation='relu', border_mode=border_mode))
            model.add(Convolution2D(512, 3, 3, activation='relu', border_mode=border_mode))
            model.add(Convolution2D(512, 3, 3, activation='relu', border_mode=border_mode))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            if dropout:
                model.add(Dropout(0.25))

    return model


def generate_vgg16(input_shape, nb_classes, full_16=False, regression=False):
    model = generate_vgg16_conv(input_shape, full_16=full_16, dropout=True)

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))

    if not regression:
        model.add(Activation('softmax'))

    loss = get_loss(regression)
    model.compile(loss=loss,
                  optimizer=get_optimizer(regression, 8 + full_16 * 8),
                  metrics=computed_metrics)
    return model


# Data takes form (X_train, Y_train, X_test, Y_test)
def run_model(model, data, batch_size=32, nb_epoch=1, patience=2,
        validation_data=(None, None)):
    X_train, Y_train, X_test, Y_test = data
    temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/tmp/')[1]

    # 50k should be a reasonable validation split
    if validation_data[0] is None:
        validation_split = 0.33333333
        if len(Y_train) * validation_split > 50000.0:
            validation_split = 50000.0 / float(len(Y_train))
        print validation_split

        begin_train = time.time()
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  # validation_split=validation_split,
                  # validation_data=(X_test, Y_test),
                  shuffle=True,
                  class_weight='auto',
                  callbacks=get_callbacks(temp_fname, patience=patience))
        train_time = time.time() - begin_train
    else:
        begin_train = time.time()
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=validation_data,
                  shuffle=True,
                  class_weight='auto',
                  callbacks=get_callbacks(temp_fname, patience=patience))
        train_time = time.time() - begin_train

    model.load_weights(temp_fname)
    os.remove(temp_fname)

    return train_time


def get_labels(model, X_test, batch_size=256, get_time=False):
    begin = time.time()
    ## Alternate way to compute the classes
    # proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    # predicted_labels = np_utils.probas_to_classes(proba)
    predicted_labels = model.predict_classes(X_test, batch_size=batch_size, verbose=0)
    end = time.time()
    if get_time:
        return predicted_labels, end - begin
    else:
        return predicted_labels


def stats_from_proba(proba, Y_test):
    # Binary and one output
    if proba.shape[1] == 1:
        proba = np.concatenate([1 - proba, proba], axis=1)
    if len(Y_test.shape) == 1:
        Y_test = np.transpose(np.array([1 - Y_test, Y_test]))
    predicted_labels = np_utils.probas_to_classes(proba)

    true_labels = np_utils.probas_to_classes(Y_test)
    precision, recall, fbeta, support = sklearn.metrics.precision_recall_fscore_support(
            predicted_labels, true_labels)
    accuracy = sklearn.metrics.accuracy_score(predicted_labels, true_labels)

    num_penalties, thresh_low, thresh_high = \
        StatsUtils.yolo_oracle(Y_test[:, 1], proba[:, 1])
    windowed_acc, windowed_supp = StatsUtils.windowed_accuracy(predicted_labels, Y_test)

    metrics = {'precision': precision,
               'recall': recall,
               'fbeta': fbeta,
               'support': support,
               'accuracy': accuracy,
               'penalities': num_penalties,
               'windowed_accuracy': windowed_acc,
               'windowed_support': windowed_supp}
    return metrics


def evaluate_model_regression(model, X_test, Y_test, batch_size=256):
    begin = time.time()
    raw_predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
    end = time.time()
    mse = sklearn.metrics.mean_squared_error(Y_test, raw_predictions)

    Y_classes = Y_test > 0.2 # FIXME
    Y_classes = np.concatenate([1 - Y_classes, Y_classes], axis=1)

    best = {'accuracy': 0}
    for cutoff in np.arange(0.01, 0.75, 0.01):
        predictions = raw_predictions > cutoff # FIXME
        proba = np.concatenate([1 - predictions, predictions], axis=1)
        metrics = stats_from_proba(proba, Y_classes)
        metrics['cutoff'] = cutoff
        print 'Cutoff: %f, metrics: %s' % (cutoff, str(metrics))
        if metrics['accuracy'] > best['accuracy']:
            best = metrics

    metrics = best
    metrics['mse'] = mse
    metrics['test_time'] = end - begin
    return metrics


def evaluate_model_multiclass(model, X_test, Y_test, batch_size=256):
    begin = time.time()
    proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_time = time.time() - begin

    metrics = stats_from_proba(proba, Y_test)
    metrics['test_time'] = test_time
    return metrics


def evaluate_model(model, X_test, Y_test, batch_size=256):
    predicted_labels, test_time = get_labels(model, X_test, batch_size, True)
    true_labels = np_utils.probas_to_classes(Y_test)

    confusion = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)

    # Minor smoothing to prevent division by 0 errors
    TN = float(confusion[0][0]) + 1
    FN = float(confusion[1][0]) + 1
    TP = float(confusion[1][1]) + 1
    FP = float(confusion[0][1]) + 1
    metrics = {'recall': TP / (TP + FN),
               'specificity': TN / (FP + TN),
               'precision': TP / (TP + FP),
               'npv':  TN / (TN + FN),
               'fpr': FP / (FP + TN),
               'fdr': FP / (FP + TP),
               'fnr': FN / (FN + TP),
               'accuracy': (TP + TN) / (TP + FP + TN + FN),
               'f1': (2 * TP) / (2 * TP + FP + FN),
               'test_time': test_time}
    return metrics


def learn_and_eval(model, data, nb_epoch=2, batch_size=128,
        validation_data=(None, None)):
    X_train, Y_train, X_test, Y_test = data
    train_time = run_model(model, data, nb_epoch=nb_epoch,
            batch_size=batch_size, validation_data=validation_data)
    metrics = evaluate_model(model, X_test, Y_test, batch_size=batch_size)
    return train_time, metrics


# NOTE: assumes first two parameters are: (image_size, nb_classes)
def try_params(model_gen, params, data,
               output_dir, base_fname, model_name, OBJECT,
               regression=False, nb_epoch=2, validation_data=(None, None)):
    def metrics_names(metrics):
        return sorted(metrics.keys())
    def metrics_to_list(metrics):
        return map(lambda key: metrics[key], metrics_names(metrics))

    summary_csv_fname = os.path.join(
            output_dir, base_fname + '_' + model_name + '_summary.csv')

    X_train, Y_train, X_test, Y_test = data
    nb_classes = params[1]
    to_write = []
    for param in params:
        param_base_fname = base_fname + '_' + model_name + '_' + '_'.join(map(str, param[2:]))
        model_fname = os.path.join(
                output_dir, param_base_fname + '.h5')
        csv_fname = os.path.join(
                output_dir, param_base_fname + '.csv')

        # Make, train, and evaluate the model
        model = model_gen(*param, regression=regression)
        if regression:
            train_time = run_model(model, data, nb_epoch=nb_epoch,
                    validation_data=validation_data)
            metrics = evaluate_model_regression(model, X_test, Y_test)
        else:
            if nb_classes == 2:
                train_time, metrics = learn_and_eval(model, data,
                        validation_data=validation_data)
            else:
                train_time = run_model(model, data, nb_epoch=nb_epoch,
                        validation_data=validation_data)
                metrics = evaluate_model_multiclass(model, X_test, Y_test)

        # Output predictions and save the model
        # Redo some computation to save my sanity
        conf1 = model.predict(X_train, batch_size=256, verbose=0)
        conf2 = model.predict(X_test,  batch_size=256, verbose=0)
        conf = np.concatenate([conf1, conf2])
        if len(conf.shape) > 1:
            assert len(conf.shape) == 2
            assert conf.shape[1] <= 2
            if conf.shape[1] == 2:
                conf = conf[:, 1]
            else:
                conf = np.ravel(conf)
        DataUtils.confidences_to_csv(csv_fname, conf, OBJECT)
        model.save(model_fname)

        to_write.append(list(param[2:]) + [train_time] + metrics_to_list(metrics))
        print param
        print train_time, metrics
        print
    print to_write
    # First two params don't need to be written out
    param_column_names = map(lambda i: 'param' + str(i), xrange(len(params[0]) - 2))
    column_names = param_column_names + ['train_time'] + metrics_names(metrics)
    DataUtils.output_csv(summary_csv_fname, to_write, column_names)

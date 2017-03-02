import time

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import GlobalAveragePooling2D, Convolution2D, MaxPooling2D
import numpy as np


def conv2_max1(RESOL=(224, 224, 3),
               conv_units=64, dense_units=64,
               classes=2):
    img_input = Input(shape=RESOL)
    x = Convolution2D(conv_units, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(conv_units, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(dense_units, activation='relu', name='fc1')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    model = Model(inputs, x, name='conv2_max1')

    return model


def run_speed_test(model, batch_sizes=[16, 32, 64, 128, 256],
                   nb_images=10000):
    data = np.random.rand(*tuple([nb_images] + list(model.inputs[0].get_shape()[1:])))
    for batch_size in batch_sizes:
        begin = time.time()
        _ = model.predict(data, batch_size=batch_size)
        end = time.time()
        total_time = end - begin
        print 'Batch size: %d, FPS: %f' % (batch_size, nb_images / total_time)


def main():
    m1 = conv2_max1()
    run_speed_test(m1)


if __name__ == '__main__':
    main()
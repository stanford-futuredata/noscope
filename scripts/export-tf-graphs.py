import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.training import saver as saver_lib
from keras.models import model_from_json

from freeze_graph import freeze_graph


def get_keras_model(fname, sess):
    from keras import backend as K
    from keras.models import load_model
    K.set_session(sess)

    learned_model = load_model(fname)
    json_string = learned_model.to_json()
    weights = learned_model.get_weights()

    K.set_learning_phase(0)

    new_model = model_from_json(json_string)
    new_model.set_weights(weights)
    return new_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Models with *.h5')
    parser.add_argument('--output_dir', required=True, help='Where to output the files (as *.pb)')
    args = parser.parse_args()

    fnames = glob.glob(os.path.join(args.model_dir, '*.h5'))
    first = np.zeros( (1, 50, 50, 3) )

    for keras_model_fname in fnames:
        sess = tf.Session()
        frozen_graph_path = os.path.splitext(os.path.split(keras_model_fname)[1])[0] + '.pb'
        frozen_graph_path = os.path.join(args.output_dir, frozen_graph_path)
        print 'Producing: ' + frozen_graph_path

        model = get_keras_model(keras_model_fname, sess)
        img1 = tf.placeholder(tf.float32, shape=(None, 50, 50, 3), name='input_img')
        tf_model = model(img1)
        output = tf.identity(tf_model, name='output_prob')

        # Run to set weights
        sess.run(output,
                 feed_dict = {img1: first})
        tf.train.write_graph(sess.graph_def, '/tmp', 'graph-structure.pb')
        saver = saver_lib.Saver()
        checkpoint_path = saver.save(sess, '/tmp/vars', global_step=0)

        input_graph_path = '/tmp/graph-structure.pb'
        input_saver_def_path = ''
        input_binary = False
        input_checkpoint_path = '/tmp/vars-0'
        output_node_names = 'output_prob'
        restore_op_name = 'save/restore_all'
        filename_tensor_name = 'save/Const:0'
        clear_devices = False
        initializer_nodes = ""
        freeze_graph(input_graph_path, input_saver_def_path,
                     input_binary, input_checkpoint_path,
                     output_node_names, restore_op_name,
                     filename_tensor_name, frozen_graph_path,
                     clear_devices, initializer_nodes)

        # Clean up
        sess.close()
        tf.reset_default_graph()


if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
import np_helper

FLAGS = tf.app.flags.FLAGS


def _read_conv_params(in_dir, name):
    weights = np_helper.load_nparray(in_dir + name + '_weights.bin', np.float32)
    biases = np_helper.load_nparray(in_dir + name + '_biases.bin', np.float32)
    return weights, biases


def read_vgg_init(in_dir):
    names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
             'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    layers = {}
    for name in names:
        weights, biases = _read_conv_params(in_dir, name)
        layers[name + '/weights'] = weights
        layers[name + '/biases'] = biases

    # transform fc6 parameters to conv6_1 parameters
    #weights, biases = _read_conv_params(in_dir, 'fc6')
    #weights = weights.reshape((7, 7, 512, 4096))
    #layers['conv6_1' + '/weights'] = weights
    #layers['conv6_1' + '/biases'] = biases
    #names.append('conv6_1')
    return layers, names

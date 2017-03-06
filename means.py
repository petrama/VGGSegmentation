import tensorflow as tf
import read_cityscapes_tf_records as reader
import train_helper
import time
import os


import eval_helper
import numpy as np

import helper

import sys

tf.app.flags.DEFINE_string('config_path', "config/cityscapes.py", """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.__dict__['__flags'].keys())


def main(argv=None):

    train_data, train_labels, train_names, train_weights = reader.inputs(shuffle=True,
                                                                                 num_epochs=1,
                                                                                 dataset_partition='train')
    sess = tf.Session()
    sess.run(tf.initialize_local_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    for i in range(1):
        print(i)
        labels, weights = sess.run([train_labels, train_weights])
        l255 = labels[0, labels[0] == 255]
        suma=0
        for j in range(19):
            print('Label {}'.format(j))

            lj=labels[0,labels[0]==j]
            wj=weights[0,labels[0]==j]
            iznos=len(lj)/(len(labels[0]))

            print(iznos)
            if len(wj)>0:
             print('tezina',wj[0])
             d=wj[0]*iznos
            else:
                d=0
            suma+=d
        print(suma)






    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    tf.app.run()
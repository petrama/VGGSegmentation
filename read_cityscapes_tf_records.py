import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

import glob


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'num_labels': tf.FixedLenFeature([], tf.int64),
            'img_name': tf.FixedLenFeature([], tf.string),
            'rgb': tf.FixedLenFeature([], tf.string),
            'label_weights': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),

        })

    image = tf.decode_raw(features['rgb'], tf.uint8)
    labels_unary = tf.decode_raw(features['labels'], tf.uint8)


    weights = tf.decode_raw(features['label_weights'], tf.float32)
    img_name = features['img_name']


    image = tf.reshape(image, shape=[FLAGS.img_height, FLAGS.img_width, FLAGS.num_channels])
    image=tf.to_float(image)


    num_pixels = FLAGS.img_height * FLAGS.img_width
    labels = tf.reshape(labels_unary, shape=[num_pixels,])
    labels=tf.to_float(labels)
    labels=tf.cast(labels,tf.int32)
    weights = tf.reshape(weights, shape=[num_pixels,])

    return (image, labels,
            img_name, weights)


def get_filenames(dataset_partition):
    return glob.glob(os.path.join(FLAGS.dataset_dir, dataset_partition, '*'))


def inputs(shuffle=True, num_epochs=False, dataset_partition='train'):
    if not num_epochs:
        num_epochs = None

    files = get_filenames(dataset_partition)


    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs,
                                                        shuffle=shuffle,
                                                        capacity=len(files))


        (image, labels, img_name, weights) = read_and_decode(filename_queue)

        (image, labels,
         img_name, weights) = tf.train.batch(
            [image, labels,
             img_name, weights], batch_size=FLAGS.batch_size,num_threads=8)


        return (image, labels, img_name, weights)


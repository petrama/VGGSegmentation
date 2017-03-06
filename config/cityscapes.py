import tensorflow as tf
import train_helper
import os

flags=tf.app.flags


SAVE_DIR=os.path.join('/home/petra/results/cityscapes/',train_helper.get_time_string())

flags.DEFINE_string('train_dir',SAVE_DIR,'new folder to save outputs')


flags.DEFINE_string('model_path', 'models/model.py',
                    'path to model definition')


# used in prepare_tfrecords
flags.DEFINE_string('dataset_dir', '/home/petra/datasets/Cityscapes/tensorflow/640x288/',
                    'Directory in which tfrecord files are saved,contains subdirs train and val')

#flags.DEFINE_string('train_dir_name','train','name of train dir inside of dataset dir')
#flags.DEFINE_string('val_dir_name','val','name of validation dir inside of dataset dir')

flags.DEFINE_string('vgg_init_dir','/home/petra/vgg16/vgg16/','folder with vgg weights')
# resized image dimensions
flags.DEFINE_integer('img_width', 640, 'Resized image width')
flags.DEFINE_integer('img_height', 288, 'Resized image height')
flags.DEFINE_integer('num_channels', 3, 'Resized image depth')

# training parameters
# Batch size
tf.app.flags.DEFINE_integer('train_size',2976,'')
tf.app.flags.DEFINE_integer('val_size',500,'')
tf.app.flags.DEFINE_integer('batch_size', 1, '')


# Number of classes in the dataset
flags.DEFINE_integer('num_classes', 19,
                     'Number of classes in the dataset')

# Regularization factor
flags.DEFINE_float('reg_factor', 0.0005, 'Regularization factor')

# Learning rate
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
                          'Epochs after which learning rate decays.')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')

tf.app.flags.DEFINE_integer('max_epochs', 1, 'Number of batches to run.')

tf.app.flags.DEFINE_string('optimizer','Adam','')

tf.app.flags.DEFINE_float('weight_decay',1e-3,'l2 weight decay for regularization')

tf.app.flags.DEFINE_string('resume_path','','')



flags.DEFINE_integer('subsample_factor', 8, 'Subsample factor of the model')

tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('draw_predictions', False, 'Whether to draw.')
tf.app.flags.DEFINE_boolean('save_net', False, 'Whether to save.')

tf.app.flags.DEFINE_integer('seed', 66478, '')

flags.DEFINE_float('r_mean', 74.042, 'Mean value for the red channel')
flags.DEFINE_float('g_mean', 83.733, 'Mean value for the green channel')
flags.DEFINE_float('b_mean', 73.106, 'Mean value for the blue channel')

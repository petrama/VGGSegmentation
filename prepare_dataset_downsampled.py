import os
import pickle
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
import skimage.transform
from tqdm import trange


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir',
    '/home/petra/datasets/Cityscapes/', 'Dataset dir')
tf.app.flags.DEFINE_integer('img_width', 640, '')
tf.app.flags.DEFINE_integer('img_height', 288, '')
#tf.app.flags.DEFINE_integer('img_width', 1024, '')
#tf.app.flags.DEFINE_integer('img_height', 448, '')
tf.app.flags.DEFINE_string('save_dir', '/home/petra/datasets/Cityscapes/tensorflow/'
    + '{}x{}'.format(FLAGS.img_width, FLAGS.img_height) + '/', '')
# leave out the car hood
tf.app.flags.DEFINE_integer('cx_start', 0, '')
tf.app.flags.DEFINE_integer('cx_end', 2048, '')
tf.app.flags.DEFINE_integer('cy_start', 0, '')
tf.app.flags.DEFINE_integer('cy_end', 900, '')

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(rgb, label_map, weight_map, depth_img, num_labels, img_name, save_dir):
  rows = rgb.shape[0]
  cols = rgb.shape[1]
  depth = rgb.shape[2]

  filename = os.path.join(save_dir + img_name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  rgb_str = rgb.tostring()
  labels_str = label_map.tostring()
  weights_str = weight_map.tostring()
  #disp_raw = depth_img.tostring()
  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(rows),
      'width': _int64_feature(cols),
      'depth': _int64_feature(depth),
      'num_labels': _int64_feature(int(num_labels)),
      'img_name': _bytes_feature(img_name.encode()),
      'rgb': _bytes_feature(rgb_str),
      'label_weights': _bytes_feature(weights_str),
      'labels': _bytes_feature(labels_str)}))
      #'disparity': _bytes_feature(disp_raw)}))
  writer.write(example.SerializeToString())
  writer.close()


def prepare_dataset(name):
  print('Preparing ' + name)
  root_dir = FLAGS.data_dir + '/rgb/' + name + '/'
  depth_dir = os.path.join(FLAGS.data_dir, 'depth', name)
  print(depth_dir)
  gt_dir = FLAGS.data_dir + '/gt_data/' + name + '/'
  cities = next(os.walk(root_dir))[1]
  save_dir = FLAGS.save_dir + name + '/'
  print('Save dir = ', save_dir)
  os.makedirs(save_dir, exist_ok=True)
  #print('Writing', filename)
  cx_start = FLAGS.cx_start
  cx_end = FLAGS.cx_end
  cy_start = FLAGS.cy_start
  cy_end = FLAGS.cy_end
  img_cnt = 0
  depth_sum = np.zeros((FLAGS.img_height, FLAGS.img_width))
  for city in cities:
    print(city)
    img_list = next(os.walk(root_dir + city))[2]
    for i in trange(len(img_list)):
      img_cnt += 1
      img_name = img_list[i]
      img_prefix = img_name[:-4]
      rgb_path = root_dir + city + '/' + img_name
      rgb = ski.data.load(rgb_path)
      rgb = np.ascontiguousarray(rgb[cy_start:cy_end,cx_start:cx_end,:])
      rgb = ski.transform.resize(
          rgb, (FLAGS.img_height, FLAGS.img_width), preserve_range=True, order=3)
      rgb = rgb.astype(np.uint8)
      depth_path = os.path.join(depth_dir, city, img_name[:-4] + '_leftImg8bit.png')
      depth_img = ski.data.load(depth_path)
      depth_img = np.ascontiguousarray(depth_img[cy_start:cy_end,cx_start:cx_end])
      depth_img = ski.transform.resize(depth_img, (FLAGS.img_height, FLAGS.img_width),
                                       order=0, preserve_range=True)
      depth_img = np.round(depth_img / 256.0).astype(np.uint8)
      depth_sum += depth_img
      print((depth_sum / img_cnt).mean((0,1)))

      gt_path = gt_dir + city + '/' + img_prefix + '.pickle'
      with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
      gt_ids = gt_data[0]
      #gt_weights = gt_data[1]
      num_labels = gt_data[2]
      class_weights = gt_data[4]
      assert num_labels == (gt_ids < 255).sum()
      gt_ids = np.ascontiguousarray(gt_ids[cy_start:cy_end,cx_start:cx_end])
      gt_ids = ski.transform.resize(gt_ids, (FLAGS.img_height, FLAGS.img_width),
                                    order=0, preserve_range=True).astype(np.uint8)
      #ski.io.imsave(save_dir + img_name, gt_ids)
      gt_ids = gt_ids.astype(np.int8)
      gt_weights = np.zeros((FLAGS.img_height, FLAGS.img_width), np.float32)
      for i, wgt in enumerate(class_weights):
        gt_weights[gt_ids == i] = wgt

      # Just to test correct casting in numpy/skimage - this must be the same
      #gt_ids_test = ski.util.img_as_ubyte(gt_ids_test).astype(np.int8)
      #assert (gt_ids != gt_ids_test).sum() == 0

      # Calucate new number of labels (255 is now = -1)
      num_labels = (gt_ids >= 0).sum()

      #convert_colors_to_indices(gt_rgb, class_color_map)
      create_tfrecord(rgb, gt_ids, gt_weights, depth_img, num_labels, img_prefix, save_dir)


def main(argv):
  prepare_dataset('val')
  prepare_dataset('train')


if __name__ == '__main__':
  tf.app.run()

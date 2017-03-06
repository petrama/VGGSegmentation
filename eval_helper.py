import os
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io


def draw_output(y, class_colors, save_path):
  width = y.shape[0]
  height = y.shape[1]
  y_rgb = np.zeros((height, width, 3), dtype=np.uint8)
  for cid in range(len(class_colors)):
    cpos = np.repeat((y == cid).reshape((height, width, 1)), 3, axis=2)
    cnum = cpos.sum() // 3
    y_rgb[cpos] = np.array(class_colors[cid][:3] * cnum, dtype=np.uint8)
    #pixels = y_rgb[[np.repeat(np.equal(y, cid).reshape((height, width, 1)), 3, axis=2)]
    #if pixels.size > 0:
    #  #pixels.reshape((-1, 3))[:,:] = class_colors[cid][:3]
    #  #pixels.resize((int(pixels.size/3), 3))
    #  print(np.array(class_colors[cid][:3] * (pixels.size // 3), dtype=np.uint8))
    #  pixels = np.array(class_colors[cid][:3] * (pixels.size // 3), dtype=np.uint8)
    #y_rgb[np.repeat(np.equal(y, cid).reshape((height, width, 1)), 3, axis=2)].reshape((-1, 3)) = \
    #    class_colors[cid][:3]
  ski.io.imsave(save_path, y_rgb)


def draw_prediction_slow(y, colors, path):
  width = y.shape[1]
  height = y.shape[0]
  col = np.zeros(3)
  yimg = np.empty((height, width, 3), dtype=np.uint8)
  for i in range(height):
    for j in range(width):
      cid = y[i,j]
      for k in range(3):
        yimg[i,j,k] = colors[cid][k]
      #img[i,j,:] = col
  #print(yimg.shape)
  #yimg = ski.transform.resize(yimg, (height*16, width*16), order=0, preserve_range=True).astype(np.uint8)
  ski.io.imsave(path, yimg)


def collect_confusion_matrix(y, yt, conf_mat,max_label):
  for i in range(y.size):
    l = y[i]
    lt = yt[i]
    if lt >= 0 and lt<max_label:
      conf_mat[l,lt] += 1

def compute_errors(conf_mat,name,class_names, verbose=True):
  num_correct = conf_mat.trace()
  num_classes = conf_mat.shape[0]
  total_size = conf_mat.sum()
  avg_pixel_acc = num_correct / total_size * 100.0
  TPFN = conf_mat.sum(0)
  TPFP = conf_mat.sum(1)
  FN = TPFN - conf_mat.diagonal()
  FP = TPFP - conf_mat.diagonal()
  class_iou = np.zeros(num_classes)
  class_recall = np.zeros(num_classes)
  class_precision = np.zeros(num_classes)
  if verbose:
    print(name + ' errors:')
  for i in range(num_classes):
    TP = conf_mat[i,i]
    if (TP + FP[i] + FN[i])>0:
      class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
    if TPFN[i] > 0:
      class_recall[i] = (TP / TPFN[i]) * 100.0
    else:
      class_recall[i] = 0
    if TPFP[i] > 0:
      class_precision[i] = (TP / TPFP[i]) * 100.0
    else:
      class_precision[i] = 0

    if verbose:
        print('\t%s IoU accuracy = %.2f %%' % (class_names[i], class_iou[i]))

  avg_class_iou = class_iou.mean()
  avg_class_recall = class_recall.mean()
  avg_class_precision = class_precision.mean()
  if verbose:
    print(name + ' IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
    print(name + ' mean class recall - TP / (TP+FN) = %.2f %%' % avg_class_recall)
    print(name + ' mean class precision - TP / (TP+FP) = %.2f %%' % avg_class_precision)
    print(name + ' pixel accuracy = %.2f %%' % avg_pixel_acc)
  return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size


def plot_training_progresss(save_dir, loss, iou, pixel_acc):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 6
  title_size = 10
  train_color = 'm'
  val_color = 'c'

  x_data = np.linspace(1, len(loss[0]), len(loss[0]))
  ax1.set_title('cross entropy loss', fontsize=title_size)
  ax1.plot(x_data, loss[0], marker='o', color=train_color, linewidth=linewidth, linestyle='-', \
      label='train')
  ax1.plot(x_data, loss[1], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
      label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('IoU accuracy')
  ax2.plot(x_data, iou[0], marker='o', color=train_color, linewidth=linewidth, linestyle='-',
      label='train')
  ax2.plot(x_data, iou[1], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
      label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('pixel accuracy')
  ax3.plot(x_data, pixel_acc[0], marker='o', color=train_color, linewidth=linewidth, linestyle='-',
      label='train')
  ax3.plot(x_data, pixel_acc[1], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
      label='validation')
  ax3.legend(loc='upper left', fontsize=legend_size)
  #ax4.set_title('')
  #plt.figure(fig.number)
  #plt.clf()
  #plt.plot(x_data, data, 'b-')
  #plt.axis([0, 6, 0, 20])
  #plt.show()
  #plt.draw()
  #plt.show(block=False)
  #plt.savefig('training_plot.pdf', bbox_inches='tight')
  save_path = os.path.join(save_dir, 'training_plot.pdf')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def plot_training_progress(save_dir, data):
  print('plot training progress: ', data)
  fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(24, 8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)


  ax2.set_title('Learning rate')
  ax2.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax2.legend(loc='upper right', fontsize=legend_size)

  ax3.set_title('IoU accuracy')
  ax3.plot(x_data, data['train_iou'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax3.plot(x_data, data['valid_iou'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax3.legend(loc='upper left', fontsize=legend_size)


  ax4.set_title('Pixel accuracy')
  ax4.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax4.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax4.legend(loc='upper left', fontsize=legend_size)

  ax5.set_title('Average precision')
  ax5.plot(x_data, data['train_prec'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax5.plot(x_data, data['valid_prec'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax5.legend(loc='upper left', fontsize=legend_size)

  ax6.set_title('Average recall')
  ax6.plot(x_data, data['train_rec'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax6.plot(x_data, data['valid_rec'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax6.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)


def map_cityscapes_to_kitti(y, id_map):
  y_kitti = np.zeros(y.shape, dtype=y.dtype)
  for i in range(len(id_map)):
    #print(i , ' --> ', id_map[i])
    #print(np.equal(y, i).sum(), '\n')
    #print('sum')
    #print(y[np.equal(y, i)].sum())
    y_kitti[np.equal(y, i)] = id_map[i]
    #print(np.equal(y, id_map[i]).sum())
    #y[np.equal(y, i)] = 2
    #print(y[np.equal(y, i)].sum())
  return y_kitti


#def plot_accuracy(fig, data):
#  x_data = np.linspace(1, len(data), len(data))
#  plt.figure(fig.number)
#  plt.clf()
#  plt.plot(x_data, data, 'b-')
#  plt.savefig(str(fig.number) + '_plot.pdf', bbox_inches='tight')

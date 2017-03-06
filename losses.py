import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def add_loss_summaries(total_loss):
  """Add summaries for losses in model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.

  for l in losses + [total_loss]:
    #print(l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name + ' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))
    #tf.scalar_summary([l.op.name + ' (raw)'], l)
    #tf.scalar_summary([l.op.name], loss_averages.average(l))

  return loss_averages_op


def total_loss_sum(losses):
  # Assemble all of the losses for the current tower only.
  #losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
  #print(losses)
  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  #total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return total_loss


def softmax(logits):
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.op_scope([logits], None, 'Softmax'):
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    softmax_1d = tf.nn.softmax(logits_1d)
    softmax_2d = tf.reshape(softmax_1d, [FLAGS.img_height, FLAGS.img_width, FLAGS.num_classes])
  return softmax_2d


def weighted_cross_entropy_loss(logits, labels, weights=None, num_labels=1, max_weight=100):
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.op_scope([logits, labels], None, 'WeightedCrossEntropyLoss'):
    labels = tf.reshape(labels, shape=[num_examples])
    num_labels = tf.to_float(tf.reduce_sum(num_labels))
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])

    log_softmax = tf.nn.log_softmax(logits_1d)
    xent = tf.reduce_sum(-tf.mul(tf.to_float(one_hot_labels), log_softmax), 1)
    if weights != None:
      weights = tf.reshape(weights, shape=[num_examples])
      xent = tf.mul(tf.minimum(tf.to_float(max_weight), weights), xent)
    total_loss = tf.div(tf.reduce_sum(xent), tf.to_float(num_labels), name='value')
    return tf.div(total_loss,tf.to_float(num_labels))
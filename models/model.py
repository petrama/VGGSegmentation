import tensorflow as tf
import tensorflow.contrib.layers as layers
from model_helper import read_vgg_init


import losses

FLAGS = tf.app.flags.FLAGS


def total_loss_sum(losses):
    # Assemble all of the losses for the current tower only.
    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    return total_loss


def create_init_op(vgg_layers):
    variables = tf.contrib.framework.get_variables()
    init_map = {}
    for var in variables:
        name_split = var.name.split('/')
        if len(name_split) != 3:
            continue
        name = name_split[1] + '/' + name_split[2][:-2]
        if name in vgg_layers:
            print(var.name, ' --> init from ', name)
            init_map[var.name] = vgg_layers[name]
            print(var.name,vgg_layers[name].shape)
        else:
            print(var.name, ' --> random init')



    init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
    return init_op, init_feed


def build(inputs, labels, weights, is_training=True):

    vgg_layers, vgg_layer_names = read_vgg_init(FLAGS.vgg_init_dir)


    weight_decay = 5e-4
    bn_params = {
        # Decay for the moving averages.
        'decay': 0.999,
        'center': True,
        'scale': True,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # None to force the updates
        'updates_collections': None,
        'is_training': is_training,
    }
    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=3, stride=1, padding='SAME', rate=1, activation_fn=tf.nn.relu,
                                        # normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
                                        # weights_initializer=layers.variance_scaling_initializer(),
                                        normalizer_fn=None, weights_initializer=None,
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.convolution2d(inputs, 64, scope='conv1_1')
        net = layers.convolution2d(net, 64, scope='conv1_2')
        net = layers.max_pool2d(net, 2, 2, scope='pool1')
        net = layers.convolution2d(net, 128, scope='conv2_1')
        net = layers.convolution2d(net, 128, scope='conv2_2')
        net = layers.max_pool2d(net, 2, 2, scope='pool2')
        net = layers.convolution2d(net, 256, scope='conv3_1')
        net = layers.convolution2d(net, 256, scope='conv3_2')
        net = layers.convolution2d(net, 256, scope='conv3_3')
        net = layers.max_pool2d(net, 2, 1, scope='pool3',padding='SAME')

        paddings = [[0, 0], [0, 0]]
        crops = [[0, 0], [0, 0]]


        block_size=2
        net=tf.space_to_batch(net,paddings=paddings,block_size=block_size)
        net = layers.convolution2d(net, 512, scope='conv4_1')
        net = layers.convolution2d(net, 512, scope='conv4_2')
        net = layers.convolution2d(net, 512, scope='conv4_3')
        net = tf.batch_to_space(net, crops=crops, block_size=block_size)

        net = layers.max_pool2d(net, 2, 1, scope='pool4',padding='SAME')

        block_size=4
        net=tf.space_to_batch(net,paddings=paddings,block_size=block_size)
        net = layers.convolution2d(net, 512, scope='conv5_1')
        net = layers.convolution2d(net, 512, scope='conv5_2')
        net = layers.convolution2d(net, 512, scope='conv5_3')
        net=tf.batch_to_space(net,crops=crops,block_size=block_size)




    with tf.contrib.framework.arg_scope([layers.convolution2d],stride=1,padding='SAME',
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        activation_fn=tf.nn.relu,normalizer_fn=layers.batch_norm,
                                        normalizer_params=bn_params,
                                        weights_regularizer=layers.l2_regularizer(FLAGS.weight_decay)):
        net = layers.convolution2d(net, 512, kernel_size=3, scope='conv6_1',rate=8)



    logits = layers.convolution2d(net, FLAGS.num_classes, 1,padding='SAME', activation_fn=None,scope='unary_2',rate=2)
    print('logits',logits.get_shape())

    logits=tf.image.resize_bilinear(logits,[FLAGS.img_height,FLAGS.img_width],name='resize_score')




    loss=get_loss(logits,labels,weights,is_training=is_training)



    if is_training:
        init_op, init_feed = create_init_op(vgg_layers)
        return logits, loss, init_op, init_feed

    return logits,loss




def get_loss(logits, labels,weights, is_training):
    #xent_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
    xent_loss=losses.weighted_cross_entropy_loss(logits,labels,weights)
    total_loss = total_loss_sum([xent_loss])
    if is_training:
        loss_averages_op = losses.add_loss_summaries(total_loss)
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

    return total_loss
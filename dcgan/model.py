import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


def dcgan_generator(input_image, is_train, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('dcgan_gen', reuse=reuse):

        n = InputLayer(input_image, name='gen_input')

        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, name='k9n64s1/conv')

        for i in range(5):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, name='k3n64s1/conv1_%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, name='k3n64s1/bn1_%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, name='k3n64s1/conv2_%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, name='k3n64s1/bn1_%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='add_in_block_%s' % i)
            nn = SubpixelConv2d(nn, scale=2, name='upscale_%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, name='bn2_%s' % i)
            nn = DeConv2d(nn, 64, (3, 3), (1, 1), name='deconv_%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, name='k3n64s1/conv_1')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, name='k3n64s1/conv_2')
        n = Conv2d(n, 3, (3, 3), (1, 1), act=tf.nn.sigmoid, name='k3n64s1/conv_3')

        return n


def dcgan_discriminator(input_images, is_train, reuse=tf.AUTO_REUSE):
    leaky_relu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope('dcgan_dis', reuse=reuse):

        n_in = InputLayer(input_images, name='dis_input')

        n_0 = Conv2d(n_in, 64, (3, 3), (1, 1), act=leaky_relu, name='n_0/conv')

        n_1 = Conv2d(n_0, 64, (3, 3), (2, 2), act=None, name='n_1/conv')
        n_1 = BatchNormLayer(n_1, act=leaky_relu, is_train=is_train, name='n_1/bn')

        n_2 = Conv2d(n_1, 128, (3, 3), (2, 2), act=None, name='n_2/conv')
        n_2 = BatchNormLayer(n_2, act=leaky_relu, is_train=is_train, name='n_2/bn')

        n_3 = Conv2d(n_2, 256, (3, 3), (2, 2), act=None, name='n_3/conv')
        n_3 = BatchNormLayer(n_3, act=leaky_relu, is_train=is_train, name='n_3/bn')

        n_4 = Conv2d(n_3, 512, (3, 3), (2, 2), act=None, name='n_4/conv')
        n_4 = BatchNormLayer(n_4, act=leaky_relu, is_train=is_train, name='n_4/bn')

        n_5 = Conv2d(n_4, 64, (3, 3), (1, 1), act=None, name='n_5/conv')
        n_5 = BatchNormLayer(n_5, act=tf.nn.relu, is_train=is_train, name='n_5/bn')

        n_6 = Conv2d(n_5, 64, (3, 3), (1, 1), act=None, name='n_6/conv')
        n_6 = BatchNormLayer(n_6, act=tf.nn.relu, is_train=is_train, name='n_6/bn')

        n_7 = FlattenLayer(n_6, name='flatten')
        n_7 = DenseLayer(n_7, n_units=1024, act=leaky_relu, name='n_8/dense')
        n_8 = DenseLayer(n_7, n_units=1, name='n_9/dense')

        logits = n_8.outputs
        n_8.outputs = tf.nn.sigmoid(n_8.outputs)

        return n_8, logits

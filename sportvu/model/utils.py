import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, trainable=True)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, trainable=False)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def max_pool3d_2x2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                          strides=[1, 2, 2, 2, 1], padding='SAME')


def bn(x, training):
  return tf.layers.batch_normalization(x, axis=-1, training=training
                                       )
def gaussian_noise_layer(input_layer, std):
  noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
  return input_layer + noise


if __name__ == '__main__':
  import numpy as np
  inp = tf.placeholder(tf.float32, shape=[None, 8], name='input')
  noise_level = tf.placeholder(tf.float32)
  noise = gaussian_noise_layer(inp, noise_level)
  noise.eval(session=tf.Session(), feed_dict={inp: np.zeros((4, 8)), noise_level:0.1})

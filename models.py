import ops
import tensorflow as tf


#
# def generator(inputs, stddev=0.02, alpha=0.2, name='generator', reuse=False):
#     with tf.variable_scope(name, reuse=reuse) as scope:
#         fc1 = tf.layers.dense(inputs, 64 * 8 * 4 * 4, name='fc1')
#         re1 = tf.reshape(fc1, (-1, 4, 4, 512), name='reshape')
#         bn1 = tf.layers.batch_normalization(re1, name='bn1')
#         # ac1 = tf.maximum(alpha * bn1, bn1,name='ac1')
#         ac1 = tf.nn.relu(bn1, name='ac1')
#
#
#         de_conv1 = tf.layers.conv2d_transpose(ac1, 256, kernel_size=[5, 5], padding='same', strides=2,
#                                               kernel_initializer=tf.random_normal_initializer(stddev=stddev),
#                                               name='decov1')
#         bn2 = tf.layers.batch_normalization(de_conv1, name='bn2')
#          # ac2 = tf.maximum(alpha * bn2, bn2,name='ac2')
#         ac2 = tf.nn.relu(bn2, name='ac2')
#
#
#         de_conv2 = tf.layers.conv2d_transpose(ac2, 128, kernel_size=[5, 5], padding='same',
#                                              kernel_initializer=tf.random_normal_initializer(stddev=stddev), strides=2,
#                                              name='decov2')
#         bn3 = tf.layers.batch_normalization(de_conv2, name='bn3')
#          # ac3 = tf.maximum(alpha * bn3, bn3,name='ac3')
#         ac3 = tf.nn.relu(bn3, name='ac3')
#
#
#         de_conv3 = tf.layers.conv2d_transpose(ac3, 64, kernel_size=[5, 5], padding='same',
#                                              kernel_initializer=tf.random_normal_initializer(stddev=stddev), strides=2,
#                                              name='decov3')
#         bn4 = tf.layers.batch_normalization(de_conv3, name='bn4')
#          # ac4 = tf.maximum(alpha * bn4, bn4,name='ac4')
#         ac4 = tf.nn.relu(bn4, name='ac4')
#
#
#         logits = tf.layers.conv2d_transpose(ac4, 3, kernel_size=[5, 5], padding='same',
#                                             kernel_initializer=tf.random_normal_initializer(stddev=stddev), strides=2,
#                                             name='logits')
#
#
#         output = tf.tanh(logits)
#
#
#         return output

#
# def discriminator(inputs, stddev=0.02, alpha=0.2, batch_size=64, name='discriminator', reuse=False):
#     with tf.variable_scope(name, reuse=reuse) as scope:
#         conv1 = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), padding='same',
#                                  kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv1')
#
#         ac1 = tf.maximum(alpha * conv1, conv1, name='ac1')
#
#         conv2 = tf.layers.conv2d(ac1, 128, (5, 5), (2, 2), padding='same',
#                                  kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv2')
#         bn2 = tf.layers.batch_normalization(conv2, name='bn2')
#         ac2 = tf.maximum(alpha * bn2, bn2, name='ac2')
#
#         conv3 = tf.layers.conv2d(ac2, 256, (5, 5), (2, 2), padding='same',
#                                  kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv3')
#         bn3 = tf.layers.batch_normalization(conv3, name='bn3')
#         ac3 = tf.maximum(alpha * bn3, bn3, name='ac3')
#
#         conv4 = tf.layers.conv2d(ac3, 512, (5, 5), (2, 2), padding='same',
#                                  kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv4')
#         bn4 = tf.layers.batch_normalization(conv4, name='bn4')
#         ac4 = tf.maximum(alpha * bn4, bn4, name='ac4')
#
#         flat = tf.reshape(ac4, shape=[batch_size, 4 * 4 * 512], name='reshape')
#
#         fc2 = tf.layers.dense(flat, 1, kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='fc2')
#         return fc2



def generator(input,reuse=False,is_training=True,batch_size=64):
    with tf.variable_scope('generator', reuse=reuse):

        with tf.variable_scope('layers1', reuse=reuse):
            weights = tf.get_variable('weights',[100,4*4*64*8], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.matmul(input, weights)
            y=tf.contrib.layers.batch_norm(y,decay=0.9,scale=True,updates_collections=None,is_training=is_training)
            y=tf.nn.relu(y)#[batchsize,4*4*64*8]
            y=tf.reshape(y, [-1, 4, 4, 64*8])#[batchsize,4,4,64*8]

        with tf.variable_scope('layers2', reuse=reuse):
            weights = tf.get_variable('weights',[5,5,64*4,64*8], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d_transpose(y, weights, output_shape=[batch_size,8,8,64*4],strides=[1, 2, 2, 1], padding='SAME')
            y = tf.contrib.layers.batch_norm(y,decay=0.9,scale=True,updates_collections=None,is_training=is_training)
            y = tf.nn.relu(y)##[batchsize,8,8,64*4]

        with tf.variable_scope('layers3', reuse=reuse):
            weights = tf.get_variable('weights',[5,5,64*2,64*4], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d_transpose(y, weights, output_shape=[batch_size,16,16,64*2],strides=[1, 2, 2, 1], padding='SAME')
            y = tf.contrib.layers.batch_norm(y,decay=0.9,scale=True,updates_collections=None,is_training=is_training)
            y = tf.nn.relu(y)##[batchsize,16,16,64*2]

        with tf.variable_scope('layers4', reuse=reuse):
            weights = tf.get_variable('weights',[5,5,64,64*2], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d_transpose(y, weights,output_shape= [batch_size,32,32,64],strides=[1, 2, 2, 1], padding='SAME')
            y = tf.contrib.layers.batch_norm(y,decay=0.9,scale=True,updates_collections=None,is_training=is_training)
            y = tf.nn.relu(y)##[batchsize,32,32,64]

        with tf.variable_scope('layers5', reuse=reuse):
            weights = tf.get_variable('weights',[5,5,3,64], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d_transpose(y, weights,output_shape= [batch_size,64,64,3],strides=[1, 2, 2, 1], padding='SAME')
            y = tf.nn.tanh(y)##[batchsize,64,64,3]

        return y

#我的判别器有问题，找到它y=tf.sigmoid(y)不能要
def discriminator(input,reuse=False,is_training=True):
    with tf.variable_scope('discriminator', reuse=reuse):

        with tf.variable_scope('layers1', reuse=reuse):
            weights = tf.get_variable('weights', [5, 5, 3, 64], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d(input, weights, strides=[1, 2, 2, 1],padding='SAME')
            y = tf.maximum(0.2* y, y)##[batchsize,32,32,64]

        with tf.variable_scope('layers2', reuse=reuse):
            weights = tf.get_variable('weights', [5, 5, 64, 64*2], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d(y,weights, strides=[1, 2, 2,1 ], padding='SAME')
            y=tf.contrib.layers.batch_norm(y, decay=0.9, scale=True,updates_collections=None,is_training=is_training)
            y = tf.maximum(0.2* y, y)##[batchsize,16,16,64*2]

        with tf.variable_scope('layers3', reuse=reuse):
            weights = tf.get_variable('weights', [5, 5, 64*2, 64*4], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d(y, weights, strides=[1, 2, 2, 1], padding='SAME')
            y = tf.contrib.layers.batch_norm(y, decay=0.9, scale=True, updates_collections=None, is_training=is_training)
            y = tf.maximum(0.2 * y, y)  ##[batchsize,8,8,64*4]

        with tf.variable_scope('layers4', reuse=reuse):
            weights = tf.get_variable('weights', [5, 5, 64*4, 64*8], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.nn.conv2d(y, weights, strides=[1, 2, 2, 1], padding='SAME')
            y = tf.contrib.layers.batch_norm(y, decay=0.9, scale=True, updates_collections=None, is_training=is_training)
            y = tf.maximum(0.2 * y, y)  ##[batchsize,4,4,64*8]

        with tf.variable_scope('layers5', reuse=reuse):
            y = tf.reshape(y, [-1, 4*4*64*8])
            weights = tf.get_variable('weights',[4*4*64*8,1], initializer=tf.random_normal_initializer(mean=0, stddev=0.02, dtype=tf.float32))
            y = tf.matmul(y, weights)
            #y=tf.contrib.layers.batch_norm(y,decay=0.9,scale=True,updates_collections=None,is_training=is_training)
            #y=tf.sigmoid(y)#[batchsize,1]

        return y

import tensorflow as tf

def _weights(name,shape, mean=0.0,stddev=0.02):
    var=tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(mean=mean,stddev=stddev,dtype=tf.float32)
    )
    return var


def _biases(name,shape,constant=0.0):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(constant))


def _batch_norm(input,is_training,norm='batch'):
    if norm=='batch':
        with tf.variable_scope("batch_norm"):
         return  tf.contrib.layers.batch_norm(input,
                                             decay=0.9,
                                             scale=True,
                                             updates_collections=None,
                                             is_training=is_training
                                             )
    else:
        return input

def _leaky_relu(input, slope=0.2):
  return tf.maximum(slope*input, input)

#转置卷积+bias+norm+relu
def _conv2d_transpose(input,num_units,reuse=False,norm='batch',is_training=True,name=None,output_size=None):

    with tf.variable_scope(name,reuse=reuse):
        input_shape=input.get_shape().as_list()#get_shape()返回的是元组 as_list()将元组转换成list
        weights=_weights("weights",shape=[5,5,num_units,input_shape[3]])  #定义卷积核
        biases=_biases("biases",shape=[num_units])                        #定义偏置

        if not output_size:
            output_size=input_shape[1]*2
        output_shape=[input_shape[0],output_size,output_size,num_units]    #

        fsconv=tf.nn.conv2d_transpose(input, weights, output_shape=output_shape,
                                      strides=[1, 2, 2, 1], padding='SAME')      #转置卷积 升采样

        #add_biases = tf.nn.bias_add(fsconv, biases) # output=fsconv+biases   卷积结果加上偏置
        normalised=_batch_norm(fsconv,is_training,norm)   #batch normalization
        output=tf.nn.relu(normalised)

    return output

def gene_last_layer(input,num_units,reuse=False,norm='batch',is_training=True,name=None,output_size=None):

    with tf.variable_scope(name,reuse=reuse):
        input_shape=input.get_shape().as_list()#get_shape()返回的是元组 as_list()将元组转换成list
        weights=_weights("weights",shape=[5,5,num_units,input_shape[3]])  #定义卷积核

        if not output_size:
            output_size=input_shape[1]*2
        output_shape=[input_shape[0],output_size,output_size,num_units]    #

        fsconv=tf.nn.conv2d_transpose(input,weights,output_shape=output_shape,
                                      strides=[1,2,2,1],padding='SAME')      #转置卷积 升采样

        normalised=_batch_norm(fsconv,is_training,norm)   #batch normalization
        output=tf.nn.tanh(normalised)

    return output



#卷积+bias+norm+leak relu
def _conv2d(input,num_units,reuse=False,norm='batch',is_training=True,name=None):

    with tf.variable_scope(name,reuse=reuse):
        input_shape=input.get_shape().as_list()#get_shape()返回的是元组 as_list()将元组转换成list
        weights=_weights("weights",shape=[5,5,input_shape[3],num_units])  #定义卷积核
        biases=_biases("biases",shape=[num_units])                        #定义偏置

        conv=tf.nn.conv2d(input,weights,
                            strides=[1,2,2,1],padding='SAME')      #转置卷积 升采样

        #add_biasconves = tf.nn.bias_add(conv, biases) # output=conv+biases   卷积结果加上偏置
        normalised=_batch_norm(conv,is_training,norm)   #batch normalization
        output=_leaky_relu(normalised)

    return output


#只针对二维 tensor
def fully_connection(input,num_units,reuse=False,norm='batch',is_training=True,name=None,activation='relu'):
    with tf.variable_scope(name,reuse=reuse):
        input_shape = input.get_shape().as_list()
        input1 = tf.reshape(input, [input_shape[0], -1])
        input1_shape = input1.get_shape().as_list()

        weights=_weights("weights",shape=[input1_shape[1],num_units])  #定义卷积核
        biases=_biases("biases",shape=[num_units])

        fulconnect=tf.matmul(input1,weights)
        #add_biases=fulconnect+biases
        normalised = _batch_norm(fulconnect, is_training, norm)

        if activation=='relu':
            output = tf.nn.relu(normalised)
        elif activation=='sigmoid':
            output = tf.sigmoid(normalised)

    return output
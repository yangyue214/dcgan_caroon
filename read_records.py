
import os
import tensorflow as tf
import glob
from PIL import Image  #注意Image,后面会用到


def read_and_decode(tf_path,batch_size):

    filename_queue = tf.train.string_input_producer([tf_path])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    imgf = tf.cast(img, tf.float32)/127.5-1.0
    img_batch = tf.train.shuffle_batch([imgf],
                                     batch_size=batch_size, capacity=2000,
                                     min_after_dequeue=500
                                       )
    return img_batch






#img_batch = read_and_decode('tf_records/cartoon.tfrecords', batch_size=10)

if __name__  ==  '__main__':
    batch_size=64
    epoch=3
    img_nums=len( glob.glob('dataset/faces'+'/*.jpg') )
    #print(img_nums)#51223
    run_nums = (img_nums//batch_size)*epoch

    img_batch = read_and_decode('tf_records/cartoon.tfrecords', batch_size=batch_size)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)  #初始化全局变量

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in range(2):
                #while not coord.should_stop():
                val = sess.run([img_batch])
                #print(val.shape, type(val))
                #print(len(val))
                if i==(run_nums-1):
                    print(val[0].shape)
            print("end")
        except tf.errors.OutOfRangeError:
            print('out of range')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)


#须知 此处read_and_decode（）img_batch 为list ，len为1，list里面是array，array尺寸是[batchsize,64,64,3]
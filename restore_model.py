import tensorflow as tf
import utils
import numpy as np

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('checkpoints/DCGAN.meta')
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

   # graph=tf.get_default_graph()
    #z=graph.get_tensor_by_name('z:0')
   # sample_out=graph.get_tensor_by_name('test_out:0')
    z = sess.graph.get_tensor_by_name('z:0')
    sample_out = sess.graph.get_tensor_by_name('test_out:0')

    z_ipt_sample = np.random.normal(size=[64, 100])
    test_sample_opt = sess.run(sample_out, feed_dict={z: z_ipt_sample})
    utils.mkdir('infer_cartoon')
    utils.imwrite(utils.immerge(test_sample_opt, 10, 10), 'infer_cartoon/' + 'test.jpg')


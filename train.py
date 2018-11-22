
#记录于2018.11.22 问题1.判别器最后一层不能使用sigmoid，使用的话效果很差，不知道是为什么
                # 问题2.dis_fake=models.discriminator(fake,  reuse=True) 有这一句的时候tensorboard不能显示graph


import tensorflow as tf
import models
import glob
import read_records
import numpy as np
import utils

batch_size =64
epoch =50
img_nums = len(glob.glob('dataset/faces' + '/*.jpg'))
# print(img_nums)#51223
run_nums = (51223 // batch_size) * epoch
alpha = 0.2


def train():

    graph = tf.Graph()

    with graph.as_default():

        z=tf.placeholder(tf.float32,shape=[64 ,100],name='z')

        img_batch = read_records.read_and_decode('tf_records/cartoon.tfrecords', batch_size=batch_size)
        #generator
        # fake=models.generator(z, stddev=0.02, alpha=alpha, name='generator', reuse=False)
        #
        # #discriminator
        # dis_real=models.discriminator(img_batch , alpha=alpha, batch_size=batch_size)
        # dis_fake=models.discriminator(fake,  alpha=alpha, reuse=True)

        #generator
        fake=models.generator(z,  reuse=False)#, is_training=True

        #discriminator
        dis_real=models.discriminator(img_batch ,  reuse=False  )#is_training=True
        dis_fake=models.discriminator(fake,  reuse=True)#,  is_training=True

        # #losses
        # gene_loss = tf.reduce_mean(tf.squared_difference(dis_fake, 0.9))
        # dis_loss = (tf.reduce_mean(tf.squared_difference(dis_real, 0.9))
        #             + tf.reduce_mean(tf.square(dis_fake))) / 2

        gene_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_fake) * 0.9, logits=dis_fake))
        d_f_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dis_fake), logits=dis_fake))
        d_r_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_real) * 0.9, logits=dis_real))
        dis_loss = d_f_loss + d_r_loss

        gen_loss_sum=tf.summary.scalar("gen_loss",gene_loss)
        dis_loss_sum=tf.summary.scalar("dis_loss",dis_loss)
        merge_sum_gen=tf.summary.merge([gen_loss_sum])
        merge_sum_dis = tf.summary.merge([dis_loss_sum])

        #variables
        gene_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        gene_opt=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.3).minimize(gene_loss ,  var_list=gene_var )
        dis_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.3).minimize(dis_loss, var_list=dis_var)

        test_sample = models.generator(z,  reuse=True)#,  is_training=False
        test_out=tf.add(test_sample,0,'test_out')

        init = tf.global_variables_initializer()
    print('t')

    with tf.Session(graph=graph) as sess:
        sess.run(init)  # 初始化全局变量

        z_ipt_sample = np.random.normal(size=[batch_size , 100])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.summary.FileWriter('./tensorboard', sess.graph)
        saver = tf.train.Saver()
        try:
            for i in range(run_nums ):
                z_ipt = np.random.normal(size=[batch_size, 100])
                #train D
                #_, dis_loss1 = sess.run([dis_opt,dis_loss],feed_dict={real:img_batch,z:z_ipt})
                sum_dis, _, dis_loss1 = sess.run([merge_sum_dis,dis_opt, dis_loss], feed_dict={ z: z_ipt})
                #train G
                sum_gen, _, gen_loss1 = sess.run([merge_sum_gen,gene_opt,gene_loss],feed_dict={z:z_ipt})


                if i%400==0:
                    print(i)
                    test_sample_opt = sess.run(test_sample , feed_dict={z: z_ipt_sample})
                    #print(type(test_sample_opt),test_sample_opt.shape)
                    utils.mkdir('out_cartoon')
                    utils.imwrite(utils.immerge(test_sample_opt, 10,10), 'out_cartoon/'+str(i)+'.jpg')
                   # writer.add_summary(sum_dis, i)
                    #writer.add_summary(sum_gen, i)
            print("train end!!!")

        except tf.errors.OutOfRangeError:
            print('out of range')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
        writer.close()
        saver.save(sess, "./checkpoints/DCGAN")



train()
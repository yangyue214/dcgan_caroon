
import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import glob


def make_tfrecords(imgpath, tf_name):
    if not os.path.exists('tf_records'):
        os.makedirs('tf_records')

    writer = tf.python_io.TFRecordWriter('tf_records/'+tf_name)  # 要生成的文件

    for img_path in glob.glob(imgpath+'/*.jpg'):
        img = Image.open(img_path)
        img = img.resize((64, 64))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()






if __name__  ==  '__main__':
    make_tfrecords('dataset/faces', 'cartoon.tfrecords')


#glist=glob.glob('dataset/faces//'+'*.jpg')
#print(glist[0])#dataset/faces\0000fdee4208b8b7e12074c920bc6166-0.jpg





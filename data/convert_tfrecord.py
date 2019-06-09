import sys
import argparse
import numpy as np
import tensorflow as tf
import os

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_txt", default='./2012_train.txt')
    parser.add_argument("--tfrecord_path_prefix", default='./train_data/train')
    parser.add_argument("--basic_path", default='/home/han/vocdata/VOCdevkit/VOC2012/JPEGImages')
    flags = parser.parse_args()
    dataset = {}
    with open(flags.dataset_txt,'r') as f:
        for line in f.readlines():
            example = line.split(' ')
            image_path = os.path.join(flags.basic_path, example[0])
            boxes_num = len(example[1:]) // 5
            boxes = np.zeros([boxes_num, 5], dtype=np.float32)
            for i in range(boxes_num):
                boxes[i] = example[1+i*5:6+i*5]
            dataset[image_path] = boxes

    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print(">> Processing %d images" %images_num)

    tfrecord_file = flags.tfrecord_path_prefix+".tfrecords"
    with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
        for i in range(images_num):
            image = tf.gfile.FastGFile(image_paths[i], 'rb').read()
            boxes = dataset[image_paths[i]]
            boxes = boxes.tostring()

            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'boxes' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [boxes])),
                }
            ))

            record_writer.write(example.SerializeToString())
        print(">> Saving %d images in %s" %(images_num, tfrecord_file))


if __name__ == "__main__":
    main(sys.argv[1:])



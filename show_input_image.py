import cv2
import numpy as np
import tensorflow as tf
from core import utils
from PIL import Image
from core.dataset import Parser, dataset
sess = tf.Session()

BATCH_SIZE = 1
SHUFFLE_SIZE = 1
CLASSES = utils.read_coco_names('./data/class.names')
ANCHORS = utils.get_anchors('./data/anchors.txt')
NUM_CLASSES = len(CLASSES)
TRAIN_TFRECORD = "./data/train_data/val.tfrecords"
TEST_TFRECORD = "./data/val_data/val.tfrecords"
parser_train = Parser(ANCHORS, NUM_CLASSES)
parser_test = Parser(ANCHORS, NUM_CLASSES)
trainset = dataset(parser_train, TRAIN_TFRECORD, BATCH_SIZE, shuffle=SHUFFLE_SIZE, multi_image_size=False)
testset = dataset(parser_test, TEST_TFRECORD, BATCH_SIZE, shuffle=None)
is_training = tf.placeholder(tf.bool)
example = testset.get_next()

for l in range(1):
    res = sess.run(example)
    image = res[0][0] * 255
    y_true = res[1:]
    boxes = utils.decode_gtbox(y_true)
    n_box = len(boxes)
    for i in range(n_box):
        image = cv2.rectangle(image,(int(float(boxes[i][0])),
                                     int(float(boxes[i][1]))),
                                    (int(float(boxes[i][2])),
                                     int(float(boxes[i][3]))), (255,0,0), 1)
        image = cv2.putText(image, "object", (int(float(boxes[i][0])),int(float(boxes[i][1]))),
                            cv2.FONT_HERSHEY_SIMPLEX,  .6, (0, 255, 0), 1, 2)
    print(image.shape)
    image = Image.fromarray(np.uint8(image))
    image.show()

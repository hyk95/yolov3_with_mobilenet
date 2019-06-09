import tensorflow as tf
import numpy as np
from core import utils

if __name__ == '__main__':
    ANCHORS = utils.get_anchors('./data/anchors.txt')
    inputs = tf.placeholder(dtype=tf.int32, shape=[None, None, None, None])
    img_size = tf.shape(inputs)[1:3]
    _ANCHORS = tf.multiply(tf.convert_to_tensor(ANCHORS), tf.cast(img_size, dtype=tf.float32))
    sess = tf.Session()
    x = sess.run(_ANCHORS, feed_dict={inputs: np.random.rand(2, 480, 480, 3)})
    print(ANCHORS * [480, 480])
    print(x)
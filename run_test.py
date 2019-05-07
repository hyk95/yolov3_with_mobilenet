import tensorflow as tf
from core import utils, yolov3
from core.dataset import dataset, Parser
from basicNet.mobilenetV2 import MobilenetV2
from config.config import *

parser   = Parser(IMAGE_H, IMAGE_W, ANCHORS, NUM_CLASSES)
testset  = dataset(parser, TEST_TFRECORD , BATCH_SIZE, shuffle=None)

example = testset.get_next()

images, *y_true = example
model = yolov3.yolov3(NUM_CLASSES, ANCHORS, basic_net=MobilenetV2)

with tf.variable_scope('yolov3'):
    pred_feature_map = model.forward(images, is_training=False)
    y_pred           = model.predict(pred_feature_map)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./checkpoint/yolov3.ckpt-2500")
    run_items = sess.run([images, y_pred, y_true])
    for i in range(8):
        image = run_items[0][i]
        pred_boxes = run_items[1][0][i:i+1]
        pred_confs = run_items[1][1][i:i+1]
        pred_probs = run_items[1][2][i:i+1]
        pred_boxes, pred_scores, pred_labels = utils.cpu_nms(pred_boxes, pred_confs*pred_probs, 1)
        im = utils.draw_boxes(image*255, pred_boxes, pred_scores, pred_labels, "1", (416, 416), show=True)

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from mobilenet.mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
from utils.boxes_utils import decoder_v2


def reorg(x, stride):
    return tf.extract_image_patches(x, [1, stride, stride, 1],
                        [1, stride, stride, 1], [1,1,1,1], padding="VALID", name="reorg")


def variable_summaries(var, name):
    with tf.name_scope('summaries'+name):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)


def eval_file(line, input_shape):
    element = line.split(" ")
    image = cv2.imread(element[0])
    h, w = input_shape
    old_h, old_w = image.shape[:2]
    image = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)
    image_data = image / 255.
    image_data = np.expand_dims(image_data, axis=0)
    return image_data, old_h, old_w


class Yolov2(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.classes = dataset.classes
        self.anchor = np.reshape(dataset.anchors, [1, 1, 1, -1, 2])
        self.num_classes = len(self.classes)
        self.num_anchors = self.anchor.shape[3]
        self.input_shape = dataset.input_shape
        self.stride = 32
        self.output_shape = [n//self.stride for n in self.input_shape]

    def create_model(self, inputs, is_training=True):
        with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training, batch_norm_decay=0.9)):
            net, end_points = mobilenet_v1(inputs, is_training=is_training)
            # shortcut
            shortcut = end_points["Conv2d_11_pointwise"]
            shortcut = slim.conv2d(shortcut, 512,
                                   kernel_size=[1, 1],
                                   stride=1,
                                   normalizer_fn=slim.batch_norm,
                                   scope="Conv2d_shortcut")
            shortcut = reorg(shortcut, 2)
            net = tf.concat([shortcut, net], axis=-1, name="Connect")
            net = slim.conv2d(net, 1024,
                              kernel_size=[3, 3],
                              stride=1,
                              normalizer_fn=slim.batch_norm,
                              scope="Conv2d_14")
            net = slim.conv2d(net, (5 + self.num_classes) * self.num_anchors,
                              kernel_size=[3, 3],
                              stride=1,
                              normalizer_fn=slim.batch_norm,
                              activation_fn=None,
                              scope="Conv_dec")
            return net

    def boxes_conf_clf(self, net_out):
        out_shape = tf.shape(net_out)
        net_out = tf.reshape(net_out, shape=[out_shape[0], out_shape[1], out_shape[2], self.num_anchors, -1])
        boxes = net_out[..., :4]
        confidence = net_out[..., 4:5]
        classes = net_out[..., 5:]
        return boxes, confidence, classes

    def decoder_pred(self, pre_boxes, pre_confidence, pre_classes):
        pre_boxes_xy = tf.nn.sigmoid(pre_boxes[..., :2])
        pre_boxes_wh = tf.exp(pre_boxes[..., 2:4]) * self.anchor / self.input_shape[0]
        pre_confidence = tf.nn.sigmoid(pre_confidence)
        pre_classes = tf.nn.softmax(pre_classes)
        return pre_boxes_xy, pre_boxes_wh, pre_confidence, pre_classes

    def pred_box_cls(self, net_out):
        boxes_shape = tf.shape(net_out)
        pre_boxes_xy, pre_boxes_wh, pre_confidence, pre_classes = self.decoder_pred(*self.boxes_conf_clf(net_out))
        anchors = tf.constant(np.reshape(self.anchor, [1, 1, -1, 2]), dtype=tf.float32)
        grid_x = tf.tile(tf.reshape(tf.range(0, boxes_shape[2]), [-1, 1, 1, 1]), [1, boxes_shape[1], 1, 1])
        grid_y = tf.tile(tf.reshape(tf.range(0, boxes_shape[1]), [1, -1, 1, 1]), [boxes_shape[2], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.cast(tf.tile(grid, [1, 1, 1, self.num_anchors, 1]), tf.float32)
        pre_score = tf.reduce_max(pre_confidence * pre_classes, axis=-1, keepdims=True)
        classes_index = tf.argmax(pre_confidence * pre_classes, axis=-1)
        bbox_xy = pre_boxes_xy + grid
        bbox_wh = tf.exp(pre_boxes_wh) * anchors
        index = tf.where(pre_score > 0.8, name=None)
        box_xy_wh_scroce = tf.concat([bbox_xy, bbox_wh, pre_score], axis=-1)
        # boxes = tf.boolean_mask(box_xy_wh_scroce, index)
        return box_xy_wh_scroce, classes_index

    def yolo_loss(self, y_true, y_pred):
        out_shape = tf.cast(tf.shape(y_pred), dtype=tf.float32)
        pre_boxes, pre_confidence, pre_classes = self.boxes_conf_clf(y_pred)
        true_boxes, true_confidence, true_classes = self.boxes_conf_clf(y_true)

        true_boxes_xy = true_boxes[..., :2]
        true_boxes_wh = tf.exp(true_boxes[..., 2:4]) * self.anchor / self.input_shape[0]
        true_boxes_wh = true_boxes_wh * out_shape[1]
        pre_boxes_xy, pre_boxes_wh, pre_confidence, pre_classes = self.decoder_pred(pre_boxes, pre_confidence, pre_classes)
        pre_boxes_wh = pre_boxes_wh * out_shape[1]

        pre_centers = pre_boxes_xy
        pre_areas = pre_boxes_wh[..., 0] * pre_boxes_wh[..., 1]
        pre_up_left, pre_down_right = pre_centers - (pre_boxes_wh * 0.5), pre_centers + (pre_boxes_wh * 0.5)

        true_centers = true_boxes_xy
        true_areas = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
        true_up_left, true_down_right = true_centers - (true_boxes_wh * 0.5), true_centers + (true_boxes_wh * 0.5)

        inter_upleft = tf.maximum(pre_up_left, true_up_left)
        inter_downright = tf.minimum(pre_down_right, true_down_right)
        inter_wh = tf.maximum(inter_downright - inter_upleft, 0.0)
        intersects = inter_wh[..., 0] * inter_wh[..., 1]
        ious = tf.truediv(intersects, true_areas + pre_areas - intersects)

        neg_iou_mask = tf.less_equal(ious, 0.3)
        neg_iou_mask = tf.expand_dims(tf.cast(neg_iou_mask, tf.float32), -1)
        best_iou_mask = tf.equal(ious, tf.reduce_max(ious, axis=3, keepdims=True))
        best_iou_mask = tf.expand_dims(tf.cast(best_iou_mask, tf.float32), -1)
        mask = best_iou_mask * true_confidence

        loss_wh = tf.reduce_sum(mask * 0.5 * tf.pow(true_boxes[..., 2:4]-pre_boxes[..., 2:4], 2)) * 5
        loss_xy = tf.reduce_sum(mask * 0.5 * tf.pow(true_boxes_xy - pre_boxes_xy, 2)) * 5
        loss_conf_obj = tf.reduce_sum(mask * 0.5 * tf.pow(true_confidence-pre_confidence, 2))
        loss_conf_noobj = tf.reduce_sum((1-mask) * 0.5 * tf.pow(true_confidence-pre_confidence, 2))
        loss_cls = tf.reduce_sum(mask * 0.5 * tf.pow(true_classes-pre_classes, 2))
        # tf.losses.add_loss([loss_wh, loss_xy, loss_conf_obj, loss_conf_noobj, loss_cls])

        loss_all = loss_wh + loss_xy + loss_conf_obj + loss_conf_noobj + loss_cls
        loss_all = loss_all / out_shape[0]

        variable_summaries(loss_xy, "loss_xy")
        variable_summaries(loss_wh, "loss_wh")
        variable_summaries(loss_conf_obj, "loss_conf_obj")
        variable_summaries(loss_conf_noobj, "loss_conf_noobj")
        variable_summaries(loss_cls, "loss_cls")
        variable_summaries(loss_all, "loss_all")
        loss_all = tf.Print(loss_all, [loss_all,
                                       loss_wh,
                                       loss_xy,
                                       loss_conf_obj,
                                       loss_conf_noobj,
                                       loss_cls,
                                       tf.reduce_sum((1-mask)),
                                       tf.reduce_sum(mask)], message='loss: ')
        return loss_all

    def yolo_train(self, epochs, batch_size, weight_path):
        dataset = self.dataset
        with tf.Graph().as_default():
            im_inputs = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], 3])
            net_out = self.create_model(im_inputs, is_training=True)
            y_true = tf.placeholder(tf.float32, [None, None, None, self.num_anchors*(self.num_classes + 5)])
            net_loss = self.yolo_loss(y_true, net_out)
            tf.losses.add_loss(net_loss)
            weight_loss = tf.losses.get_regularization_losses()
            net_out_loss = tf.losses.get_losses()
            all_loss = weight_loss + net_out_loss
            cost = tf.add_n(all_loss, name="total_loss")
            global_step = tf.Variable(0, trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                Adam_optim = tf.train.AdamOptimizer(learning_rate=0.0001)
                optim = slim.learning.create_train_op(cost, Adam_optim, global_step=global_step)
            train_writer = tf.summary.FileWriter("../tem_log", tf.get_default_graph())
            merge_summary = tf.summary.merge_all()
            saver = tf.train.Saver(var_list=tf.global_variables())
            load_fn = slim.assign_from_checkpoint_fn(
                weight_path,
                tf.global_variables(),
                ignore_missing_vars=True)
            with tf.Session() as sess:
                print("load:", weight_path)
                load_fn(sess)
                # saver.restore(sess, weight_path)
                # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                for epoch in range(epochs):
                    startTime = time.time()
                    for iter_ in range(self.dataset.num_data // batch_size):
                        x, y = dataset.read_data_box(batch_size)
                        if iter_ % 50 == 0:
                            loss, _, train_summary, step = sess.run([cost, optim, merge_summary, global_step], feed_dict={im_inputs: x, y_true: y})
                            train_writer.add_summary(train_summary, step)
                            print("epoch:{}/iter:{}/loss:{}/".format(epoch, iter_, loss))
                        else:
                            _ = sess.run([optim], feed_dict={im_inputs: x, y_true: y})
                    endTime = time.time()
                    print("epoch_time:{}".format(endTime - startTime))
                saver.save(sess, "../model_tem/yolov2_mobile.ckpt")

    def yolo_eval(self, weight_path, annotation_path):
        eval = open("eval.txt", 'w')
        with open(annotation_path) as f:
            all_lines = f.readlines()
        num_data = len(all_lines)
        with tf.Graph().as_default():
            im_inputs = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], 3])
            net_out = self.create_model(im_inputs, is_training=False)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, weight_path)
                for i in range(num_data):
                    file_name = all_lines[i].split(" ")[0]
                    image_data, old_h, old_w = eval_file(all_lines[i], self.input_shape)
                    scale_w, scale_h = old_w/self.input_shape[0], old_h/self.input_shape[1]
                    startTime = time.time()
                    result = sess.run(net_out, feed_dict={im_inputs: image_data})
                    endTime = time.time()
                    print("detector_time:{}".format(endTime - startTime))
                    decode_boxes, classes = decoder_v2(result, self.dataset.anchors)
                    for box in decode_boxes[0]:
                        image_data[0] = cv2.rectangle(image_data[0], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 125, 35), 2)
                    cv2.imshow("1", image_data[0])
                    cv2.waitKey()
                    #for box, classes in zip(decode_boxes[0], classes[0]):
                    #    file_name += " " + str(box[0]*scale_w) + ',' + str(box[1]*scale_h) + ',' + str(box[2]*scale_w) + ',' + str(box[3]*scale_h) + ',' + str(classes[0])
                    #eval.write(file_name)
                    #eval.write("\n")


if __name__ == '__main__':
    from datasets.VOC_Data import VOC_Data
    import cv2
    annotation_path = "../config/pettrainval.txt"
    classes_path = "../config/pet_classes.txt"
    anchors_path = "../config/yolo2_anchors.txt"
    weight_path = "../model_final/yolov2_mobile.ckpt"
    dataset = VOC_Data(annotation_path, classes_path, anchors_path)
    yolov2 = Yolov2(dataset)
    yolov2.yolo_eval(weight_path=weight_path, annotation_path=annotation_path)
    # yolov2.yolo_train(epochs=1, batch_size=8, weight_path="../model_final/yolov2_mobile.ckpt")

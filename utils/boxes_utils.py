import numpy as np


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
    return keep


def box_transform_x1y1(box_xywh, classes):
    x1y1 = box_xywh[..., :2] - box_xywh[..., 2:4]/2
    x1y1 = np.maximum(np.minimum(x1y1, 416), 0)
    x2y2 = box_xywh[..., :2] + box_xywh[..., 2:4]/2
    x2y2 = np.maximum(np.minimum(x2y2, 416), 0)
    boxes = np.concatenate([x1y1, x2y2, box_xywh[..., 4:5]], axis=-1)
    keep = py_cpu_nms(boxes, 0.5)
    return boxes[keep], classes[keep]


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)


def decoder_v2(boxes, anchors):
    boxes_shape = boxes.shape
    anchors = np.reshape(anchors, [1, 1, -1, 2])
    grid_x = np.tile(np.reshape(np.arange(0, boxes_shape[2]), [-1, 1, 1, 1]), [1, boxes_shape[1], 1, 1])
    grid_y = np.tile(np.reshape(np.arange(0, boxes_shape[1]), [1, -1, 1, 1]), [boxes_shape[2], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    rawboxes = np.reshape(boxes, [boxes_shape[0], boxes_shape[1], boxes_shape[2], 5, -1])
    # y_true = np.reshape(y_true, [boxes_shape[0], boxes_shape[1], boxes_shape[2], 5, -1])
    # y_true_xy = y_true[..., :2]
    # y_true_wh = y_true[..., 2:4]
    raw_xy = sigmoid(rawboxes[..., :2])
    raw_wh = rawboxes[..., 2:4]
    raw_confidences = sigmoid(rawboxes[..., 4])
    raw_cls = np.max(softmax(rawboxes[..., 5:]), axis=-1)
    classes = np.expand_dims(np.argmax(rawboxes[..., 5:], axis=-1), axis=-1)
    raw_score = raw_confidences * raw_cls
    # y_true_conf = y_true[..., 4]
    all_box = []
    all_classes = []
    for i in range(boxes_shape[0]):
        index = raw_score[i] > 0.7
        # true_index = y_true_conf[i] > 0.9
        xy = (raw_xy[i] + grid)
        xy = xy[index] * 32
        wh = np.exp(raw_wh[i]) * anchors
        wh = wh[index]
        score = np.expand_dims(raw_score[i][index], axis=-1)
        classes = classes[i][index]
        # print(score)
        box = np.concatenate([xy, wh, score], axis=-1)
        box, classes = box_transform_x1y1(box, classes)
        all_box.append(box)
        all_classes.append(classes)
    return all_box, all_classes
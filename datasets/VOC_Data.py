import numpy as np
import cv2
import os


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def preprocess_true_boxes_V1(true_boxes, input_shape, num_boxes, num_classes):
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh / input_shape
    m = true_boxes.shape[0]
    grid_shapes = input_shape // 32
    y_true = np.zeros((m, grid_shapes[0], grid_shapes[1], num_boxes, 5), dtype='float32')
    y_classify = np.zeros((m, grid_shapes[0], grid_shapes[1], num_classes), dtype='float32')
    valid_mask = boxes_wh[..., 0] > 0
    for b in range(m):
        xys = true_boxes[b, valid_mask[b]][..., :2]
        whs = true_boxes[b, valid_mask[b]][..., 2:4]
        c = true_boxes[b, valid_mask[b]][..., 4].astype(np.int32)
        for xy, wh in zip(xys, whs):
            i, j = np.floor(xy * grid_shapes).astype("int32")
            y_true[b, i, j, :, 0:2] = xy
            y_true[b, i, j, :, 2:4] = wh
            y_true[b, i, j, :, 4] = 1.
            y_classify[b, i, j, c] = 1.
    y_true = np.reshape(y_true, [m, grid_shapes[0], grid_shapes[1], -1])
    y_true = np.concatenate([y_true, y_classify], axis=-1)
    return y_true


def preprocess_true_boxes_V2(true_boxes, input_shape, num_boxes, num_classes, anchors):
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh
    m = true_boxes.shape[0]
    grid_shapes = input_shape // 32
    y_true = np.zeros((m, grid_shapes[0], grid_shapes[1], num_boxes, 5+num_classes), dtype='float32')
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0
    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)
        xys = true_boxes[b, valid_mask[b]][..., :2]
        whs = np.log(true_boxes[b, valid_mask[b]][..., 2:4]/anchors[0, best_anchor])
        c = true_boxes[b, valid_mask[b]][..., 4].astype(np.int32)
        for xy, wh_ in zip(xys, whs):
            i, j = np.floor(xy * grid_shapes).astype("int32")
            y_true[b, i, j, best_anchor, 0:2] = (xy * grid_shapes) - [i, j]
            y_true[b, i, j, best_anchor, 2:4] = wh_
            y_true[b, i, j, best_anchor, 4] = 1.
            y_true[b, i, j, best_anchor, 5+c] = 1
    return np.reshape(y_true, [m, grid_shapes[0], grid_shapes[1], -1])


def decoder_v1(boxes, input_shape, num_boxes):
    grid_shapes = input_shape // 32
    m = boxes.shape[0]
    rawboxes = boxes[..., :5*num_boxes]
    rawboxes = np.reshape(rawboxes, [m, grid_shapes[0], grid_shapes[1], num_boxes, 5])
    rawcls = boxes[..., 5*num_boxes:]

    raw_xy = rawboxes[..., :2]
    raw_wh = rawboxes[..., 2:4]
    raw_confidences = rawboxes[..., 4]

    all_box = []
    for i in range(m):
        index = raw_confidences[i] > 0.8
        xy = raw_xy[i][index] * input_shape[0]
        wh = raw_wh[i][index] * input_shape[0]
        x1y1 = xy - wh / 2.
        x2y2 = xy + wh / 2.
        box = np.concatenate([x1y1, x2y2], axis=-1)
        all_box.append(box)
    return np.array(all_box)


def decoder_v2(boxes, anchors):
    boxes_shape = boxes.shape
    anchors = np.reshape(anchors, [1, 1, -1, 2])
    grid_x = np.tile(np.reshape(np.arange(0, boxes_shape[2]), [-1, 1, 1, 1]), [1, boxes_shape[1], 1, 1])
    grid_y = np.tile(np.reshape(np.arange(0, boxes_shape[1]), [1, -1, 1, 1]), [boxes_shape[2], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    rawboxes = np.reshape(boxes, [boxes_shape[0], boxes_shape[1], boxes_shape[2], 5, -1])
    raw_xy = sigmoid(rawboxes[..., :2])
    raw_wh = rawboxes[..., 2:4]
    raw_confidences = sigmoid(rawboxes[..., 4])
    all_box = []
    for i in range(boxes_shape[0]):
        index = raw_confidences[i] > 0.9
        xy = (raw_xy[i] + grid)
        xy = xy[index] * 32
        wh = np.exp(raw_wh[i]) * anchors
        wh = wh[index]
        x1y1 = xy - wh / 2.
        x2y2 = xy + wh / 2.
        box = np.concatenate([x1y1, x2y2], axis=-1)
        all_box.append(box)
    return np.array(all_box)


class VOC_Data(object):
    def __init__(self, annotation_path, classes_path, anchors_path, input_shape=(416, 416)):
        self.name = "voc_data"
        self.annotation_path = annotation_path
        self.B = 5
        self.input_shape = input_shape
        self._classes_path = classes_path
        self._anchors_path = anchors_path
        self.classes = self._get_class()
        self.anchors = self._get_anchors()
        self.num_classes = len(self.classes)
        self.read_lines()

    def _get_class(self):
        classes_path = os.path.expanduser(self._classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return np.array(class_names)

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self._anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def read_lines(self):
        with open(self.annotation_path) as f:
            all_lines = f.readlines()
        self.num_data = len(all_lines)
        self.all_lines = all_lines

    def get_data_label(self, annotation_line):
        element = annotation_line.split(" ")
        image = cv2.imread(element[0])
        ih, iw = image.shape[:2]
        box = np.array([np.array(list(map(int, box.split(',')))) for box in element[1:]], dtype=np.float32)
        h, w = self.input_shape
        image = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)
        image_data = image / 255.
        label = np.zeros(self.num_classes)
        class_index = box[:, 4].astype(np.int32)
        label[class_index] = 1
        return image_data, label

    def get_data_box(self, annotation_line, max_boxes=20):
        element = annotation_line.split(" ")
        image = cv2.imread(element[0])
        ih, iw = image.shape[:2]
        box = np.array([np.array(list(map(int, box.split(',')))) for box in element[1:]], dtype=np.float32)
        h, w = self.input_shape
        image = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)
        image_data = image/255.
        box[:, [0, 2]] = box[:, [0, 2]] * w / iw
        box[:, [1, 3]] = box[:, [1, 3]] * h / ih
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data = np.zeros((max_boxes, 5))
        box_data[:len(box)] = box
        return image_data, box_data

    def data_label_generator(self, lines):
        all_images = []
        all_boxes = []
        for line in lines:
            image_data, box_data = self.get_data_label(line)
            all_images.append(image_data)
            all_boxes.append(box_data)
        all_images = np.array(all_images)
        all_boxes = np.array(all_boxes)
        return all_images, all_boxes

    def data_box_generator(self, lines):
        all_images = []
        all_boxes = []
        for line in lines:
            image_data, box_data = self.get_data_box(line)
            all_images.append(image_data)
            all_boxes.append(box_data)
        all_images = np.array(all_images)
        all_boxes = np.array(all_boxes)
        # y_true = preprocess_true_boxes_V1(all_boxes, self.input_shape, self.B, self.num_classes)
        y_true = preprocess_true_boxes_V2(all_boxes, self.input_shape, self.B, self.num_classes, self.anchors)
        return all_images, y_true

    def read_data_label(self, batch_size):
        batch_lines = np.random.choice(self.all_lines, size=batch_size)
        image, label = self.data_label_generator(batch_lines)
        return image, label

    def read_data_box(self, batch_size):
        batch_lines = np.random.choice(self.all_lines, size=batch_size)
        image, box = self.data_box_generator(batch_lines)
        return image, box


if __name__ == '__main__':
    annotation_path = "../config/pettrainval.txt"
    classes_path = "../config/pet_classes.txt"
    anchors_path = "../config/yolo2_anchors.txt"
    dataset = VOC_Data(annotation_path, classes_path, anchors_path)
    images, labels = dataset.read_data_box(1)
    decode_boxes = decoder_v2(labels, dataset.anchors)
    # decode_boxes = decoder_v1(boxes, np.array([416, 416]), 5, 20)
    for i in range(len(images)):
        for box in decode_boxes[i]:
            images[i] = cv2.rectangle(images[i], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,125,35), 2)
            images[i] = cv2.rectangle(images[i], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 125, 35), 2)
        cv2.imshow("1", images[i])
        cv2.waitKey()
    print(0)


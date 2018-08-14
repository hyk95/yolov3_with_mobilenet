import cv2
import numpy as np


def read_file(annotation_path, input_shape):

    with open(annotation_path) as f:
        all_lines = f.readlines()
    num_data = len(all_lines)
    for i in range(num_data):
        element = all_lines[i].split(" ")
        image = cv2.imread(element[0])
        h, w = input_shape
        old_h, old_w = image.shape[:2]
        box = np.array([np.array(list(map(int, box.split(',')))) for box in element[1:]], dtype=np.float32)
        image = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)
        image_data = image / 255.

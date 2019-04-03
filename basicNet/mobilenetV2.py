import tensorflow as tf
slim = tf.contrib.slim
from basicNet.mobilenet import mobilenet_v2


class MobilenetV2(object):
    """network for performing feature extraction"""
    def __init__(self, inputs):
        self.name = "MobilenetV2"
        self.outputs = self.forward(inputs)

    def forward(self, inputs):
        logits, endpoints = mobilenet_v2.mobilenet(inputs, num_classes=1, scope=self.name)
        route_13 = endpoints["layer_18"]
        route_26 = endpoints["layer_14"]
        route_52 = endpoints["layer_7"]
        return route_52, route_26, route_13

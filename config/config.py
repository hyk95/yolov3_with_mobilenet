from core import utils


IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE       = 8
STEPS            = 25000
LR               = 0.001 # if Nan, set 0.0005, 0.0001
DECAY_STEPS      = 100
DECAY_RATE       = 0.9
SHUFFLE_SIZE     = 200
CLASSES          = utils.read_coco_names('./data/class.names')
ANCHORS          = utils.get_anchors('./data/anchors.txt', IMAGE_H, IMAGE_W)
NUM_CLASSES      = len(CLASSES)
EVAL_INTERNAL    = 100
SAVE_INTERNAL    = 500
from core import utils

BATCH_SIZE       = 8
STEPS            = 25000
LR               = 0.001 # if Nan, set 0.0005, 0.0001
DECAY_STEPS      = 1000
DECAY_RATE       = 0.99
SHUFFLE_SIZE     = 250
CLASSES          = utils.read_coco_names('./data/class.names')
ANCHORS          = utils.get_anchors('./data/anchors.txt')
NUM_CLASSES      = len(CLASSES)
EVAL_INTERNAL    = 1000
SAVE_INTERNAL    = 1000
TRAIN_TFRECORD   = "./data/train_data/train.tfrecords"
TEST_TFRECORD    = "./data/val_data/val.tfrecords"
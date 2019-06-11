from config.config import *
from core.dataset import dataset, Parser


def load_train_val_data(train_path, val_path):
    parser_train  = Parser(ANCHORS, NUM_CLASSES, image_size=(416, 416))
    parser_test = Parser(ANCHORS, NUM_CLASSES)
    trainset = dataset(parser_train, train_path, BATCH_SIZE, shuffle=SHUFFLE_SIZE, multi_image_size=False)
    testset = dataset(parser_test, val_path, BATCH_SIZE, shuffle=None)
    return trainset, testset
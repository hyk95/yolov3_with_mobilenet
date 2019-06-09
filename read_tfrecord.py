from config.config import *
from core.dataset import dataset, Parser


def load_train_val_data(train_path, val_path):
    train_tfrecord = train_path
    test_tfrecord = val_path
    parser_train  = Parser(ANCHORS, NUM_CLASSES)
    parser_test = Parser(ANCHORS, NUM_CLASSES)
    trainset = dataset(parser_train, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)
    testset  = dataset(parser_test, test_tfrecord , BATCH_SIZE, shuffle=None)
    return trainset, testset
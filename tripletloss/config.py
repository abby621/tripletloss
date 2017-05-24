import os
import numpy as np
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2
import cv2

WHICH_DATASET = 'mnist'

blob = caffe_pb2.BlobProto()

if WHICH_DATASET == 'tc':
    TARGET_SIZE = 227
    CROP_SZ = 227
    TRAIN_BATCH_SIZE = 30
    TEST_BATCH_SIZE = 30
elif WHICH_DATASET == 'mnist':
    TARGET_SIZE = 28
    CROP_SZ = 28
    TRAIN_BATCH_SIZE = 270
    TEST_BATCH_SIZE = 270

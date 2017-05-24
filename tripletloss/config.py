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
    data = open('/project/focus/datasets/tc_tripletloss/mean.binaryproto', 'rb' ).read()
elif WHICH_DATASET == 'cifar':
    TARGET_SIZE = 32
    CROP_SZ = 32
    TRAIN_BATCH_SIZE = 50
    TEST_BATCH_SIZE = 50
    data = open('/project/focus/datasets/cifar-10/mean.binaryproto', 'rb' ).read()
elif WHICH_DATASET == 'mnist':
    TARGET_SIZE = 28
    CROP_SZ = 28
    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 50
    data = open('/project/focus/datasets/mnist/mnist_mean.binaryproto','rb').read()

blob.ParseFromString(data)
arr = np.array(blobproto_to_array(blob))
NUM_CHANNELS = arr.shape[1]
IM_MEAN = arr.squeeze()

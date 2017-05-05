import os
import numpy as np
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2
import cv2

# # Training lmdb
# TRAIN_FILE = '/project/focus/datasets/tc_tripletloss/train.txt'
# # Test lmdb
# TEST_FILE = '/project/focus/datasets/tc_tripletloss/test.txt'
# # Validation lmdb
# VAL_FILE = '/project/focus/datasets/tc_tripletloss/val.txt'
# # Small sample subset
# SAMPLE_FILE = '/project/focus/datasets/tc_tripletloss/sample.txt'

# Training lmdb
TRAIN_FILE = '/project/focus/datasets/cifar-10/train.txt'
# Test lmdb
TEST_FILE = '/project/focus/datasets/cifar-10/test.txt'
# Validation lmdb
VAL_FILE = '/project/focus/datasets/cifar-10/test.txt'
# Small sample subset
SAMPLE_FILE = '/project/focus/datasets/tc_tripletloss/sample.txt'

# Snapshot iteration
SNAPSHOT_ITERS = 10000

# Max training iteration
MAX_ITERS = 400000

# The number of samples in each minibatch
BATCH_SIZE = 30

# Use flipped images also?
FLIPPED = False

TARGET_SIZE = 32
CROP_SZ = 32

blob = caffe_pb2.BlobProto()
# data = open('/project/focus/datasets/tc_tripletloss/mean.binaryproto', 'rb' ).read()
# blob.ParseFromString(data)
# arr = np.array(blobproto_to_array(blob))
# NUM_CHANNELS = arr.shape[1]
# IM_MEAN = arr[0].mean(1).mean(1)

# data = open('/project/focus/datasets/mnist/mnist_mean.binaryproto','rb').read()
# blob.ParseFromString(data)
# arr = np.array(blobproto_to_array(blob))
# NUM_CHANNELS = arr.shape[1]
# IM_MEAN = arr[0].mean(0)

data = open('/project/focus/datasets/cifar-10/mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(blobproto_to_array(blob))
NUM_CHANNELS = arr.shape[1]
IM_MEAN = arr.reshape((TARGET_SIZE,TARGET_SIZE,NUM_CHANNELS))

TRIPLET_TRAINING = False # if we're not fine tuning with triplet loss, this should be false

TEST_NET = '/project/focus/abby/tripletloss/lenet_deploy.prototxt'
TEST_WEIGHTS = '/project/focus/abby/tripletloss/models/outputs/cifar-10/most_recent.caffemodel'
TEST_LAYER = 'ip1'

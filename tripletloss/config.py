import os
import numpy as np
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2
import cv2
from hoteldata import trainhoteldata, testhoteldata

WHICH_DATASET = 'mnist'

# Snapshot iteration
SNAPSHOT_ITERS = 10000

# Max training iteration
MAX_ITERS = 400000

# Use flipped images also?
FLIPPED = False

# Do triplet loss training?
TRIPLET_TRAINING = False # if we're doing the initial feature embedding, this should be false

if WHICH_DATASET == 'tc':
    TRAIN_FILE = '/project/focus/datasets/tc_tripletloss/train.txt'
    TEST_FILE = '/project/focus/datasets/tc_tripletloss/test.txt'
    VAL_FILE = '/project/focus/datasets/tc_tripletloss/val.txt'
    OUTPUT_DIR = '/project/focus/abby/tripletloss/models/outputs/traffickcam/'
    if TRIPLET_TRAINING:
        SOLVER_PROTOTXT = '/project/focus/abby/tripletloss/models/traffickcam/lenet_tripletloss_solver.prototxt'
        PRETRAINED_MODEL = '/project/focus/abby/tripletloss/models/outputs/traffickcam/most_recent.caffemodel'
    else:
        SOLVER_PROTOTXT = '/project/focus/abby/tripletloss/models/traffickcam/lenet_solver.prototxt'
        PRETRAINED_MODEL = None
    TARGET_SIZE = 256
    CROP_SZ = 224
    TRAIN_BATCH_SIZE = 30
    TEST_BATCH_SIZE = 30
    data = open('/project/focus/datasets/tc_tripletloss/mean.binaryproto', 'rb' ).read()
elif WHICH_DATASET == 'cifar':
    TRAIN_FILE = '/project/focus/datasets/cifar-10/train.txt'
    TEST_FILE = '/project/focus/datasets/cifar-10/test.txt'
    VAL_FILE = '/project/focus/datasets/cifar-10/test.txt'
    OUTPUT_DIR = '/project/focus/abby/tripletloss/models/outputs/cifar/'
    if TRIPLET_TRAINING:
        SOLVER_PROTOTXT = '/project/focus/abby/tripletloss/models/cifar/lenet_tripletloss_solver.prototxt'
        PRETRAINED_MODEL = '/project/focus/abby/tripletloss/models/outputs/cifar/most_recent.caffemodel'
    else:
        SOLVER_PROTOTXT = '/project/focus/abby/tripletloss/models/cifar/lenet_solver.prototxt'
        PRETRAINED_MODEL = None
    TARGET_SIZE = 32
    CROP_SZ = 32
    TRAIN_BATCH_SIZE = 180
    TEST_BATCH_SIZE = 30
    data = open('/project/focus/datasets/cifar-10/mean.binaryproto', 'rb' ).read()
elif WHICH_DATASET == 'mnist':
    TRAIN_FILE = '/project/focus/datasets/mnist/train.txt'
    TEST_FILE = '/project/focus/datasets/mnist/test.txt'
    VAL_FILE = '/project/focus/datasets/mnist/test.txt'
    OUTPUT_DIR = '/project/focus/abby/tripletloss/models/outputs/mnist/'
    if TRIPLET_TRAINING:
        SOLVER_PROTOTXT = '/project/focus/abby/tripletloss/models/mnist/lenet_tripletloss_solver.prototxt'
        PRETRAINED_MODEL = '/project/focus/abby/tripletloss/models/outputs/mnist/most_recent.caffemodel'
    else:
        SOLVER_PROTOTXT = '/project/focus/abby/tripletloss/models/mnist/lenet_solver.prototxt'
        PRETRAINED_MODEL = None
    TARGET_SIZE = 28
    CROP_SZ = 28
    TRAIN_BATCH_SIZE = 270
    TEST_BATCH_SIZE = 30
    data = open('/project/focus/datasets/mnist/mnist_mean.binaryproto','rb').read()

# grab the data, organized how we'll need it for selecting triplets (the lmdb will handle the data at the data layer)
TRAINING_DATA = trainhoteldata()
TEST_DATA = testhoteldata()

# Grab image mean
blob = caffe_pb2.BlobProto()
blob.ParseFromString(data)
arr = np.array(blobproto_to_array(blob))
NUM_CHANNELS = arr.shape[1]
if NUM_CHANNELS == 1:
    IM_MEAN = arr.reshape((TARGET_SIZE,TARGET_SIZE))
else:
    IM_MEAN = arr.reshape((TARGET_SIZE,TARGET_SIZE,NUM_CHANNELS))

global CURRENT_NET

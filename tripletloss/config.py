import os
from numpy import *

# Training image data path
IMAGEPATH = '/home/seal/dataset/fast-rcnn/caffe-fast-rcnn/data/Facedevkit/tripletloss/'

# Training lmdb
TRAIN_FILE = '/project/focus/datasets/tc_tripletloss/train.txt'
# Test lmdb
TEST_FILE = '/project/focus/datasets/tc_tripletloss/test.txt'
# Validation lmdb
VAL_FILE = '/project/focus/datasets/tc_tripletloss/val.txt'

# Snapshot iteration
SNAPSHOT_ITERS = 10000

# Max training iteration
MAX_ITERS = 400000

# The number of samples in each minibatch
BATCH_SIZE = 30

# Use flipped images also?
FLIPPED = True

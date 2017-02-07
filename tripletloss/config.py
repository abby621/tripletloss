import os
import numpy as np
from caffe.io import blobproto_to_array
from caffe.proto import caffe_pb2

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

blob = caffe_pb2.BlobProto()
data = open('/project/focus/datasets/tc_tripletloss/mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(blobproto_to_array(blob))
mean_arr = arr[0]
IM_MEAN = cv2.resize(mean_arr, (256,256),
                interpolation=cv2.INTER_LINEAR)

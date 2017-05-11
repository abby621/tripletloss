# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
   The layer combines the input image into triplet.Priority select the semi-hard samples
"""
import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import math
import config
import json
import lmdb
import random
from blob import prep_im_for_blob, im_list_to_blob

# TODO: Grab triplets on the fly here.
def im_paths_to_blob(paths):
    im_blob = []
    for im_path in paths:
        if config.NUM_CHANNELS == 1:
            im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
        else:
            im = cv2.imread(im_path)
        im = prep_im_for_blob(im)
        anchor_im_blob.append(im)
    # Create a blob to hold the input images
    im_data = im_list_to_blob(im_blob)
    return im_data

class TripletSelectLayer(caffe.Layer):

    def setup(self, bottom, top):
        param = json.loads(self.param_str)
        self.phase = param['phase']

        if self.phase == 'TRAIN':
            self.triplet_data = config.TRAINING_DATA
            self.triplet = config.TRAIN_BATCH_SIZE/3
        else:
            self.triplet_data = config.TEST_DATA
            self.triplet = config.TEST_BATCH_SIZE/3

        # randomly select our anchors from the data in this batch
        random_anchors = random.sample(len(self.triplet_data._im_labels),self.triplet)

        anchor_im_paths = [self.triplet_data._im_paths[a] for a in random_anchors]
        anchor_labels = [self.triplet_data._im_labels[a] for a in random_anchors]
        anchor_im_data = im_paths_to_blob(anchor_im_paths)

        ## TODO: update to select + and - examples based on distance
        possible_positives = [np.where(self.triplet_data._im_labels==a)[0] for a in anchor_labels]
        positive_im_inds = [random.choice(ind) for ind in possible_positives]
        positive_im_paths = [self.triplet_data._im_paths[a] for a in positive_im_inds]
        positive_labels = [self.triplet_data._im_labels[a] for a in positive_im_inds]
        positive_im_data = im_paths_to_blob(positive_im_paths)

        possible_negatives = [np.where(self.triplet_data._im_labels!=a)[0] for a in anchor_labels]
        negative_im_inds = [random.choice(ind) for ind in possible_negatives]
        negative_im_paths = [self.triplet_data._im_paths[a] for a in negative_im_inds]
        negative_labels = [self.triplet_data._im_labels[a] for a in negative_im_inds]
        negative_im_data = im_paths_to_blob(negative_im_paths)

        print anchor_labels, positive_labels, negative_labels

        """Setup the TripletSelectLayer."""

        top[0].reshape(self.triplet,shape(bottom[0].data)[1])
        top[1].reshape(self.triplet,shape(bottom[0].data)[1])
        top[2].reshape(self.triplet,shape(bottom[0].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        top_anchor = []
        top_positive = []
        top_negative = []
        labels = []
        self.tripletlist = []
        self.no_residual_list=[]
        aps = {}
        ans = {}

        anchor_feature = bottom[0].data[0]
        for i in range(self.triplet):
            positive_feature = bottom[0].data[i+self.triplet]
            a_p = anchor_feature - positive_feature
            ap = np.dot(a_p,a_p)
            aps[i+self.triplet] = ap
        aps = sorted(aps.items(), key = lambda d: d[1], reverse = True)
        for i in range(self.triplet):
            negative_feature = bottom[0].data[i+self.triplet*2]
            a_n = anchor_feature - negative_feature
            an = np.dot(a_n,a_n)
            ans[i+self.triplet*2] = an
        ans = sorted(ans.items(), key = lambda d: d[1], reverse = True)

        for i in range(self.triplet):
            top_anchor.append(bottom[0].data[i])
            top_positive.append(bottom[0].data[aps[i][0]])
            top_negative.append(bottom[0].data[ans[i][0]])
            if aps[i][1] >= ans[i][1]:
               self.no_residual_list.append(i)
            self.tripletlist.append([i,aps[i][0],ans[i][0]])

        top[0].data[...] = np.array(top_anchor).astype(float32)
        top[1].data[...] = np.array(top_positive).astype(float32)
        top[2].data[...] = np.array(top_negative).astype(float32)

    def backward(self, top, propagate_down, bottom):
        for i in range(len(self.tripletlist)):
            if not i in self.no_residual_list:
                bottom[0].diff[self.tripletlist[i][0]] = top[0].diff[i]
                bottom[0].diff[self.tripletlist[i][1]] = top[1].diff[i]
                bottom[0].diff[self.tripletlist[i][2]] = top[2].diff[i]
            else:
                bottom[0].diff[self.tripletlist[i][0]] = np.zeros(shape(top[0].diff[i]))
                bottom[0].diff[self.tripletlist[i][1]] = np.zeros(shape(top[1].diff[i]))
                bottom[0].diff[self.tripletlist[i][2]] = np.zeros(shape(top[2].diff[i]))

        #print 'backward-no_re:',bottom[0].diff[0][0]
        #print 'tripletlist:',self.no_residual_list

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

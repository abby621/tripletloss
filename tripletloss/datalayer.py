# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training to train the network.
   This is a example for online triplet selection
   Each minibatch contains a set of anchor-positive pairs, random select negative exemplar
"""

import caffe
import numpy as np
from numpy import *
import yaml
from hoteldata import hoteldata
import random
import cv2
from blob import prep_im_for_blob, im_list_to_blob
import config
import pickle
import os, commands
from scipy.stats import norm

def get_features(im,net):
    net.blobs['data'].reshape(1,3,config.CROP_SZ,config.CROP_SZ)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', config.IM_MEAN)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    orig_im = caffe.io.load_image(im)
    caffe_input = transformer.preprocess('data',orig_im)
    net.blobs['data'].data[...] = caffe_input
    out = net.forward()
    feats = net.blobs[config.TEST_LAYER].data.copy()
    return feats

def feat_dist(f1,f2):
    feat_dist = (f1 - f2)**2
    feat_dist = np.sum(feat_dist,axis=1)
    feat_dist = np.sqrt(feat_dist)
    return feat_dist

class DataLayer(caffe.Layer):
    """Sample data layer used for training."""

    def _get_next_minibatch(self):
        num_images = self._batch_size

        # Sample to use for each image in this batch
        sample = []
        sample_labels = []

        # recompute our triplet distribution function every 10k images
        if self._index % 10000 == 0 and self._index != 0 and config.TRIPLET_TRAINING:
            print 'Copying the most recent snapshot to the most_recent path....'
            # copy the most recent snapshot to the 'most_recent.caffemodel' location
            status, most_recent = commands.getstatusoutput("ls -dtr1 /project/focus/abby/tripletloss/models/outputs/places_cnds/*.caffemodel | tail -1")
            cp_cmd = 'cp ' + most_recent + ' /project/focus/abby/tripletloss/models/outputs/places_cnds/most_recent.caffemodel'
            os.system(cp_cmd)
            # run the code to find and save the new triplet distribution parameters
            os.system("python /project/focus/abby/tripletloss/tripletloss/find_triplet_params.py /project/focus/abby/tripletloss/places_cnds_train.prototxt /project/focus/abby/tripletloss/models/outputs/places_cnds/most_recent.caffemodel")

        # load the triplet parameters and generate the distributions
        if config.TRIPLET_TRAINING:
            stat_file = '/project/focus/abby/tripletloss/params/triplet_stats.pickle'
            with open(stat_file,'rb') as f:
                triplet_stats = pickle.load(f)
            f.close()
            pos_norm = norm(loc=triplet_stats['pos_mean'],scale=triplet_stats['pos_std'])
            neg_norm = norm(los=triplet_stats['neg_mean'],scale=triplet_stats['neg_std'])

        num_ims = len(self.data_container._train_im_paths)
        positive_examples = []
        negative_examples = []
        while len(positive_examples) < self._triplet or len(negative_examples) < self._triplet:
            positive_examples = []
            negative_examples = []

            anchor_im_path = self.data_container._train_im_paths[self._index]
            anchor_im_label = self.data_container._train_im_labels[self._index]
            if config.TRIPLET_TRAINING:
                anchor_im_feat = get_features(anchor_im_path,self.test_net)

            # include a candidate as a positive example if:
            # it is from the same hotel
            # it is not the exact same image
            # roughly, this image is in the self._pos_thresh closest positive images
            pos_ctr = 0
            while len(positive_examples) < self._triplet and pos_ctr < num_ims:
                if self.data_container._train_im_labels[pos_ctr]==anchor_im_label and pos_ctr != self._index:
                    if config.TRIPLET_TRAINING:
                        pos_feat = get_features(self.data_container._train_im_paths[pos_ctr],self.test_net)
                        pos_dist = feat_dist(anchor_im_feat,pos_feat)
                        pos_score = pos_norm.cdf(pos_dist)
                        if pos_score < self._pos_thresh:
                            positive_examples.append(pos_ctr)
                    else:
                        positive_examples.append(pos_ctr)
                pos_ctr += 1

            # include a candidate as a negative example if:
            # it is from a different hotel
            # roughly, this image is in the self._neg_thresh farthest images
            neg_ctr = 0
            while len(negative_examples) < self._triplet and neg_ctr < num_ims:
                if self.data_container._train_im_labels[neg_ctr]!=anchor_im_label and neg_ctr != self._index:
                    if config.TRIPLET_TRAINING:
                        neg_feat = get_features(self.data_container._train_im_paths[neg_ctr],self.test_net)
                        neg_dist = feat_dist(anchor_im_feat,neg_feat)
                        neg_score = neg_norm.cdf(neg_dist)
                        if neg_score > self._neg_thresh:
                            negative_examples.append(neg_ctr)
                    else:
                        negative_examples.append(neg_ctr)
                neg_ctr += 1

            self._index = self._index + 1
            if self._index >= len(self.data_container._train_im_paths):
                self._index = 0
                self._epoch += 1
                # Change self._pos_thresh and self._neg_thresh to make the task more challenging w/ each epoch
                if self._pos_thresh + 0.05 <= 1: self._pos_thresh += 0.05
                if self._neg_thresh - 0.05 >= 0: self._neg_thresh -= 0.05

        while len(sample) < self._triplet:
            sample.append(anchor_im_path)
            sample_labels.append(anchor_im_label)

        # Sample positive examples
        for p in positive_examples:
            sample.append(self.data_container._train_im_paths[p])
            sample_labels.append(self.data_container._train_im_labels[p])

        # Sample negative examples
        for n in negative_examples:
            sample.append(self.data_container._train_im_paths[n])
            sample_labels.append(self.data_container._train_im_labels[n])

        im_blob = self._get_image_blob(sample)
        blobs = {'data': im_blob,
             'labels': sample_labels}
        return blobs

    def _get_image_blob(self,sample):
        im_blob = []
        labels_blob = []
        for i in range(len(sample)):
            im = cv2.imread(sample[i])
            im = prep_im_for_blob(im)
            im_blob.append(im)
        # Create a blob to hold the input images
        blob = im_list_to_blob(im_blob)
        return blob

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
        # parse the layer parameter string, which must be valid YAML
        # layer_params = yaml.load(self.param_str_)
        self._batch_size = config.BATCH_SIZE
        self._triplet = self._batch_size/3
        assert self._batch_size % 3 == 0
        self._name_to_top_map = {
            'data': 0,
            'labels': 1}

        self.data_container =  hoteldata()
        self._index = 0
        self._epoch = 0

        self._pos_thresh = 0.45
        self._neg_thresh = 0.65

        # load this net for triplet computation
        if config.TRIPLET_TRAINING == True:
            self.test_net = caffe.Net(config.TEST_NET, config.TEST_WEIGHTS, caffe.TEST)

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(self._batch_size, config.NUM_CHANNELS, config.TARGET_SIZE, config.TARGET_SIZE)

        top[1].reshape(self._batch_size)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            #top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

if __name__ == '__main__':
    #print data_container._sample
    test = DataLayer()
    for i in range(500):
        blob = test._get_next_minibatch()

# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training to train the network.
   This is a example for online triplet selection
   Each minibatch contains a set of archor-positive pairs, random select negative exemplar
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


class DataLayer(caffe.Layer):
    """Sample data layer used for training."""

    def _get_next_minibatch(self):
        num_images = self._batch_size

        # Sample to use for each image in this batch
        sample = []
        sample_labels = []
        if self._index >= len(self.data_container._train_im_paths):
            self._index = 0

        train_im_path = self.data_container._train_im_paths[self._index]
        train_im_label = self.data_container._train_im_labels[self._index]

        self._index = self._index + 1
        while len(sample) < self._triplet:
            sample.append(train_im_path)
            sample_labels.append(train_im_label)

        positive_examples = [i for i,x in enumerate(self.data_container._train_im_labels) if x==train_im_label and i!=self._index]
        negative_examples = [i for i,x in enumerate(self.data_container._train_im_labels) if x!=train_im_label]

        # Sample positive samples
        while len(sample) < self._triplet*2:
            pos_index = random.randint(0,len(positive_examples)-1)
            if (self.data_container._train_im_paths[positive_examples[pos_index]]) not in sample:
                sample.append(self.data_container._train_im_paths[positive_examples[pos_index]])
                sample_labels.append(self.data_container._train_im_labels[positive_examples[pos_index]])

        # Sample negative samples
        while len(sample) < self._triplet*3:
            neg_index = random.randint(0,len(negative_examples)-1)
            if (self.data_container._train_im_paths[negative_examples[neg_index]]) not in sample:
                sample.append(self.data_container._train_im_paths[negative_examples[neg_index]])
                sample_labels.append(self.data_container._train_im_labels[negative_examples[neg_index]])

        im_blob = self._get_image_blob(sample)
        #print sample
        blobs = {'data': im_blob,
             'labels': sample_labels}
        return blobs

    def _get_image_blob(self,sample):
        im_blob = []
        labels_blob = []
        for i in range(self._batch_size):
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

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(self._batch_size, 3, 224, 224)

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

class TestBlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self):
        self._batch_size = 30
        self.data_container =  hoteldata()
        self._index = 0

    def _get_next_minibatch(self):
        num_images = self._batch_size

        # Sample to use for each image in this batch
        sample = []
        sample_labels = []
        if self._index >= len(self.data_container._train_im_paths):
            self._index = 0

        train_im_path = self.data_container._train_im_paths[self._index]
        train_im_label = self.data_container._train_im_labels[self._index]

        self._index = self._index + 1
        while len(sample) < self._triplet:
            sample.append(train_im_path)
            sample_labels.append(train_im_label)

        positive_examples = [i for i,x in enumerate(self.data_container._train_im_labels) if x==train_im_label and i!=self._index]
        negative_examples = [i for i,x in enumerate(self.data_container._train_im_labels) if x!=train_im_label]

        # Sample positive samples
        while len(sample) < self._triplet*2:
            pos_index = random.randint(0,len(positive_examples)-1)
            if (self.data_container._train_im_paths[positive_examples[pos_index]]) not in sample:
                sample.append(self.data_container._train_im_paths[positive_examples[pos_index]])
                sample_labels.append(self.data_container._train_im_labels[positive_examples[pos_index]])

        # Sample negative samples
        while len(sample) < self._triplet*3:
            neg_index = random.randint(0,len(negative_examples)-1)
            if (self.data_container._train_im_paths[negative_examples[neg_index]]) not in sample:
                sample.append(self.data_container._train_im_paths[negative_examples[neg_index]])
                sample_labels.append(self.data_container._train_im_labels[negative_examples[neg_index]])

        # TODO: Send labels to _get_image_blob, if necessary?
        im_blob = self._get_image_blob(sample)
        #print sample
        blobs = {'data': im_blob,
             'labels': sample_labels}
        return blobs

    def _get_image_blob(self,sample):
        im_blob = []
        labels_blob = []
        for i in range(self._batch_size):
            im = cv2.imread(sample[i])
            im = prep_im_for_blob(im)
            im_blob.append(im)
        # Create a blob to hold the input images
        blob = im_list_to_blob(im_blob)
        return blob

if __name__ == '__main__':
    #print data_container._sample
    test = TestBlobFetcher()
    for i in range(500):
        blob = test._get_next_minibatch()
        print blob["labels"]
        print str(i),np.shape(blob["data"]),blob["labels"]#,blob

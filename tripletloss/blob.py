# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
import config as config

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], config.NUM_CHANNELS),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im.reshape((max_shape[0], max_shape[1], config.NUM_CHANNELS))
    if config.NUM_CHANNELS == 3:
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im):
    im = im.astype(np.float32, copy=False)
    im -= config.IM_MEAN
    im = cv2.resize(im, (config.TARGET_SIZE,config.TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    return im

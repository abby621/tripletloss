# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
import config
import random
from PIL import Image
import string

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, config.NUM_CHANNELS, max_shape[0], max_shape[1]),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        print np.mean(im)
        do_save = random.random() > .95
        if do_save:
            save_path = '/project/focus/abby/tripletloss/example_ims/test/' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) + '.jpg'
            pil_im = Image.fromarray(im).convert('RGB')
            pil_im.save(save_path)
        blob[i, :, 0:im.shape[0], 0:im.shape[1]] = im.reshape((config.NUM_CHANNELS,max_shape[0], max_shape[1]))
    # if config.NUM_CHANNELS == 3:
    #     channel_swap = (0, 3, 1, 2)
    #     blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im):
    im = im.astype(np.float32, copy=False)
    im2 = im - config.IM_MEAN
    # im = cv2.resize(im, (config.TARGET_SIZE,config.TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    return im2

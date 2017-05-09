import os
import config as cfg
import numpy as np
import csv

class hoteldata(phase):
    global _train_im_paths
    global _train_im_labels

    def __init__(self):
        if self.phase == 'train':
            with open(cfg.TRAIN_FILE,'rU') as tf:
                readerT = csv.reader(tf, delimiter=' ')
                trainIms = list(readerT)
            self._train_im_paths = [i[0] for i in trainIms]
            self._train_im_labels = [i[1] for i in trainIms]
        elif self.phase == 'test':
            with open(cfg.TEST_FILE,'rU') as tf:
                readerT = csv.reader(tf, delimiter=' ')
                trainIms = list(readerT)
            self._train_im_paths = [i[0] for i in trainIms]
            self._train_im_labels = [i[1] for i in trainIms]

        if cfg.FLIPPED:
            flipped_train_ims = [i.split('.')[0]+'_flip.jpg' for i in self._train_im_paths]
            self._train_im_paths.extend(flipped_train_ims)
            self._train_im_labels.extend(self._train_im_labels)

if __name__ == '__main__':
    hotel = hoteldata()

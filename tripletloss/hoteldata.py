import os
import config as cfg
import numpy as np
import csv

class hoteldata():
    global _train_im_paths
    global _train_im_labels
    global _val_im_paths
    global _val_labels

    def __init__(self):
        with open(cfg.SAMPLE_FILE,'rU') as tf:
            readerT = csv.reader(tf, delimiter=' ')
            trainIms = list(readerT)
        self._train_im_paths = [i[0] for i in trainIms]
        self._train_im_labels = [i[1] for i in trainIms]

        with open(cfg.VAL_FILE,'rU') as vf:
            readerV = csv.reader(vf, delimiter=' ')
            valIms = list(readerV)
        self._val_im_paths = [i[0] for i in valIms]
        self._val_labels = [i[1] for i in valIms]

        if cfg.FLIPPED:
            flipped_train_ims = [i.split('.')[0]+'_flip.jpg' for i in self._train_im_paths]
            self._train_im_paths.extend(flipped_train_ims)
            self._train_im_labels.extend(self._train_im_labels)

            flipped_val_ims = [i.split('.')[0]+'_flip.jpg' for i in self._val_im_paths]
            self._val_im_paths.extend(flipped_val_ims)
            self._val_labels.extend(self._val_labels)

if __name__ == '__main__':
    hotel = hoteldata()

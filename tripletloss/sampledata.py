import os
import config as cfg
import numpy as np
import csv

# class sampledata():
#
#     def __init__(self):
#         self._sample_person = {}
#         self._sample = []
#         self._sample_label = {}
#         self._sample_test = []
#         face_path = cfg.IMAGEPATH
#
#         for num, personname in enumerate(sorted(os.listdir(face_path))):
#             person_path = face_path + personname + '/face'
#             picnames = [{'picname': personname + '/face/' + i, 'flipped': False}
#                         for i in sorted(os.listdir(person_path))
#                         if os.path.getsize(os.path.join(person_path, i)) > 0]
#             pic_train = int(len(picnames) * cfg.PERCENT)
#             self._sample_person[personname] = picnames[:pic_train]
#             self._sample_label[personname] = num
#             self._sample.extend(picnames[:pic_train])
#             self._sample_test.extend(picnames[pic_train:])
#
#             if cfg.FLIPPED:
#                 picnames_flipped = [{'picname': i['picname'], 'flipped': True}
#                                     for i in picnames[:pic_train]]
#                 self._sample_person[personname].extend(picnames_flipped)
#                 self._sample.extend(picnames_flipped)
#
#         print 'Number of training persons: {}'.format(len(self._sample_person))
#         print 'Number of training images: {}'.format(len(self._sample))
#         print 'Number of testing images: {}'.format(len(self._sample_test))

class sampledata():
    def __init__(self):
        with open(cfg.TRAIN_FILE,'rU') as tf:
            readerT = csv.reader(tf, delimiter=' ')
            trainIms = list(readerT)
        self.train_im_paths = [i[0] for i in trainIms]
        self.train_labels = [i[1] for i in trainIms]

        with open(cfg.VAL_FILE,'rU') as vf:
            readerV = csv.reader(vf, delimiter=' ')
            valIms = list(readerV)
        self.val_im_paths = [i[0] for i in valIms]
        self.val_labels = [i[1] for i in valIms]

        if cfg.FLIPPED:
            flipped_train_ims = [i.split('.')[0]+'_flip.jpg' for i in self.train_im_paths]
            self.train_im_paths.extend(flipped_train_ims)
            self.train_labels.extend(self.train_labels)

            flipped_val_ims = [i.split('.')[0]+'_flip.jpg' for i in self.val_im_paths]
            self.val_im_paths.extend(flipped_val_ims)
            self.val_labels.extend(self.val_labels)

if __name__ == '__main__':
    sample = sampledata()

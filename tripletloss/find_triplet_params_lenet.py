import caffe
import numpy as np
import csv
from lshash import LSHash
from sklearn.decomposition import IncrementalPCA
import random
from PIL import Image
import sys
import multiprocessing
import pickle
import os, sys
import config

"""
Command: python find_triplet_params.py /project/focus/abby/tripletloss/params/triplet_stats_lenet.pickle /project/focus/abby/tripletloss/lenet_train_test.prototxt /project/focus/abby/tripletloss/models/outputs/mnist/most_recent.caffemodel
"""
help = """
    Command format: python find_triplet_params.py stat_file net_file weights_file
    stat_file: the .pickle file to write the triplet statistics out to
    net_file: the file name for the network definition
    weights_file: the file name for the caffemodel weights
    """

def getFeatures(im,net,featLayer,config.IM_MEAN):
    net.blobs['data'].reshape(1,config.NUM_CHANNELS,config.CROP_SZ,config.CROP_SZ)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', config.IM_MEAN)
    if config.NUM_CHANNELS == 3:
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    if config.NUM_CHANNELS == 3:
        orig_im = caffe.io.load_image(im)
    else:
        orig_im = caffe.io.load_image(im,color=False)
    caffe_input = transformer.preprocess('data',orig_im)
    net.blobs['data'].data[...] = caffe_input
    out = net.forward()
    feats = net.blobs[featLayer].data.copy()
    return feats

def get_triplet_stats(stat_file,net_model,net_weights):
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # net_model = '/project/focus/abby/hotelnet/models/places_cnds/deploy.prototxt';

    # net_weights = '/project/focus/abby/hotelnet/models/places_cnds/places_cdns.caffemodel'
    net = caffe.Net(net_model, net_weights, caffe.TEST);

    im_file = '/project/focus/datasets/mnist/train.txt'
    with open(im_file,'rU') as f:
        rd = csv.reader(f,delimiter=' ')
        im_list = list(rd)

    ims_by_class = {}
    for i in im_list:
        if not i[1] in ims_by_class:
            ims_by_class[i[1]] = []
        ims_by_class[i[1]].append(i[0])

    class_keys = ims_by_class.keys()
    bad_classes = []
    for h in class_keys:
        if len(ims_by_class[h]) < 2:
            bad_classes.append(h)

    good_classes = [h for h in class_keys if h not in bad_classes]

    random_triplets = []

    for r in good_classes:
        pos_pair = random.sample(ims_by_class[r],2)
        neg_example = random.choice(class_keys)
        while neg_example == r:
            neg_example = random.choice(class_keys)
        neg_im = random.choice(ims_by_class[neg_example])
        random_triplets.append((pos_pair[0],pos_pair[1],neg_im))

    featLayer = 'ip2'

    pos_dists = []
    neg_dists = []
    for triplet in random_triplets:
        anchor_feats = getFeatures(triplet[0],net,config.CROP_SZ,featLayer,config.IM_MEAN)
        pos_feats = getFeatures(triplet[1],net,config.CROP_SZ,featLayer,config.IM_MEAN)
        neg_feats = getFeatures(triplet[2],net,config.CROP_SZ,featLayer,config.IM_MEAN)

        posDist = (anchor_feats - pos_feats)**2
        posDist = np.sum(posDist,axis=1)
        posDist = np.sqrt(posDist)
        pos_dists.append(posDist)

        negDist = (anchor_feats - neg_feats)**2
        negDist = np.sum(negDist,axis=1)
        negDist = np.sqrt(negDist)
        neg_dists.append(negDist)

    pos_dists = np.asarray(pos_dists)
    pos_mean = np.mean(pos_dists)
    pos_std = np.std(pos_dists)
    pos_var = np.var(pos_dists)

    neg_dists = np.asarray(neg_dists)
    neg_mean = np.mean(neg_dists)
    neg_std = np.std(neg_dists)
    neg_var = np.var(neg_dists)

    triplet_stats = {'pos_mean':pos_mean,'pos_std':pos_std,'neg_mean':neg_mean,'neg_std':neg_std}

#     stat_file = '/project/focus/abby/tripletloss/params/triplet_stats.pickle'
    if os.path.exists(stat_file):
        os.remove(stat_file)

    with open(stat_file,'wb') as f:
        pickle.dump(triplet_stats,f,protocol=pickle.HIGHEST_PROTOCOL)

    f.close()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise Exception(help)

    stat_file = sys.argv[1]
    net_model = sys.argv[2]
    net_weights = sys.argv[3]

    get_triplet_stats(stat_file,net_model,net_weights)
# import pickle
# stat_file = '/project/focus/abby/tripletloss/params/triplet_stats.pickle'
# with open(stat_file,'rb') as f:
#     triplet_stats = pickle.load(f)
#
# f.close()
#
# from scipy.stats import norm
# pos_norm = norm(loc=triplet_stats['pos_mean'],scale=triplet_stats['pos_std'])
# neg_norm = norm(los=triplet_stats['neg_mean'],scale=triplet_stats['neg_std'])
#
# pos_norm.cdf(12)
# 1 - neg_norm.cdf(12)

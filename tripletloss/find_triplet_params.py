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

"""
Command: python find_triplet_params.py /project/focus/abby/tripletloss/params/triplet_stats.pickle /project/focus/abby/hotelnet/models/places_cnds/deploy.prototxt /project/focus/abby/hotelnet/models/places_cnds/places_cdns.caffemodel
"""
help = """
    Command format: python find_triplet_params.py stat_file net_file weights_file
    stat_file: the .pickle file to write the triplet statistics out to
    net_file: the file name for the network definition
    weights_file: the file name for the caffemodel weights
    """

def getFeatures(im,net,cropSz,featLayer,meanIm):
    net.blobs['data'].reshape(1,3,cropSz,cropSz)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', meanIm)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    orig_im = caffe.io.load_image(im)
    caffe_input = transformer.preprocess('data',orig_im)
    net.blobs['data'].data[...] = caffe_input
    out = net.forward()
    feats = net.blobs[featLayer].data.copy()
    return feats
    
def get_triplet_stats(stat_file,net_model,net_weights):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    # net_model = '/project/focus/abby/hotelnet/models/places_cnds/deploy.prototxt';
    cropSz = 227
    batchSz = 10
    
    # net_weights = '/project/focus/abby/hotelnet/models/places_cnds/places_cdns.caffemodel'
    net = caffe.Net(net_model, net_weights, caffe.TEST);
    
    im_file = '/project/focus/datasets/tc_tripletloss/train.txt'
    with open(im_file,'rU') as f:
        rd = csv.reader(f,delimiter=' ')
        im_list = list(rd)
    
    ims_by_hotel = {}
    for i in im_list:
        if not i[1] in ims_by_hotel:
            ims_by_hotel[i[1]] = []
        ims_by_hotel[i[1]].append(i[0])
    
    hotel_keys = ims_by_hotel.keys()
    bad_hotels = []
    for h in hotel_keys:
        if len(ims_by_hotel[h]) < 2:
            bad_hotels.append(h)
    
    good_hotels = [h for h in hotel_keys if h not in bad_hotels]
    
    random_anchor_hotels = random.sample(good_hotels,1000)
    random_triplets = []
    
    for r in random_anchor_hotels:
        pos_pair = random.sample(ims_by_hotel[r],2)
        neg_hotel = random.choice(hotel_keys)
        while neg_hotel == r:
            neg_hotel = random.choice(hotel_keys)
        neg_im = random.choice(ims_by_hotel[neg_hotel])
        random_triplets.append((pos_pair[0],pos_pair[1],neg_im))
    
    meanIm_path = '/project/focus/abby/hotelnet/models/places205CNN_mean.binaryproto'
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(meanIm_path,'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    meanIm = arr[0].mean(1).mean(1)
    
    featLayer = 'fc7'
    
    pos_dists = []
    neg_dists = []
    for triplet in random_triplets:
        anchor_feats = getFeatures(triplet[0],net,cropSz,featLayer,meanIm)
        pos_feats = getFeatures(triplet[1],net,cropSz,featLayer,meanIm)
        neg_feats = getFeatures(triplet[2],net,cropSz,featLayer,meanIm)
    
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

import caffe
import lmdb
import os
import scipy.misc

dataset_dir = '/project/focus/datasets/mnist/'
lmdb_path = os.path.join(dataset_dir,'mnist_train_lmdb')
txt_path = os.path.join(dataset_dir,'train.txt')
if os.path.exists(txt_path):
    os.remove(txt_path)

txt_file = open(txt_path,'a')
im_path = os.path.join(dataset_dir,'train')
if not os.path.exists(im_path):
    os.makedirs(im_path)

lmdb_env = lmdb.open(lmdb_path)
lmdb_txt = lmdb_env.begin()
lmdb_cursor = lmdb_txt.cursor()
datum = caffe.proto.caffe_pb2.Datum()

ctr = 0
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    save_dir = os.path.join(im_path,str(label))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir,str(ctr)+'.jpg')
    data = caffe.io.datum_to_array(datum)
    pil_image = Image.fromarray(data.squeeze())
    pil_image.save(save_path)
    print '%s %s' % (save_path,str(label))
    txt_file.write('%s %s\n' % (save_path,str(label)))
    ctr += 1

txt_file.close()

import sys, os
import cv2
import time
import numpy as np
from params import get_params
import pickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

params     = get_params()

sleep_time = params['sleep_time']
disp_n     = params['disp_n']

# Add Faster R-CNN module to pythonpath
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'caffe-fast-rcnn', 'python'))
sys.path.insert(0, os.path.join(params['fast_rcnn_path'],'lib'))

import caffe
from fast_rcnn.config import cfg
import test as test_ops


def learn_transform(params, feats):
  '''Norm, PCA, Whitten'''
  feats = normalize(feats)
  pca   = PCA(params['dimension'], whiten=True)

  pca.fit(feats)

  pca_model_pkl = params['pca_model'] + '_' + params['dataset'] + '.pkl'
  print "\npca model file:", pca_model_pkl, "\n"

  pickle.dump(pca, open(pca_model_pkl ,'wb'))


class Extractor():

  def __init__(self, params):

    self.dimension = params['dimension']
    self.dataset   = params['dataset']
    self.pooling   = params['pooling']

    print "\n"
    print "dimension:", self.dimension
    print "dataset:",   self.dataset
    print "pooling:",   self.pooling
    print "\n"

    # Read query image lists
    with open(params['query_list'], 'r') as f:
      self.query_names = f.read().splitlines()
    print "len of queries:", len(self.query_names)
    print "example:",        self.query_names[0]
    print "\n"

    # Read query image lists
    with open(params['frame_list'], 'r') as f:
      self.database_list = f.read().splitlines()
    print "len of database:", len(self.database_list)
    print "example:",         self.database_list[0]
    print "\n"

    # Parameters needed
    self.layer         = params['layer']
    self.save_db_feats = params['database_feats']
    print "extracted layer name:", self.layer
    print "save_db_feats:",        self.save_db_feats
    print "\n"

    # Init network
    if params['gpu']:
      caffe.set_mode_gpu()
      caffe.set_device(0)
    else:
      caffe.set_mode_cpu()
    print "Extracting from:", params['net_proto']


    print "\n"
    print "init network from pre-trained model"
    cfg.TEST.HAS_RPN = True  # using rpn to generate props
    self.net         = caffe.Net(params['net_proto'], params['net'], caffe.TEST)
    print "init network done"
    print "\n"
    time.sleep(sleep_time)

  def extract_feat_image(self,image):

    im            = cv2.imread(image)
    scores, boxes = test_ops.im_detect(self.net, im, boxes=None)
    feat          = self.net.blobs[self.layer].data

    return feat


  def pool_feats(self,feat):

    if self.pooling is 'max':
      feat = np.max(np.max(feat, axis=2), axis=1)
    else:
      feat = np.sum(np.sum(feat, axis=2), axis=1)

    return feat

  def save_feats_to_disk(self):

    print "Extracting database features..."
    t0        = time.time()
    disp_c    = 0

    # Init empty np array to store all databsae features
    n_ims     = len(self.database_list)
    dimension = self.dimension
    xfeats    = np.zeros((n_ims, dimension))

    for frame in self.database_list:
      disp_c +=1

      # Extract raw feature from cnn
      feat = self.extract_feat_image(frame).squeeze()

      # Compose single feature vector
      feat = self.pool_feats(feat)

      # Add to the array of features
      xfeats[disp_c - 1, :] = feat

      # Display every now and then
      if disp_c % disp_n == 0:
        print disp_c, '/', n_ims, "|", time.time() - t0

    # Dump to disk
    feats_path = self.save_db_feats
    pickle.dump(xfeats,open(feats_path, 'wb'))

    print " ============================ "


if __name__ == "__main__":

  params = get_params()

  E      = Extractor(params)

  E.save_feats_to_disk()

  # norm, pca, whitten -> save pca model
  feats_path = params['database_feats']
  feats      = pickle.load(open(feats_path,'rb'))
  learn_transform(params, feats)

import os
import sys
import time
import numpy as np
import argparse
import importlib

import tensorflow as tf

import provider

SEED =0 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/Pointconv'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointconv_noweight_cls', help='Model name')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/FixAxis/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 250]')
parser.add_argument('--layer', default='fc512', help='')
parser.add_argument('--rotate', type=bool, default=False, help='whether to random rotate input')
parser.add_argument('--dataset', default='modelnet', help='Which dataset to use [modelnet, mnist, shapenet]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MAX_EPOCH = FLAGS.max_epoch
LAYER = FLAGS.layer
MODEL = importlib.import_module(FLAGS.model) # import network module

os.environ['CUDA_VISIBLE_DEVICES']= str(GPU_INDEX)

# Dataset Path
ModelNet_File = '/nfs-data/user4/dataset/modelnet400/test_files.txt'
ShapeNet_File = '/nfs-data/user4/dataset/shapenet_sid0/test_files.txt'
Mnist_File = '/nfs-data/user4/dataset/Minist_Entropy0/test_files.txt'
Fivesmall_File = '/nfs-data/user4/dataset/five_background_small/test_files.txt'

def set_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def merge(main_data, attach_data):
    if main_data is None:
        main_data = attach_data
    else:
        main_data = np.concatenate((main_data,attach_data), axis = 0)
    return main_data

def get_dataset(dataset_name):
    data = None
    label = None
    num_class = None
    if dataset_name == 'modelnet':
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, ModelNet_File))
        num_class = 40
    elif dataset_name == 'mnist':
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, Mnist_File))
        num_class = 10
    elif dataset_name == 'shapenet':
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, ShapeNet_File))
        num_class = 16
    elif dataset_name == 'five_small':
        TEST_FILES = provider.getDataFiles(os.path.join(Fivesmall_File))
        num_class = 40
    
    #Merge data from multiple files  
    for fn in range(len(TEST_FILES)):
        tmp_data, tmp_label = provider.loadDataFile(TEST_FILES[fn])
        data = merge(data,tmp_data)
        label = merge(label,tmp_label)
    return data, label, num_class

class ComputeSigma():
    def __init__(self, layer, sigma_init, NUM_CLASSES):
        self.layer = layer
        self.sigma_init = sigma_init
        self.is_training = False
        self.pointclouds_pl, self.labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        self.is_training_pl = tf.placeholder(tf.bool, shape=())
        self.net, self.sigma, self.end_points = MODEL.get_model(self.pointclouds_pl, self.is_training_pl, NUM_CLASSES, self.sigma_init)
        self.loss, self.feature_loss, self.sigma_loss = MODEL.get_similarity_loss(self.end_points[self.layer], self.sigma, None, batch_size=BATCH_SIZE)

def evaluate(layer, sigma_init):
    is_training = False
    data, label, num_class = get_dataset(FLAGS.dataset)
    compute = ComputeSigma(layer, sigma_init, num_class)

     # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['sigma','batch'])
    saver_old = tf.train.Saver(variables_to_restore)

    #Init variables
    init = tf.global_variables_initializer()
    sess.run(init, {compute.is_training_pl:False})
    saver_old.restore(sess, MODEL_PATH)
    ops = {'pointclouds_pl': compute.pointclouds_pl,
           'labels_pl': compute.labels_pl,
           'is_training_pl': compute.is_training_pl,
           'loss': compute.loss,
           'feature_loss': compute.feature_loss,
           'sigma_loss': compute.sigma_loss,
           'sigma': compute.sigma}

    feature_loss = 0
    for epoch in range(MAX_EPOCH):
        feature_loss_tp = evaluate_jitter_one_epoch(sess, ops, data, label)
        feature_loss += feature_loss_tp
    print(layer+' sigma loss:',feature_loss/float(MAX_EPOCH))


def evaluate_jitter_one_epoch(sess, ops, data, label):
    is_training = False
    feature_loss_sum = 0
    batch_sum = 0
    for idx in range(data.shape[0]//BATCH_SIZE):
        #print('----' + str(idx) + '-----')
        current_data = data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,0:NUM_POINT,:]
        current_label = label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        current_label = np.squeeze(current_label)

        batch_sum += BATCH_SIZE
        if FLAGS.rotate:
            current_data = provider.rotate_point_cloud_randomaxis(current_data)
        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['is_training_pl']: is_training,}
        loss_val, feature_loss, sigma_loss, sigma = sess.run([ops['loss'], ops['feature_loss'], ops['sigma_loss'], ops['sigma']], feed_dict=feed_dict)
        feature_loss_sum += feature_loss

    print('feature loss: %f' % (feature_loss_sum / float(batch_sum)))

    return feature_loss_sum / batch_sum


if __name__ == "__main__":
    set_seed(SEED)
    with tf.Graph().as_default():
        evaluate(LAYER, 0.007)

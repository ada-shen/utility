import os
import sys
import time
import argparse
import numpy as np
import importlib

import tensorflow as tf

import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/Pointconv'))
# sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointconv_noweight_cls', help='Model name')
parser.add_argument('--layer', default='fc512', help='Layer ')
parser.add_argument('--sigma_init', type=float, default=0.03, help='Initial learning rate [default: 0.001]')
parser.add_argument('--pretrain_model', default='log/five_small/model.ckpt', help='')
parser.add_argument('--log_dir', default='log/jitter_bypoint', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=2001, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=40000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--sigma_weight', type=float, default=0.004, help='')
parser.add_argument('--sigmaf', type=float, default=0.004, help='')
parser.add_argument('--seed', type=int, default=0, help='Random Seed')
parser.add_argument('--dataset', type=str, default='modelnet', help='Dataset to use [modelnet mnist shapenet five_small]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LAYER = FLAGS.layer
PRETRAIN_MODEL = FLAGS.pretrain_model
SIGMA_INIT = FLAGS.sigma_init
SIGMA_WEIGHT = FLAGS.sigma_weight
SIGMAF = FLAGS.sigmaf
DATASET = FLAGS.dataset
SEED = FLAGS.seed

os.environ['CUDA_VISIBLE_DEVICES']= str(GPU_INDEX)

MODEL = importlib.import_module(FLAGS.model) # import network module

BOUNDARY = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Dataset Path
ModelNet_File = '/nfs-data/user4/dataset/Modelnet_Entropy0/test_files.txt'
ShapeNet_File = '/nfs-data/user4/dataset/shapenet_sid0/test_files.txt'
Mnist_File = '/nfs-data/user4/dataset/Minist_Entropy0/test_files.txt'
Fivesmall_File = '/nfs-data/user4/dataset/ModelnetFivesmall_Entropy0/test_files.txt'
save_path = '/nfs-data/user4/new_results/'

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

def log_string(out_str):
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.parser.add_argument('--decay_step', type=int, default=10000, help='Decay step for lr decay [default: 200000]')
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.0000001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train_jitter_one_epoch(sess, ops, data, label, idx):
    is_training = False
    current_data = data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,0:NUM_POINT,:]
    current_label = label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
    current_label = np.squeeze(current_label)
    loss_sum = 0

    rotated_data = current_data
    feed_dict = {ops['pointclouds_pl']: rotated_data,
                 ops['labels_pl']: current_label,
                 ops['is_training_pl']: is_training,}
    step, _, loss_val, feature_loss, sigma_loss, sigma = sess.run([ops['step'], ops['train_op'], ops['loss'], 
        ops['feature_loss'], ops['sigma_loss'], ops['sigma']], feed_dict=feed_dict)
    loss_sum += loss_val
    sigma = np.array(sigma)
    log_string('mean loss: %f' % (loss_sum / 1.0))
    log_string('feature loss: %f' % (feature_loss / 1.0))
    log_string('sigma loss: %f' % (sigma_loss / 1.0))
    print('mean of sigma:',np.mean(np.abs(sigma)))

    return loss_sum / 1.0, feature_loss / 1.0, sigma_loss / 1.0, sigma

def train_batch(NUM_CLASSES, sigma_init, idx, repeat_num, sigmaf, max_epoch, data, label):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.get_variable('batch',[],initializer=tf.constant_initializer(0),trainable=False)
            bn_decay = get_bn_decay(batch)

            # Get model and loss
            point_feature, sigma, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, sigma_init, bn_decay=bn_decay)
            loss, feature_loss, sigma_loss = MODEL.get_similarity_loss(end_points[LAYER], sigma, sigmaf,sigma_weight=SIGMA_WEIGHT, batch_size=BATCH_SIZE)

            # Get training operator
            var_list = tf.contrib.framework.get_variables('sigma')
            learning_rate = get_learning_rate(batch)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch,var_list=var_list)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['sigma','batch'])
        saver_old = tf.train.Saver(variables_to_restore)
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:False})
        saver_old.restore(sess, PRETRAIN_MODEL)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'lr' : learning_rate,
               'batch' : batch,
               'loss': loss,
               'feature_loss': feature_loss,
               'sigma_loss': sigma_loss,
               'sigma': sigma,
               'train_op': train_op,
               'step': batch}
        #define the save parament
        loss_val = []
        feature_loss_val = []
        sigma_val = []
        best_sigma = None

        diff = 100

        for epoch in range(max_epoch):
            log_string('**** EPOCH %03d ****' % (epoch))
            loss_tp, feature_loss_tp, _, sigma_tp = train_jitter_one_epoch(sess, ops, data, label, idx)
            loss_val.append(loss_tp)
            feature_loss_val.append(feature_loss_tp)
            if epoch % 50 == 0:
                sigma_val.append(sigma_tp)
            if sigmaf is not None:
                if feature_loss_tp >= 0.5:
                    if np.abs(feature_loss_tp-BOUNDARY) < diff:
                        diff = np.abs(feature_loss_tp-BOUNDARY)
                        best_sigma = sigma_tp

        if sigmaf is not None:
            sigma_val.append(best_sigma)
            return loss_val, feature_loss_val, sigma_val

if __name__ == "__main__":
    set_seed(SEED)
    data, label, num_class = get_dataset(DATASET)
    sigmaf = SIGMAF

    sigma_list = []
    loss_list = []
    feature_loss_list = []
    for idx in range(data.shape[0]//BATCH_SIZE):
        print ('The initialized sigma f is :', sigmaf)
        start_time = time.time()
        loss_val, featureloss_val, sigma_val = train_batch(num_class, sigma_init=0.03, idx=idx, repeat_num=None, sigmaf=sigmaf, max_epoch=MAX_EPOCH,data=data,label=label)
        tf.reset_default_graph()
        sigma_list.append(np.array(sigma_val))
        loss_list.append(np.array(loss_val))
        feature_loss_list.append(np.array(featureloss_val))
        end_time = time.time()
        print('time: ', end_time-start_time)
    
        np.savez(save_path+str(FLAGS.model)+'_'+FLAGS.dataset+'_sigma.npz', sigma=sigma_list, loss = loss_list, feature_loss = feature_loss_list)


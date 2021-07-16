from __future__ import division
from __future__ import print_function

import os
import sys
import json
import argparse
import numpy as np
import importlib
import provider

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/Dgcnn'))
sys.path.append(os.path.join(BASE_DIR, 'tf_utils'))


save_dir = '/nfs-data/user4/rotate_file_1'
dataset_dir = "/nfs-data/user4/"
pretrain_dir = "/nfs-data/user3/utility/pretrain_model"

def load_parament(json_path, dataset_name, model_name):
    with open(json_path,'r') as load_f:
        load_dict = json.load(load_f)
    data_dict = load_dict["dataset"][dataset_name]
    model_dict = load_dict["model"][model_name][dataset_name]

    num_classes = data_dict["num_classes"]
    data_root = dataset_dir + data_dict["path"]
    checkpoint = pretrain_dir + model_dict["checkpoint"]
    sigmaf = model_dict["sigmaf"]

    if sigmaf == -1:
        raise Exception("The sigmaf value is not valid! Please use compute_sigmaf.py \
                        to calculate sigmaf before running!")
    return num_classes, data_root, checkpoint, sigmaf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn', help='Model name')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=801, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=6000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--layer', default='fc512', help='Layer name [default: fc1024]')
parser.add_argument('--sigma_init', type=float, default=0.03, help='Initial learning rate [default: 0.001]')
parser.add_argument('--sigma_weight', type=float, default=0.0035, help='the very layer have calculate or not')
parser.add_argument('--rotate_num', type=int, default=40, help='the number of rotate number')
parser.add_argument('--sample_num', type=int, default=50, help='the number of sample')
parser.add_argument('--model_name', type=str, help='available ori/weight/density/sift')
parser.add_argument('--dataset_name', type=str, help='available modelnet/shapenet/mnist')
parser.add_argument('--json', default="p2s_config.json", type=str)
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
MAX_EPOCH = args.max_epoch
BASE_LEARNING_RATE = args.learning_rate
GPU_INDEX = args.gpu
MOMENTUM = args.momentum
OPTIMIZER = args.optimizer
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate
LAYER = args.layer
SIGMA_INIT = args.sigma_init
SIGMA_WEIGHT = args.sigma_weight
NUM_CLASS, DATA_PATH, PRETRAIN_PATH, SIGMAF = \
      load_parament(args.json, args.dataset_name, args.model_name)

MODEL = importlib.import_module(args.model) # import network module
LOG_DIR = os.path.join(args.log_dir, args.model.split('_')[0]+'_'+args.model_name+'_'+args.dataset_name)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

os.system('cp train_rotate.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args)+'\n')
os.environ['CUDA_VISIBLE_DEVICES']= str(GPU_INDEX)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAX_NUM_POINT = 2048
BOUNDARY = 2.0

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def set_seed(seed=0):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def mkdir_process(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
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

def train_jitter_one_epoch(sess, ops, train_writer, dataset, idx, rotate_pram, sigmaf):
    is_training = False
    current_data, current_label = dataset

    # use batch size point cloud as input point cloud
    current_data = current_data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,0:NUM_POINT,:]
    current_data = np.reshape(current_data,[BATCH_SIZE,NUM_POINT, -1])
    current_label = current_label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]

    current_label = np.squeeze(current_label).reshape((1))
    rotated_data = provider.rotate_point_cloud_randomangle(current_data,rotate_pram)
    loss_sum = 0
    feed_dict = {ops['pointclouds_pl']: rotated_data,
                 ops['labels_pl']: current_label,
                 ops['is_training_pl']: is_training,}
    pred_val, step, _, loss_val, feature_loss, sigma_loss, sigma, gf_index = sess.run([ops['pred'], ops['step'],
        ops['train_op'], ops['loss'], ops['feature_loss'], ops['sigma_loss'], ops['sigma'], ops['gf_index']], feed_dict=feed_dict)

    pred_val = np.argmax(pred_val[0:BATCH_SIZE], 1)
    correct = np.sum(pred_val == current_label)
    loss_sum += loss_val
    sigma = np.array(sigma)
    log_string('batch: %d' % step)
    log_string('mean loss: %f' % (loss_sum / 1.0))
    log_string('feature loss: %f' % (feature_loss / 1.0))
    log_string('sigma loss: %f' % (sigma_loss / 1.0))
    log_string('accuracy: %f' % (correct / BATCH_SIZE))
    return loss_sum / 1.0, feature_loss / 1.0, sigma_loss / 1.0, sigma, gf_index

def train_batch(sigma_init, dataset, idx, repeat_num, rotate_pram, sigmaf, max_epoch, noise_num=None):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.get_variable('batch',[],initializer=tf.constant_initializer(0),trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points, sigma, point_merge = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASS, sigma_init=sigma_init, bn_decay=bn_decay)
            loss, feature_loss, sigma_loss = MODEL.get_similarity_loss(end_points[LAYER], sigma, sigmaf, SIGMA_WEIGHT)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            tvars = [var for var in tf.trainable_variables() if var.name.startswith('sigma_net')]
            train_op = optimizer.minimize(loss, global_step=batch,var_list=tvars)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['sigma','batch'])
        saver_old = tf.train.Saver(variables_to_restore)
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:False})
        saver_old.restore(sess, PRETRAIN_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'gf_index' : pred,
               'lr' : learning_rate,
               'pf': point_merge,
               'loss': loss,
               'feature_loss': feature_loss,
               'sigma_loss': sigma_loss,
               'sigma': sigma,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        #define the save parament
        loss_val = []
        feature_loss_val = []
        sigma_val = []
        best_sigma = None
        diff = 100

        for epoch in range(max_epoch):
            if noise_num is not None:
                log_string('**** EPOCH %03d ****' % (noise_num))
            else:
                log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            loss_tp, feature_loss_tp, _, sigma_tp, _ = train_jitter_one_epoch(sess, ops, train_writer, dataset, idx, rotate_pram, sigmaf)
            loss_val.append(loss_tp)
            feature_loss_val.append(feature_loss_tp)
            if epoch % 40 == 0:
                sigma_val.append(sigma_tp)
            if sigmaf is not None:
                if feature_loss_tp >= BOUNDARY-1.2:
                    if np.abs(feature_loss_tp-BOUNDARY) < diff:
                        diff = np.abs(feature_loss_tp-BOUNDARY)
                        best_sigma = sigma_tp

        if sigmaf is not None:
            data_path = os.path.join(npz_path, str(idx)+'_'+str(repeat_num)+'.npz')
            sigma_val.append(best_sigma)
            np.savez(data_path, loss = loss_val ,f_loss = feature_loss_val, sigma = sigma_val)
        return feature_loss_val[0]

if __name__ == "__main__":
    set_seed(args.seed)
    choose_dataset = 0
    global npz_path
    model_prefix = args.model.split('_')[0]
    npz_path = os.path.join(save_dir, model_prefix+'_'+args.model_name+'_'+args.dataset_name)
    mkdir_process(npz_path)
    TEST_FILES = provider.getDataFiles(os.path.join(DATA_PATH, 'test_files.txt'))
    dataset = provider.loadDataFile(TEST_FILES[choose_dataset])
    for idx in range(0, args.sample_num):
        for i in range(0, args.rotate_num):
            rotate_pram = provider.generate_random_axis(BATCH_SIZE)
            _ = train_batch(sigma_init=args.sigma_init, dataset=dataset, idx=idx, repeat_num=i, rotate_pram=rotate_pram, sigmaf=SIGMAF, max_epoch=MAX_EPOCH)
    LOG_FOUT.close()

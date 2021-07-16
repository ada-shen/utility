import os
import sys
import time
import argparse
import importlib
import numpy as np

import tensorflow as tf

import provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models/Pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=3, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ad', help='Model name')
parser.add_argument('--log_dir', default='log/jitter', help='Log dir [default: log]')
parser.add_argument('--pretrain_model', default='log/FixAxis/model.ckpt', help='')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=5000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--lam', type=float, default=1, help='Balance Parameter')
parser.add_argument('--seed', type=int, default=0, help='Random Seed')
parser.add_argument('--dataset', type=str, default='modelnet', help='Random Seed')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LAM = FLAGS.lam
PRETRAIN_MODEL = FLAGS.pretrain_model
SEED = FLAGS.seed
DATASET = FLAGS.dataset

MODEL = importlib.import_module(FLAGS.model) # import network module

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Dataset Path
ModelNet_File = '/nfs-data/user4/dataset/modelnet400/test_files.txt'
ShapeNet_File = '/nfs-data/user4/dataset/shapenet_0/test_files.txt'
Mnist_File = '/nfs-data/user4/dataset/3d-minist0/test_files.txt'
save_path = '/nfs-data/user4/new_results/'

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
    
    #Merge data from multiple files  
    for fn in range(len(TEST_FILES)):
        tmp_data, tmp_label = provider.loadDataFile(TEST_FILES[fn])
        data = merge(data,tmp_data)
        label = merge(label,tmp_label)
    return data, label, num_class

def log_string(out_str):
    print(out_str)

def set_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
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

def train(NUM_CLASSES, data, target_label):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.get_variable('batch', [],
                         initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)

            # Get model and loss
            pred, pert_vec, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss, clr_loss, pert_loss = MODEL.get_adversarial_loss(pred, labels_pl, pert_vec, lam =LAM)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)

            var_list = tf.contrib.framework.get_variables('epsilon')

            # Get training operator
            learning_rate = get_learning_rate(batch)
            #tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch,var_list = var_list)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['epsilon','batch'])
        saver_old = tf.train.Saver(variables_to_restore)
        # Init variables
        init = tf.global_variables_initializer()

        sess.run(init, {is_training_pl: True})
        saver_old.restore(sess, PRETRAIN_MODEL)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pert_vec': pert_vec,
               'clr_loss':clr_loss,
               'pert_loss': pert_loss,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            correct, l2n = train_adversarial_one_epoch(sess, ops, data, target_label)
            if correct == 1:
                break
        return l2n


def train_adversarial_one_epoch(sess, ops, data, target_label):
    is_training = False
    loss_sum = 0
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['labels_pl']: target_label,
                 ops['is_training_pl']: is_training,}

    step, _, loss_val, clr_loss, pert_loss_val, pred_val, pert_vec = sess.run([ops['step'],ops['train_op'], ops['loss'], ops['clr_loss'], ops['pert_loss'], ops['pred'], ops['pert_vec']], feed_dict=feed_dict)
    pred_val = np.argmax(pred_val, 1)
    correct = np.sum(pred_val == target_label)
    loss_sum += loss_val
    if correct == 1:
        log_string('mean loss: %f' % (loss_sum / 1.0))
        log_string('clr_loss: %f' % (clr_loss / 1.0))
        log_string('pert loss: %f' % (pert_loss_val / 1.0))
        log_string(' correct: %f' % (correct / 1.0))
    return correct,pert_loss_val

def merge(all,tmp):
    if all is None:
        all = tmp
    else:
        all = np.concatenate((all,tmp),axis=0)
    return all

if __name__ == "__main__":
    set_seed(SEED)
    log_string('pid: %s'%(str(os.getpid())))

    data, label, num_class = get_dataset(DATASET)
    delta = []
    sample_sum = 0
    if DATASET == 'modelnet':
        total_num = 2400
        step = 24
    elif DATASET == 'mnist':
        total_num = 1000
        step = 10
    elif DATASET == 'shapenet':
        total_num = 2800
        step = 28
    for i in range(0,total_num,step):
        start_time = time.time()
        use_data = data[i:i+1,:NUM_POINT,:]
        use_label = label[i]
        target_labels = np.arange(num_class)
        target_labels = target_labels[target_labels!=use_label]
        sample_delta_sum = 0
        for target in target_labels:
            print('-------------sample---'+str(i)+'--------------')
            sample_delta_sum += (train(num_class, use_data,np.reshape(target,(-1,))))
        sample_delta = sample_delta_sum / target_labels.shape[0]
        delta.append(sample_delta)
        print('sample '+ str(i)+': '+ str(sample_delta))
        end_time = time.time()
        print('time:' , end_time-start_time)
        sample_sum += sample_delta
        tf.reset_default_graph()
    print('Adversarial:', FLAGS.model, sample_sum/100)
    np.savez(save_path+FLAGS.model+'_'+DATASET+'.npz', ad=np.array(delta))

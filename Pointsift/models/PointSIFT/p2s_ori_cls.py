'''
Basic Point2Sequence classification model
'''
import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'tf_utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointSIFT_util import pointnet_sa_module, point2sequence_module
from noisy import add_noisy_by_point

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Classification Point2Sequence, input is BxNx3, output BxCLASSES """
    batch_size = point_cloud.get_shape()[0].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # p2s add oe module
    l1_xyz, l1_points = point2sequence_module(l0_xyz, l0_points, 384, [16,32,64,128], [[32,64,128], [64,64,128], [64,64,128], [128,128,128]], 128, 128, is_training, bn_decay, scope='layer1', add_oe=False)
    l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    # Fully connected layers
    net = tf.reshape(l2_points, [l2_points.get_shape()[0].value, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, num_class, activation_fn=None, scope='fc3')

    return net, end_points

def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)

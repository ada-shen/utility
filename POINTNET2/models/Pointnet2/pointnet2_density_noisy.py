"""
    PointNet++ Model for point clouds classification
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tensorflow as tf
import numpy as np
import models.Pointnet2.utils.tf_util as tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_density
from models.noisy import add_noisy_by_point

BANDW = 0.05
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, NUM_CLASSES, sigma_init=None, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    ##add noise area##
    point_merge, sigma = add_noisy_by_point(point_cloud, sigma_init)
    l0_xyz = point_merge
    l0_points = None
    end_points['l0_xyz'] = l0_xyz
    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices = pointnet_sa_module_density(l0_xyz, l0_points, npoint=512, radius=0.2, sigma=2*BANDW, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', sigma_init=sigma_init, use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module_density(l1_xyz, l1_points, npoint=128, radius=0.4, sigma=4*BANDW, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2', sigma_init=sigma_init)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module_density(l2_xyz, l2_points, npoint=None, radius=1.6, sigma=16*BANDW ,nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3', sigma_init=sigma_init)

    # Fully connected layers
    net = tf.reshape(l3_points, [l3_points.get_shape()[0].value, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['fc512'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, NUM_CLASSES, activation_fn=None, scope='fc3')
    return net, sigma, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # transform = end_points['transform'] # BxKxK
    # K = transform.get_shape()[1].value
    # mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    # mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    # mat_diff_loss = tf.nn.l2_loss(mat_diff)
    # tf.summary.scalar('mat loss', mat_diff_loss)

    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def get_similarity_loss(end_points , sigma, sigmaf,  sigma_weight = 0.003, batch_size=None):
    batch_size = end_points.get_shape()[0].value//2
    real_feature = tf.gather(end_points, [i  for i in range(batch_size)])
    jitter_feature = tf.gather(end_points, [i  for i in range(batch_size,2*batch_size)])
    feature_loss = tf.reduce_sum(tf.square(jitter_feature-real_feature))/batch_size
    print('feature shape:', real_feature.shape, jitter_feature.shape)
    if sigmaf is None:
        point_weight = 1.0
    else:
        point_weight = 1.0 / sigmaf
    #sigma_loss = tf.reduce_sum(tf.log(tf.square(sigma)))/batch_size
    sigma_loss = tf.reduce_sum(tf.log(tf.abs(sigma)))/batch_size
    return point_weight * feature_loss - sigma_weight * sigma_loss, point_weight * feature_loss, sigma_weight * sigma_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)

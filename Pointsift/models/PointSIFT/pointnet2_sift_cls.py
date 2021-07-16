import os
import sys

import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'tf_utils'))
import tf_util
from pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_sa_module, pointnet_sa_module_noise, pointSIFT_res_module_noise, pointSIFT_nores_module
from noisy import add_noisy_by_point

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, num_class, bn_decay=None, feature=None):
    """ Semantic segmentation PointNet, input is B x N x 3, output B x num_class """
    batch_size = point_cloud.get_shape()[0].value
    end_points = {}

    l0_xyz = point_cloud
    l0_points = feature
    # end_points['l0_xyz'] = l0_xyz

    # pointnet++ add sift module only with OE module
    # c0_l0_xyz, c0_l0_points, c0_l0_indices = pointSIFT_nores_module_noise(l0_xyz, l0_points, radius=0.25, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer0_c0', merge='concat')
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

    # c0_l1_xyz, c0_l1_points, c0_l1_indices = pointSIFT_nores_module_noise(l1_xyz, l1_points, radius=0.5, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='layer1_c0')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    c0_l2_xyz, c0_l2_points, c0_l2_indices = pointSIFT_nores_module(l2_xyz, l2_points, radius=0.85, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='layer2_c0')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(c0_l2_xyz, c0_l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    
    l4_points = l3_points

    # FC layers
    # net = tf.reshape(l4_points, [batch_size, -1])
    net = tf_util.conv1d(l4_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['fc512'] = tf.squeeze(net)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc3')
    net = tf.squeeze(net)
    return net, end_points

def get_loss(pred, label):
    """
    :param pred: BxNxC
    :param label: BxN
    :param smpw: BxN
    :return:
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

def get_similarity_loss(end_points , sigma, sigmaf, sigma_weight = 0.001):
    batch_size = end_points.get_shape()[0].value // 2
    real_feature = tf.gather(end_points, [i  for i in range(batch_size)])
    jitter_feature = tf.gather(end_points, [i  for i in range(batch_size, 2*batch_size)])
    print('feature shape:',end_points.shape, real_feature.shape,jitter_feature.shape)
    feature_loss = tf.reduce_sum(tf.square(jitter_feature-real_feature))/batch_size
    if sigmaf is None:
        point_weight = 1.0
    else:
        point_weight = 1.0 / sigmaf
    #sigma_loss = tf.reduce_sum(tf.log(tf.square(sigma)))/batch_size
    sigma_loss = tf.reduce_sum((tf.log(tf.abs(sigma))))/batch_size
    loss = point_weight * feature_loss - sigma_weight * sigma_loss
    tf.add_to_collection('losses', loss)
    return loss, point_weight * feature_loss, sigma_weight * sigma_loss


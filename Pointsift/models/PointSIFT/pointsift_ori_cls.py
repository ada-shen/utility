import os
import sys

import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'tf_utils'))
import tf_util
from pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_fp_module, pointnet_sa_module


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    # smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None, feature=None):
    """ Semantic segmentation PointNet, input is B x N x 3, output B x num_class """
    batch_size = point_cloud.get_shape()[0].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = feature

    # c0
    c0_l0_xyz, c0_l0_points, c0_l0_indices = pointSIFT_res_module(l0_xyz, l0_points, radius=0.1, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer0_c0', merge='concat')
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(c0_l0_xyz, c0_l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')

    # c1
    c0_l1_xyz, c0_l1_points, c0_l1_indices = pointSIFT_res_module(l1_xyz, l1_points, radius=0.25, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='layer1_c0')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(c0_l1_xyz, c0_l1_points, npoint=256, radius=0.2, nsample=32, mlp=[128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')

    # c2
    c0_l2_xyz, c0_l2_points, c0_l2_indices = pointSIFT_res_module(l2_xyz, l2_points, radius=0.5, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='layer2_c0')
    c1_l2_xyz, c1_l2_points, c1_l2_indices = pointSIFT_res_module(c0_l2_xyz, c0_l2_points, radius=0.5, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='layer2_c1', same_dim=True)
    l2_cat_points = tf.concat([c0_l2_points, c1_l2_points], axis=-1)
    fc_l2_points = tf_util.conv1d(l2_cat_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='conv_2_fc', bn_decay=bn_decay)

    # c3
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(c1_l2_xyz, fc_l2_points, npoint=64, radius=0.4, nsample=32, mlp=[512,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # FC layers
    net = tf_util.conv1d(l4_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
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

import os
import sys

import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'tf_utils'))
import tf_util
from pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_sa_module, pointnet_sa_module_noise, pointSIFT_res_module_noise, pointSIFT_nores_module_noise
from noisy import add_noisy_by_point

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    # smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, num_class, sigma_init, bn_decay=None, feature=None):
    """ Semantic segmentation PointNet, input is B x N x 3, output B x num_class """
    batch_size = point_cloud.get_shape()[0].value
    end_points = {}

    point_merge, sigma = add_noisy_by_point(point_cloud, sigma_init)

    l0_xyz = point_merge
    l0_points = feature
    # end_points['l0_xyz'] = l0_xyz

    # without sift module###
    l1_xyz, l1_points, l1_indices = pointnet_sa_module_noise(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module_noise(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module_noise(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module_noise(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # FC layers
    net = tf_util.conv1d(l4_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['fc512'] = tf.squeeze(net)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc3')
    net = tf.squeeze(net)
    return net, end_points, sigma, point_merge

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

def get_similarity_loss_delta(end_points , sigma, sigmaf,  sigma_weight = 0.005):
    batch_sz = end_points.get_shape()[0].value
    real_feature = tf.gather(end_points, [0])
    real_feature = tf.tile(real_feature,[batch_sz-1,1])
    jitter_feature = tf.gather(end_points, [i  for i in range(1,batch_sz)])
    feature_loss = tf.reduce_sum(tf.square(jitter_feature-real_feature))/batch_sz
    if sigmaf is None:
        point_weight = 1.0
    else:
        point_weight = 1.0 / sigmaf
    C = np.log(np.e*np.pi*2)*0.5
    sigma_loss = tf.reduce_sum(tf.log(tf.abs(sigma)))
    return point_weight * feature_loss - sigma_weight * sigma_loss, point_weight * feature_loss, sigma_weight * sigma_loss


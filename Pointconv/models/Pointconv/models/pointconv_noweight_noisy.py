import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from PointConv_fixsample import  feature_encoding_noweight
from noisy import add_noisy_by_point

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    # smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, num_class, sigma_init, sigma=0.05, bn_decay=None, weight_decay = None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    #######noisy generate################################
    point_merge, sigma_val = add_noisy_by_point(point_cloud, sigma_init)
    #####################################################
    l0_xyz = point_merge
    l0_points = point_merge

    # Feature no density layers
    l1_xyz, l1_points = feature_encoding_noweight(l0_xyz, l0_points, npoint=1024, radius = 0.1, sigma = sigma, K=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer1')
    # end_points['sample1'] = l1_points
    l2_xyz, l2_points = feature_encoding_noweight(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2')
    # end_points['sample2'] = l2_points
    l3_xyz, l3_points = feature_encoding_noweight(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3')
    # end_points['sample3'] = l3_points
    l4_xyz, l4_points = feature_encoding_noweight(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8 * sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')
    # end_points['sample4'] = l4_points
    l5_xyz, l5_points = feature_encoding_noweight(l4_xyz, l4_points, npoint=1, radius = 1.6, sigma = 16 * sigma, K=36, mlp=[512,1024], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer5')

    # FC layers
    # end_points['fc1024'] = l5_points
    net = tf_util.conv1d(l5_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['fc512'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay, weight_decay=weight_decay)
    # end_points['fc128'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc3')
    net = tf.squeeze(net)
    # end_points['fc40'] = net
    return net, end_points, sigma_val, point_merge

# sigma_weight fc1024  0.00007
# sigma_weight fc512  0.00005
# sigma_weight fc40  0.0001
# random rotate axis:0.001; stablity2:
def get_similarity_loss(end_points , sigma, sigmaf,  sigma_weight = 0.0003):
    batch_sz = end_points.get_shape()[0].value // 2
    real_feature = tf.gather(end_points, [i  for i in range(batch_sz)])
    jitter_feature = tf.gather(end_points, [i  for i in range(batch_sz,2*batch_sz)])
    print('feature shape:', real_feature.shape, jitter_feature.shape)
    feature_loss = tf.reduce_sum(tf.square(jitter_feature-real_feature))/batch_sz
    if sigmaf is None:
        point_weight = 1.0
    else:
        point_weight = 1.0 / sigmaf
    sigma_loss = tf.reduce_sum(tf.log(tf.abs(sigma)))/batch_sz
    return point_weight * feature_loss - sigma_weight * sigma_loss, point_weight * feature_loss, sigma_weight * sigma_loss

def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """
    # classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    classify_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    weight_reg = tf.add_n(tf.get_collection('losses'))
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
    total_loss = classify_loss_mean + weight_reg
    tf.summary.scalar('classify loss', classify_loss_mean)
    tf.summary.scalar('total loss', total_loss)
    return total_loss, classify_loss_mean, weight_reg

if __name__=='__main__':
    import pdb
    pdb.set_trace()

    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10, 1.0)
        print(net)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../..', 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(BASE_DIR, '../..', 'tf_ops/grouping'))
from tf_interpolate import three_nn, three_interpolate
import tf_grouping
import pointconv_fixsample, pointconv_util
import tf_util

def unite_sample_group(xyz, npoint):
    ori_size = int(xyz.get_shape()[0].value // 2)
    sample_xyz = tf.gather(xyz,[i for i in range(ori_size)])
    center_xyz, new_xyz = pointconv_fixsample.sampling(npoint, xyz, sample_xyz)
    sample_xyz = tf.concat((sample_xyz, sample_xyz), axis=0)
    return new_xyz, sample_xyz, center_xyz

def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                padding = 'VALID', stride=[1, 1],
                                bn = True, is_training = is_training, activation_fn=activation_fn,
                                scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

    return net

def weight_net(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            if i != len(hidden_units) -1:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = True, is_training = is_training, activation_fn=activation_fn,
                                    scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
            else:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = False, is_training = is_training, activation_fn=None,
                                    scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp%d'%(i))
    return net

def nonlinear_transform(data_in, mlp, scope, is_training, bn_decay=None, weight_decay = None, activation_fn = tf.nn.relu):

    with tf.variable_scope(scope) as sc:

        net = data_in
        l = len(mlp)
        if l > 1:
            for i, out_ch in enumerate(mlp[0:(l-1)]):
                net = tf_util.conv2d(net, out_ch, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = True, is_training = is_training, activation_fn=tf.nn.relu,
                                    scope = 'nonlinear%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

                #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp_nonlinear%d'%(i))
        net = tf_util.conv2d(net, mlp[-1], [1, 1],
                            padding = 'VALID', stride=[1, 1],
                            bn = False, is_training = is_training,
                            scope = 'nonlinear%d'%(l-1), bn_decay=bn_decay,
                            activation_fn=tf.nn.sigmoid, weight_decay = weight_decay)

    return net

def feature_encoding_layer(xyz, feature, npoint, radius, sigma, K, mlp, is_training, bn_decay, weight_decay, scope, end_points, bn=True, use_xyz=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''

    with tf.variable_scope(scope) as sc:
        num_points = xyz.get_shape()[1]
        if num_points == npoint:
            _, sample_xyz, _ = unite_sample_group(xyz, npoint)
            new_xyz = sample_xyz
            center_xyz = xyz
        else:
            new_xyz, sample_xyz, center_xyz = unite_sample_group(xyz, npoint)

        grouped_xyz, grouped_feature, idx = pointconv_fixsample.grouping(feature, K, xyz, new_xyz, sample_xyz, center_xyz)

        density = pointconv_fixsample.kernel_density_estimation_ball(xyz, radius, sigma)
        inverse_density = tf.div(1.0, density)
        grouped_density = tf.gather_nd(inverse_density, idx) # (batch_size, npoint, nsample, 1)
        # grouped_density = tf_grouping.group_point(inverse_density, idx)
        inverse_max_density = tf.reduce_max(grouped_density, axis = 2, keepdims = True)
        density_scale = tf.div(grouped_density, inverse_max_density)

        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                grouped_feature = tf_util.conv2d(grouped_feature, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
                end_points[scope+'_'+str(i)] = grouped_feature
        weight = weight_net_hidden(grouped_xyz, [32], scope = 'weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        density_scale = nonlinear_transform(density_scale, [16, 1], scope = 'density_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        new_points = tf.multiply(grouped_feature, density_scale)
        new_points = tf.transpose(new_points, [0, 1, 3, 2])

        new_points = tf.matmul(new_points, weight)

        new_points = tf_util.conv2d(new_points, mlp[-1], [1,new_points.get_shape()[2].value],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        return center_xyz, new_points

def feature_encoding_noweight(xyz, feature, npoint, radius, sigma, K, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        num_points = xyz.get_shape()[1]
        if num_points == npoint:
            _, sample_xyz, _ = unite_sample_group(xyz, npoint)
            new_xyz = sample_xyz
            center_xyz = xyz
        else:
            new_xyz, sample_xyz, center_xyz = unite_sample_group(xyz, npoint)

        grouped_xyz, grouped_feature, idx = pointconv_fixsample.grouping(feature, K, xyz, new_xyz, sample_xyz, center_xyz)

        density = pointconv_fixsample.kernel_density_estimation_ball(xyz, radius, sigma)
        inverse_density = tf.div(1.0, density)
        grouped_density = tf.gather_nd(inverse_density, idx) # (batch_size, npoint, nsample, 1)

        inverse_max_density = tf.reduce_max(grouped_density, axis = 2, keepdims = True)
        density_scale = tf.div(grouped_density, inverse_max_density)

        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                grouped_feature = tf_util.conv2d(grouped_feature, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

        
        density_scale = nonlinear_transform(density_scale, [16, 1], scope = 'density_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        
        new_points = tf.multiply(grouped_feature, density_scale)
        
        new_points = tf.transpose(new_points, [0, 1, 3, 2])


        new_points = tf_util.conv2d(new_points, mlp[-1], [1,new_points.get_shape()[2].value],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        return center_xyz, new_points

def placeholder_inputs(batch_size, num_point, channel):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    feature_pts_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, channel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, feature_pts_pl, labels_pl

if __name__=='__main__':
    import numpy as np
    pts = np.random.random((32, 2048, 3)).astype('float32')
    fpts = pts
    sigma = 0.1
    N = 512
    K = 64
    D = 1
    C_list = [64, 128]
    mlp_w = [64]
    mlp_d = [64]
    is_training = tf.placeholder(tf.bool, shape=())

    import pdb
    pdb.set_trace()

    with tf.device('/gpu:1'):
        #points = tf.constant(pts)
        #features = tf.constant(fpts)
        points_pl, features_pl, labels_pl = placeholder_inputs(32, 2048, 3)
        sub_pts, features = feature_encoding_layer(points_pl, features_pl, N, sigma, K, [10, 20], is_training, bn_decay = 0.1, weight_decay = 0.1, scope = "FE")
        feature_decode = feature_decoding_layer(points_pl, sub_pts, features_pl, features, sigma, K, [10, 23], is_training, bn_decay=0.1, weight_decay = 0.1, scope= "FD")






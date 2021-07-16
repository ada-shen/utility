import tensorflow as tf
import numpy as np

def add_noisy_by_point(point_cloud, sigma_init):
    B, N, C = point_cloud.shape
    with tf.variable_scope("sigma_net", reuse=tf.AUTO_REUSE):
        if sigma_init is None:
            sigma = tf.get_variable(name = 'sigma', shape = [B, N], initializer=tf.constant_initializer(0.007),
                                    dtype=tf.float32 ,  trainable = False)
        else:
            sigma = tf.get_variable(name = 'sigma', shape = [B, N], initializer=tf.constant_initializer(sigma_init),
                                    dtype=tf.float32 ,  trainable = True)
    sigma = tf.clip_by_value(sigma,-1*0.08,0.08)
    epsional = tf.random_normal(shape = point_cloud.shape, mean = 0, stddev = 1, seed = 0)
    sigma_val = tf.expand_dims(sigma, -1)
    sigma_val = tf.tile(sigma_val, [1,1,C])
    noisy = tf.multiply(sigma_val, epsional)
    new_point = tf.add(point_cloud , noisy)

    point_merge = tf.concat([point_cloud, new_point], axis = 0)
    return point_merge, sigma

import tensorflow as tf
import numpy as np

def add_noisy_by_point(point_cloud, sigma_init):
    B, N, C = point_cloud.shape
    with tf.variable_scope("sigma_net"):
        if sigma_init is None:
            sigma = tf.get_variable(name = 'sigma', shape = [B, N], initializer=tf.constant_initializer(0.007),
                                    dtype=tf.float32 ,  trainable = False)
        else:
            #sigma = tf.get_variable(name = 'sigma', shape = [B, N], initializer=tf.truncated_normal_initializer(stddev=sigma_init, seed=1),
            sigma = tf.get_variable(name = 'sigma', shape = [B, N], initializer=tf.constant_initializer(sigma_init),
                                    dtype=tf.float32 ,  trainable = True)
    sigma = tf.clip_by_value(sigma, -0.08, 0.08)
    epsional = tf.random_normal(shape = point_cloud.shape, mean = 0, stddev = 1)
    sigma_val = tf.expand_dims(sigma, -1)
    sigma_val = tf.tile(sigma_val, [1,1,C])
    noisy = tf.multiply(sigma_val, epsional)
    new_point = tf.add(point_cloud , noisy)

    point_merge = tf.concat([point_cloud, new_point], axis = 0)
    return point_merge, sigma

def perturbation_point_uniform(point_cloud):
    epsilon = tf.get_variable(name='epsilon', shape = point_cloud.shape, initializer = tf.contrib.layers.xavier_initializer(),
                                    dtype= tf.float32 , trainable = True)
    noise_sum = tf.sqrt(tf.reduce_sum(tf.square(epsilon),axis=[1,2]))
    noise_sum = tf.expand_dims(tf.expand_dims(noise_sum, axis=-1), axis=-1)
    noise = epsilon / noise_sum
    pert_pc = tf.add(point_cloud, noise)
    return pert_pc, noise

def perturbation_point_xyz(point_cloud):
    B, N, C = point_cloud.shape
    epsilon = tf.get_variable(name='epsilon', shape = point_cloud.shape, initializer = tf.contrib.layers.xavier_initializer(),
                                    dtype= tf.float32 , trainable = True)
    pert_pc = tf.add(point_cloud, epsilon)
    return pert_pc, epsilon
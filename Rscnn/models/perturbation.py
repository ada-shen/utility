
import tensorflow as tf
import numpy as np

def perturbation_point_xyz(point_cloud):
    B, N, C = point_cloud.shape
    epsilon = tf.get_variable(name='epsilon', shape = [B,N,C], initializer = tf.contrib.layers.xavier_initializer(),dtype= tf.float32 , trainable = False)

    pert_pc = tf.add(point_cloud, epsilon)
    return pert_pc, epsilon

def perturbation_point_uniform(point_cloud):
    B, N, C = point_cloud.shape
    epsilon = tf.get_variable(name='epsilon', shape = point_cloud.shape, initializer = tf.contrib.layers.xavier_initializer(), dtype= tf.float32 , trainable = True)

    noise_sum = tf.sqrt(tf.reduce_sum(tf.square(epsilon),axis=[1,2]))
    noise_sum = tf.expand_dims(tf.expand_dims(noise_sum, axis=-1), axis=-1)
    noise = epsilon / noise_sum

    pert_pc = tf.add(point_cloud, noise)
    return pert_pc, noise
#!/usr/bin/env python
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
import tensorflow as tf
import numpy as np
import tf_util
import provider
SAVE_PATH1 = '/home/data2/sw/point2-master/data/consist_sid/knn_result'
SAVE_PATH2 = '/home/data2/sw/point2-master/data/consist_sid/new_knn'
SAVE_PATH3 = '/home/data2/sw/point2-master/data/consist_sid/rotate_result'
READ_PATH1 = '/home/ubuntu/project/skyler/results'
READ_PATH2 = '/home/data2/sw/point2-master/data/consist_sid/rotate_source'
ousider_list = ['netpp','pointsift','netppmsg','netppmsg1','plusdensity','plusweight']
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def local_consistency(radius,nsample,xyz,sigma,batch_size,knn=False):
    if knn:
        _,idx = knn_point(nsample, xyz, xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, xyz)

    sigma = tf.expand_dims(sigma, axis=-1)
    grouped_xyz = group_point(sigma, idx) # (batch_size, npoint, nsample,1)
    grouped_xyz = tf.squeeze(grouped_xyz)
    if grouped_xyz.get_shape()[0].value != batch_size:
        grouped_xyz = tf.expand_dims(grouped_xyz, axis=0)
#    print grouped_xyz.shape
    local_max = tf.reduce_max(grouped_xyz, axis = 2)
    local_min = tf.reduce_min(grouped_xyz, axis = 2)
#    print local_max.shape,local_min.shape
    local_distribution = local_max - local_min
    return local_distribution

def local_mean_sigma(radius,nsample,xyz,sigma,batch_size,knn=False):
    if knn:
        _,idx = knn_point(nsample, xyz, xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, xyz)

    sigma = tf.expand_dims(sigma, axis=-1)
    grouped_xyz = group_point(sigma, idx) # (batch_size, npoint, nsample,1)
    grouped_xyz = tf.squeeze(grouped_xyz)
    if grouped_xyz.get_shape()[0].value != batch_size:
        grouped_xyz = tf.expand_dims(grouped_xyz, axis=0)
#    print grouped_xyz.shape
    local_sigma = grouped_xyz - tf.tile(sigma,[1,1,nsample])
    local_distribution = tf.reduce_mean(local_sigma, axis=-1)
    return local_distribution

def drop_ousider(data):
    for i in range(data.shape[0]):
        data[i][np.abs(data[i])>0.079] = np.mean(np.abs(data[i][np.abs(data[i]<0.079)]))
        # data[i][np.abs(data[i])>0.2] = np.mean(np.abs(data[i][np.abs(data[i]<0.2)]))
    return data

def calculate(file_name):
    with tf.Graph().as_default():
        with tf.device('/gpu:1'):
            batch_size = 16
            num_point = 1024
            nsample = 16
            pointclouds = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
            sigma = tf.placeholder(tf.float32, shape=(batch_size, num_point))
            local_distribution = local_consistency(radius=1.0,nsample=nsample,xyz=pointclouds,sigma=sigma,batch_size=batch_size,knn=True)

        file_path  = os.path.join(READ_PATH1, file_name)
        dataset_name = file_name.split('_')[-1].split('.')[0]
        model_name = file_name.split('_')[0]
        if 'modelnet' in file_path :
            dataset_path = '/nfs-data/user4/dataset/Modelnet_Entropy0/test_files.txt'
        elif 'mnist' in file_path:
            dataset_path = '/nfs-data/user4/dataset/Minist_Entropy0/test_files.txt'
        elif 'shapenet' in file_path:
            dataset_path = '/nfs-data/user4/dataset/shapenet_sid0/test_files.txt'
        elif 'five_small' in file_path:
            dataset_path = '/nfs-data/user4/dataset/ModelnetFivesmall_Entropy0/test_files.txt'
        print(file_path)
        print(np.load(file_path)['sigma'].shape)
        sigma_val = np.load(file_path)['sigma']#[:,-1,:,:].reshape(128,1024)#
        # print(sigma_val.shape)
        # if model_name in ousider_list or model_name[0:5]=='netpp':
        sigma_val = drop_ousider(sigma_val)
        print('max sigma:',np.max(np.abs(sigma_val)))
        #pf = np.load(file_path)['pf'][0,...]
        pf, _ = provider.loadDataFile(provider.getDataFiles(dataset_path)[0])
        print('data shape:',sigma_val.shape, pf.shape)

        ###rotate operator###
        # sigma_val = sigma_val.reshape((-1,1024))
        # pf = pf[0:8,0:num_point,...]
        # pf = np.tile(np.expand_dims(pf,axis=1), [1,40,1,1]).reshape((-1,1024,3))

        mean_val = np.log(np.square(sigma_val))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        consistency_list = []
        for i in range(pf.shape[0]//batch_size):
            feed_dict={pointclouds:pf[batch_size*i:batch_size*(i+1),0:num_point,...],sigma:mean_val[batch_size*i:batch_size*(i+1),...]}
            # print(feed_dict)
            local_d = sess.run(local_distribution,feed_dict=feed_dict)
            consistency_list.append(np.abs(local_d))
        consistency_list = np.concatenate(consistency_list, axis=0)
        local_d = np.mean(np.abs(consistency_list))
        print('model:', model_name, '  dataset_name:', dataset_name, '  knn:', local_d)
        # result_path = os.path.join(SAVE_PATH1,'consist'+str(nsample)+'_'+model_name+'_'+dataset_name+'.npz')
        # np.savez(result_path, pf = pf[0:batch_size,...], sigma = mean_val[0:batch_size,...], local_d = local_d)
        return consistency_list

if __name__=='__main__':
    # FILENAME = os.listdir(READ_PATH1)
    # for file_name in FILENAME:
    #     if file_name == 'nettppdensity_modelnet.npz' or 'pointnet2_density_sigma.npz' or 'netppmsg_modelnet.npz':
    #         pass
    #     print(file_name)
        # calculate(file_name)
    data2 = np.array(calculate('/nfs-data/user3/utility/rotate_file/sid_pointsift_modelnet/sid_pointsift_modelnet.npz')).reshape(-1,1024)
    # data1 = np.array(calculate('/home/ubuntu/project/skyler/results/pointnet2_cls_msg_noisy_sigma_modelnet.npz')).reshape(-1,1024)
    # print(data2)
    # sigma_val1 = np.load('/nfs-data/user4/new_results/pointnet_cls_five_small_sigma.npz')['sigma'][:,-1,...].reshape(-1,1024)
    # sigma_val2 = np.load('/nfs-data/user4/new_results/pointnet_cls_five_small_sigma1.npz')['sigma'][:,-1,...].reshape(-1,1024)
    # print(np.mean(sigma_val1))
    # print(np.mean(sigma_val2))


    

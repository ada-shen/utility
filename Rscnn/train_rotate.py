
import os
import json
import yaml
import provider
import shutil
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

from models import RSCNN_SSN_Cls_Noisy
from models import RSCNN_MSN_Cls_Noisy
from models import RSCNN_DWS_Cls_Noisy
from models import get_similarity_loss
from data import DataSetCls
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#save_dir = '/nfs-data/user4/rotate_file'
#dataset_dir = "/nfs-data/user4/"
#pretrain_dir = "/nfs-data/user4/Relation-Shape-CNN-master"

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def argument():
    parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
    parser.add_argument('-config', default='cfgs/config_rotate.yaml', type=str)
    parser.add_argument('-json', default="./cfgs/rotate_config.json", type=str)
    parser.add_argument('-epochs', type=int, default=1000, help='the number of total training epoch')
    parser.add_argument('-gpu', type=str, help='available gpu 0/1/2/3')
    parser.add_argument('-dataset_name', type=str,default = 'modelnet',help='modelnet/shapenet/mnist')
    parser.add_argument('-model_name', type=str,default = 'ori', help='available ori/weight/density/sift')
    parser.add_argument('-rotate_num', type=int, default=40, help='the number of rotate number')
    parser.add_argument('-sigma_init', type=float, default=0.03, help='the value of initialized sigma')
    parser.add_argument('-sigma_weight', type=float, default=0.0035, help='the weight of sigma in the loss')
    parser.add_argument('-start_idx', type=int, default=0, help='the idx of start sample')
    parser.add_argument('-end_idx', type=int, default=50, help='the idx of start sample')
    args = parser.parse_args()
    return args


def main(args, idx, repeat_num, rotate_pram, Sigma_Init=0.03):
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_dataset = DataSetCls(num_points = args.num_points, root = args.data_root, transforms=test_transforms, train=False)

    model = RSCNN_DWS_Cls_Noisy(num_classes = args.num_classes, batch_size = args.batch_size, sigma_init = Sigma_Init ,input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, block_choice=args.model_name)

    model.cuda()
    optimizer = optim.Adam(model.add_noisy_by_point.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)

    if args.checkpoint is not '':
        model_dict = model.state_dict()
        pretrained_dict =  torch.load(args.checkpoint)
        # add module block
        if args.model_name != 'error':
            model_dict['fc1.fc.weight']=pretrained_dict['fc1.fc.weight']
            model_dict['fc1.bn.bn.weight']=pretrained_dict['fc1.bn.bn.weight']
            model_dict['fc1.bn.bn.bias']=pretrained_dict['fc1.bn.bn.bias']
            model_dict['fc1.bn.bn.running_mean']=pretrained_dict['fc1.bn.bn.running_mean']
            model_dict['fc1.bn.bn.running_var']=pretrained_dict['fc1.bn.bn.running_var']
            # model_dict['fc1.bn.bn.num_batches_tracked']=pretrained_dict['fc1.bn.bn.num_batches_tracked']
            model_dict['fc2.fc.weight']=pretrained_dict['fc2.fc.weight']
            model_dict['fc2.bn.bn.weight']=pretrained_dict['fc2.bn.bn.weight']
            model_dict['fc2.bn.bn.bias']=pretrained_dict['fc2.bn.bn.bias']
            model_dict['fc2.bn.bn.running_mean']=pretrained_dict['fc2.bn.bn.running_mean']
            model_dict['fc2.bn.bn.running_var']=pretrained_dict['fc2.bn.bn.running_var']
            # model_dict['fc2.bn.bn.num_batches_tracked']=pretrained_dict['fc2.bn.bn.num_batches_tracked']
            model_dict['fc3.fc.weight']=pretrained_dict['fc3.fc.weight']
            model_dict['fc3.fc.bias']=pretrained_dict['fc3.fc.bias']
        else:
            # original module
            model_dict['fc1.fc.weight']=pretrained_dict['FC_layer.0.fc.weight']
            model_dict['fc1.bn.bn.weight']=pretrained_dict['FC_layer.0.bn.bn.weight']
            model_dict['fc1.bn.bn.bias']=pretrained_dict['FC_layer.0.bn.bn.bias']
            model_dict['fc1.bn.bn.running_mean']=pretrained_dict['FC_layer.0.bn.bn.running_mean']
            model_dict['fc1.bn.bn.running_var']=pretrained_dict['FC_layer.0.bn.bn.running_var']
            model_dict['fc2.fc.weight']=pretrained_dict['FC_layer.2.fc.weight']
            model_dict['fc2.bn.bn.weight']=pretrained_dict['FC_layer.2.bn.bn.weight']
            model_dict['fc2.bn.bn.bias']=pretrained_dict['FC_layer.2.bn.bn.bias']
            model_dict['fc2.bn.bn.running_mean']=pretrained_dict['FC_layer.2.bn.bn.running_mean']
            model_dict['fc2.bn.bn.running_var']=pretrained_dict['FC_layer.2.bn.bn.running_var']
            model_dict['fc3.fc.weight']=pretrained_dict['FC_layer.4.fc.weight']
            model_dict['fc3.fc.bias']=pretrained_dict['FC_layer.4.fc.bias']
        #############################################################
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Load model successfully: %s' % (args.checkpoint))

    # training
    train(args, test_dataset, model, optimizer, lr_scheduler, bnm_scheduler, rotate_pram, idx, repeat_num)

def train(args, test_dataset, model, optimizer, lr_scheduler, bnm_scheduler, rotate_pram, idx, repeat_num):
    print('initial sigmaf:'+str(args.sigmaf))
    npz_dir = 'rs_'+args.model_name+'_'+args.dataset_name
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    model.eval()

    points, target = test_dataset.__getitem__(range(args.batch_size*idx,args.batch_size*(idx+1)))
    loss_val = []
    feature_loss_val = []
    pf_val = []
    sigma_val = []
    points = provider.rotate_point_cloud_randomangle(points, rotate_pram)
    points = torch.from_numpy(points).float()
    best_floss = 1e10
    for epoch in range(args.epochs):
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if bnm_scheduler is not None:
            bnm_scheduler.step(epoch-1)

        points, target = points.cuda(), target.cuda()
        points, target = Variable(points), Variable(target)

        optimizer.zero_grad()

        pred, sigma, end_points = model(points)
        loss, feature_loss, sigma_loss = get_similarity_loss(end_points[args.layer], sigma, args.sigmaf, args.sigma_weight)
        if epoch % 50 == 0:
            sigma_val.append(sigma.detach().cpu().numpy())
        if feature_loss.detach().cpu().numpy() - 2.0 < best_floss:
            best_floss = feature_loss.detach().cpu().numpy()
            best_sigma = sigma.detach().cpu().numpy()
        if epoch == 0 and feature_loss.detach().cpu().numpy() - 2.0 >= best_floss:
            print('sorry!This Sample is shit! The Feature loss is ',feature_loss.detach().cpu().numpy())
            break
        loss_val.append(loss.detach().cpu().numpy())
        feature_loss_val.append(feature_loss.detach().cpu().numpy())
        sigma_val.append(best_sigma)
        loss.backward()
        optimizer.step()
        if epoch % args.print_freq_iter == 0:
            print('[epoch %3d] \t train loss: %0.6f \t feature loss: %0.6f \t sigma loss: %0.6f \t lr: %0.5f' %(epoch+1, loss.data.clone(), feature_loss.data.clone(), sigma_loss.data.clone(), lr_scheduler.get_lr()[0]))
    if epoch != 0:
        np.savez(os.path.join(npz_dir, str(idx)+'_'+str(repeat_num)+'.npz'),loss = loss_val, feature_loss=feature_loss_val, sigma = sigma_val)

def mkdir_process(file_path):
    if not os.path.exists(file_path):
        # shutil.rmtree(file_path)
        os.makedirs(file_path)

def load_parament(json_path, dataset_name, model_name):
    with open(json_path,'r') as load_f:
        load_dict = json.load(load_f)
    data_dict = load_dict["dataset"][dataset_name]
    model_dict = load_dict["model"][model_name][dataset_name]

    num_classes = data_dict["num_classes"]
    data_root = data_dict["path"]
    checkpoint = model_dict["checkpoint"]
    sigmaf = model_dict["sigmaf"]

    return num_classes, data_root, checkpoint, sigmaf

if __name__ == "__main__":
    set_seed(0)
    args = argument()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    mkdir_process(os.path.join('rs_'+args.model_name+'_'+args.dataset_name))
    args.num_classes, args.data_root, args.checkpoint, args.sigmaf = load_parament(args.json, args.dataset_name, args.model_name)
    all_rotate_pram = np.load('./rotate_pram.npz')['rotate_pram']
    for idx in range(args.start_idx, args.end_idx):
        for repeat_num in range(args.rotate_num):
            rotate_pram = all_rotate_pram[idx, repeat_num].tolist()
            main(args, idx, repeat_num, rotate_pram, args.sigma_init)

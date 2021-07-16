import os
import yaml
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
import data.data_utils as d_utils
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils

#save_path = '/nfs-data/user4/new_results/'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
parser.add_argument('--config', default='cfgs/config_noisy.yaml', type=str)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)
print("\n**************************")
for k, v in config['common'].items():
    setattr(args, k, v)
    print('\n[%s]:'%(k), v)
print("\n**************************\n")
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def main(idx, Sigma_Init=0.03):    
    if args.model_name == 'rscnn':
        model = RSCNN_SSN_Cls_Noisy(num_classes = args.num_classes, sigma_init = Sigma_Init ,batch_size=args.batch_size, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    elif args.model_name == 'rscnn_msn':
        model = RSCNN_MSN_Cls_Noisy(num_classes = args.num_classes, sigma_init = Sigma_Init ,batch_size=args.batch_size, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    elif args.model_name == 'rscnn_density':
        model = RSCNN_DWS_Cls_Noisy(num_classes = args.num_classes, sigma_init = Sigma_Init ,input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, block_choice='density')
    elif args.model_name == 'rscnn_weight':
        model = RSCNN_DWS_Cls_Noisy(num_classes = args.num_classes, sigma_init = Sigma_Init ,input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, block_choice='weight')
    elif args.model_name == 'rscnn_sift':
        model = RSCNN_DWS_Cls_Noisy(num_classes = args.num_classes, sigma_init = Sigma_Init ,input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, block_choice='sift')
    
    model.cuda()
    optimizer = optim.Adam(model.add_noisy_by_point.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model_dict = model.state_dict()
        pretrained_dict =  torch.load(args.checkpoint)

        model_dict['fc1.fc.weight']=pretrained_dict['FC_layer.0.fc.weight']
        model_dict['fc1.bn.bn.weight']=pretrained_dict['FC_layer.0.bn.bn.weight']
        model_dict['fc1.bn.bn.bias']=pretrained_dict['FC_layer.0.bn.bn.bias']
        model_dict['fc1.bn.bn.running_mean']=pretrained_dict['FC_layer.0.bn.bn.running_mean']
        model_dict['fc1.bn.bn.running_var']=pretrained_dict['FC_layer.0.bn.bn.running_var']
        # model_dict['fc1.bn.bn.num_batches_tracked']=pretrained_dict['FC_layer.0.bn.bn.num_batches_tracked']
        model_dict['fc2.fc.weight']=pretrained_dict['FC_layer.2.fc.weight']
        model_dict['fc2.bn.bn.weight']=pretrained_dict['FC_layer.2.bn.bn.weight']
        model_dict['fc2.bn.bn.bias']=pretrained_dict['FC_layer.2.bn.bn.bias']
        model_dict['fc2.bn.bn.running_mean']=pretrained_dict['FC_layer.2.bn.bn.running_mean']
        model_dict['fc2.bn.bn.running_var']=pretrained_dict['FC_layer.2.bn.bn.running_var']
        # model_dict['fc2.bn.bn.num_batches_tracked']=pretrained_dict['FC_layer.2.bn.bn.num_batches_tracked']
        model_dict['fc3.fc.weight']=pretrained_dict['FC_layer.4.fc.weight']
        model_dict['fc3.fc.bias']=pretrained_dict['FC_layer.4.fc.bias']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Load model successfully: %s' % (args.checkpoint))
    
    # training
    return train(test_dataset, model, optimizer, lr_scheduler, bnm_scheduler, args, idx)
    
def train(test_dataset, model, optimizer, lr_scheduler, bnm_scheduler, args, idx):
    print('initial sigmaf:'+str(args.sigmaf))
    model.eval()
    
    points, target = test_dataset.__getitem__(range(args.batch_size*idx,args.batch_size*(idx+1)))
    loss_val = []
    feature_loss_val = []
    sigma_val = []
    for epoch in range(args.epochs):
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if bnm_scheduler is not None:
            bnm_scheduler.step(epoch-1)
        
        points, target = points.cuda(), target.cuda()
        points, target = Variable(points), Variable(target)
            
        optimizer.zero_grad()
        
        pred, sigma, end_points = model(points)
        loss, feature_loss, sigma_loss = get_similarity_loss(end_points[args.layer], sigma, args.sigmaf, sigma_weight = 0.003)
        if epoch % 100 == 0:
            sigma_val.append(sigma.detach().cpu().numpy())
        loss_val.append(loss.detach().cpu().numpy())
        feature_loss_val.append(feature_loss.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        if epoch % args.print_freq_iter == 0:
            print('[epoch %3d] \t train loss: %0.6f \t feature loss: %0.6f \t sigma loss: %0.6f \t lr: %0.5f' %(epoch+1, loss.data.clone(), feature_loss.data.clone(), sigma_loss.data.clone(), lr_scheduler.get_lr()[0]))
    
    return loss_val, feature_loss_val, sigma_val

if __name__ == "__main__":
    set_seed(args.seed)

    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_dataset = DataSetCls(num_points = args.num_points, root = args.data_root, transforms=test_transforms, train=False)
    sigma_list = []
    loss_list = []
    feature_loss_list = []
    for idx in range(len(test_dataset)//args.batch_size):
        
        loss_val, featureloss_val, sigma_val = main(idx)
        sigma_list.append(np.array(sigma_val))
        loss_list.append(np.array(loss_val))
        feature_loss_list.append(np.array(featureloss_val))

        np.savez(str(args.model_name)+'_sigma.npz', sigma=sigma_list, loss=loss_list, feature_loss=feature_loss_list)

import os
import numpy as np
import argparse
import random
import yaml

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
parser.add_argument('--config', default='cfgs/config_sigmaf.yaml', type=str)
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)
print("\n**************************")
for k, v in config['common'].items():
    setattr(args, k, v)
    print('\n[%s]:'%(k), v)
print("\n**************************\n")
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def main(Sigma_Init=0.007):
    set_seed(args.seed)

    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    
    test_dataset = DataSetCls(num_points = args.num_points, root = args.data_root, transforms=train_transforms,train=False)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        drop_last = True,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=False
    )
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
        model_dict['fc2.fc.weight']=pretrained_dict['FC_layer.2.fc.weight']
        model_dict['fc2.bn.bn.weight']=pretrained_dict['FC_layer.2.bn.bn.weight']
        model_dict['fc2.bn.bn.bias']=pretrained_dict['FC_layer.2.bn.bn.bias']
        model_dict['fc2.bn.bn.running_mean']=pretrained_dict['FC_layer.2.bn.bn.running_mean']
        model_dict['fc2.bn.bn.running_var']=pretrained_dict['FC_layer.2.bn.bn.running_var']
        model_dict['fc3.fc.weight']=pretrained_dict['FC_layer.4.fc.weight']
        model_dict['fc3.fc.bias']=pretrained_dict['FC_layer.4.fc.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Load model successfully: %s' % (args.checkpoint))
    
    num_batch = len(test_dataset)/args.batch_size
    
    # training
    evaluate(test_dataloader, model, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)
    

def evaluate(test_dataloader, model, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    model.eval()
    sigmaf_sum = 0
    for epoch in range(20):
        print('-----epoch:'+str(epoch)+'-----')
        feature_loss_sum = 0
        batch_count = 0
        for i, data in enumerate(test_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            
            optimizer.zero_grad()
            
            pred, sigma, end_points = model(points)
            loss, feature_loss, sigma_loss = get_similarity_loss(end_points[args.layer], sigma, None)
            feature_loss_sum += feature_loss.detach()
            batch_count += 1
        sigmaf_sum += feature_loss_sum/batch_count
        print('feature loss: %f' %(feature_loss_sum/batch_count))
    print(sigmaf_sum/20)

if __name__ == "__main__":
    main(args.sigma_init)
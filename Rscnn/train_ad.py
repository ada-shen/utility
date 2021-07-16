import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls_Ad 
from models import RSCNN_MSN_Cls_Ad
from models import RSCNN_DWS_Cls_Ad
from models import get_adversarial_loss
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import provider
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ModelNet_File = '/nfs-data/user4/dataset/modelnet400/test_files.txt'
#ShapeNet_File = '/nfs-data/user4/dataset/shapenet_0/test_files.txt'
#Mnist_File = '/nfs-data/user4/dataset/3d-minist0/test_files.txt'
#save_path = '/root/project/results/'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
parser.add_argument('--config', default='cfgs/config_ad.yaml', type=str)

args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)
# print("\n**************************")
for k, v in config['common'].items():
    setattr(args, k, v)
    # print('\n[%s]:'%(k), v)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
Num_classes=args.num_classes

ModelNet_File = args.ModelNet_File
ShapeNet_File = args.ShapeNet_File
Mnist_File = args.Mnist_File

def get_dataset(dataset_name):
    data = None
    label = None
    if dataset_name == 'modelnet':
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, ModelNet_File))
    elif dataset_name == 'mnist':
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, Mnist_File))
    elif dataset_name == 'shapenet':
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, ShapeNet_File))
    
    #Merge data from multiple files  
    for fn in range(len(TEST_FILES)):
        tmp_data, tmp_label = provider.loadDataFile(TEST_FILES[fn])
        data = merge(data,tmp_data)
        label = merge(label,tmp_label)
    return data, label

def main(data, target_label, model_name):
    if model_name == 'rscnn_density':
        model = RSCNN_DWS_Cls_Ad(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, block_choice='density')
    elif model_name == 'rscnn_weight':
        model = RSCNN_DWS_Cls_Ad(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, block_choice='weight')
    elif model_name == 'rscnn_sift':
        model = RSCNN_DWS_Cls_Ad(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True, block_choice='sift')
    elif model_name == 'rscnn_msn':
        model = RSCNN_MSN_Cls_Ad(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    elif model_name == 'rscnn':
        model = RSCNN_SSN_Cls_Ad(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)

    model.cuda()
    optimizer = optim.Adam(model.perturbation_point_xyz.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model_dict = model.state_dict()
        pretrained_dict =  torch.load(args.checkpoint)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # print('Load model successfully: %s' % (args.checkpoint))
    criterion = nn.CrossEntropyLoss()
    # training
    return train(model, optimizer, criterion, lr_scheduler, bnm_scheduler, data, target_label)

def train(model, optimizer, criterion, lr_scheduler, bnm_scheduler, data, target):
    model.eval()
    target_label = target
    points, target = torch.from_numpy(data).float().cuda(), torch.from_numpy(target).cuda()
    points, target = Variable(points), Variable(target)
    for epoch in range(5000):
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if bnm_scheduler is not None:
            bnm_scheduler.step(epoch-1)
        
        optimizer.zero_grad()
        
        pred, pert_vec = model(points)
        loss, clr_loss, pert_loss = get_adversarial_loss(pred, target, pert_vec, criterion, lam = 1)
        
        pred_val = np.argmax(pred.detach().cpu().numpy(), 1)
        correct = np.sum(pred_val == target_label)
        if(correct == 1):
            print('epoch: '+str(epoch)+' loss:'+str(loss.data)+'clr_loss:'+str(clr_loss.data)+'pert_loss:'+str(pert_loss.data))
            break
        loss.backward()
        optimizer.step()
    return pert_loss.detach().cpu().numpy()

def merge(all,tmp):
    if all is None:
        all = tmp
    else:
        all = np.concatenate((all,tmp),axis=0)
    return all

if __name__ == "__main__":
    set_seed(args.seed)
    data, label = get_dataset(args.dataset)
    delta = []
    sample_sum = 0
    if args.dataset == 'modelnet':
        total_sample = 2400
        step = 24
    elif args.dataset == 'mnist':
        total_sample = 1000
        step = 10
    elif args.dataset == 'shapenet':
        total_sample = 2800
        step = 28
    for i in range(0,total_sample,step):
        start_time = time.time()
        use_data = data[i:i+1,:args.num_points,:]
        use_label = label[i]
        target_labels = np.arange(Num_classes)
        target_labels = target_labels[target_labels!=use_label]
        sample_delta_sum = 0
        for target in target_labels:
            print('-------------sample---'+str(i)+'--------------')
            sample_delta_sum += (main(use_data,np.reshape(target,(-1,)), args.model_name))
        sample_delta = sample_delta_sum / target_labels.shape[0]
        delta.append(sample_delta)
        print('sample '+ str(i)+': '+ str(sample_delta))
        end_time = time.time()
        print('time:' , end_time-start_time)
        sample_sum += sample_delta
    print('Adversarial:', args.model_name, sample_sum/100)
    np.savez(args.model_name+'_'+args.dataset+'.npz', ad=np.array(delta))




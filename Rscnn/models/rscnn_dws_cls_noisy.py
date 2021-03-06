import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules_fixsample import PointnetSAModule, PointnetSAModuleMSG
from noisy import add_noisy_by_point
import numpy as np

# Relation-Shape CNN: Single-Scale Neighborhood
class RSCNN_SSN(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, sigma_init, batch_size, input_channels=0, relation_prior=1, use_xyz=True, block_choice=None):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.add_noisy_by_point = add_noisy_by_point(batch_size, 1024, sigma_init)
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.23],
                nsamples=[48],
                mlps=[[input_channels, 128]],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior,
                block_choice = block_choice
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.32],
                nsamples=[64],
                mlps=[[128, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior,
                block_choice = block_choice
            )
        )
        
        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample = 128,
                mlp=[512, 1024], 
                use_xyz=use_xyz,
                block_choice = None,
            )
        )

        self.fc1 = pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = pt_utils.FC(512, 256, activation=nn.ReLU(inplace=True), bn=True)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = pt_utils.FC(256, num_classes, activation=None)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        end_points = {}
        xyz, features = self._break_up_pc(pointcloud)
        xyz, sigma = self.add_noisy_by_point(xyz)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        
        x = self.fc1(features.squeeze(-1))
        end_points['fc1'] = x
        x = self.dp1(x)
        x = self.fc2(x)
        end_points['fc2'] = x
        x = self.dp2(x)
        x = self.fc3(x)
        end_points['fc3'] = x
        return x, sigma, end_points


if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 2048, 6))
    sim_data = sim_data.cuda()
    sim_cls = Variable(torch.ones(32, 16))
    sim_cls = sim_cls.cuda()

    seg = RSCNN_SSN(num_classes=50, input_channels=3, use_xyz=True)
    seg = seg.cuda()
    out = seg(sim_data, sim_cls)
    print('seg', out.size())
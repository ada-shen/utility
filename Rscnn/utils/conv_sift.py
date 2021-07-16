import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import math
#from sklearn.neighbors.kde import KernelDensity

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

## newly added farthest_point_sample with permutation invariance ##
def fixed_farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint+1):
        if i == 0:
            centroid = torch.mean(xyz,axis=1,keepdims=True)
        else:
            centroids[:, i-1] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, nsample, xyz, points, density_scale = None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

def sample_and_group_all(xyz, points, density_scale = None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    #new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim = 1, keepdim = True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density

def group(nsample, xyz, points):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

######  PointConv Module  #########
def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density

class DensityNet(nn.Module):
    def __init__(self, hidden_unit = [8, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv1d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm1d(1))

    def forward(self, xyz_density):
        B, N = xyz_density.shape
        density_scale = xyz_density.unsqueeze(1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale =  bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale) + 0.5
            else:
                density_scale = F.relu(density_scale)

        return density_scale

class PointConvDensity(nn.Module):
    def __init__(self):
        super(PointConvDensity, self).__init__()
        self.densitynet = DensityNet()
        self.weightnet = WeightNet(3,16)
        self.bandwidth = 0.1
    def forward(self, xyz, grouped_xyz_norm, idx, group_points):
        B = xyz.shape[0]
        N = xyz.shape[1]
        xyz_density = compute_density(xyz, self.bandwidth)
        density_scale = self.densitynet(xyz_density)
        grouped_density = index_points(density_scale.view(B,N,1), idx.long())
        grouped_xyz = grouped_xyz_norm
        grouped_xyz = grouped_xyz * grouped_density.permute(0,3,1,2)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=group_points.permute(0, 2, 1, 3), other = weights.permute(0, 2, 3, 1))
        new_points = new_points.permute(0, 2, 1, 3).contiguous()
        return new_points

class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        #xyz : BxCxKxN
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))
        return weights

class PointConvWeight(nn.Module):
    def __init__(self, mid_channel = 16):
        super(PointConvWeight, self).__init__()
        self.weightnet = WeightNet(3, mid_channel)

    def forward(self, group_xyz, group_points):
        weights = self.weightnet(group_xyz) # (B, 16, npoint, nsample) group_points (B, C, npoint, nsample)
        # print('lalal:', group_points.size(), weights.size())
        new_points = torch.matmul(input=group_points.permute(0, 2, 1, 3).contiguous(), other = weights.permute(0, 2, 3, 1).contiguous())
        new_points = new_points.permute(0, 2, 1, 3).contiguous()
        return new_points
        
class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = compute_density(xyz, self.bandwidth)
        #import ipdb; ipdb.set_trace()
        density_scale = self.densitynet(xyz_density)

      
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
    
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        grouped_xyz = grouped_xyz * grouped_density.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points

######  PointSift Module  #########
def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride),
        nn.BatchNorm2d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.ReLU())
    return seq

class PointSIFT_module_basic(nn.Module):
    def __init__(self):
        super(PointSIFT_module_basic, self).__init__()

    def index_points(self, points, idx):
        """
        Description:
            this function select the specific points from the whole points according to the idx.
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, D1, D2, ..., Dn]
        Return:
            new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        # print("lalla:",points.shape, batch_indices.shape, idx.shape)
        new_points = points[batch_indices, idx.long(), :]
        return new_points

    def pointsift_select_c(self, radius, xyz):
        """
        code by c/c++ logic
        :param radius:
        :param xyz:
        :return:
        """
        Dist = lambda x, y, z: x ** 2 + y ** 2 + z ** 2
        B, N, _ = xyz.shape
        idx = torch.empty(B, N, 8)
        judge_dist = radius ** 2
        temp_dist = torch.ones(B, N, 8) * 1e10
        for b in range(B):
            for n in range(N):
                idx[b, n, :] = n
                x, y, z = xyz[b, n]
                for p in range(N):
                    if p == n: continue
                    tx, ty, tz = xyz[b, p]
                    dist = Dist(x - tx, y - ty, z - tz)
                    if dist > judge_dist: continue
                    _x, _y, _z = tx > x, ty > y, tz > z
                    temp_idx = (_x * 4 + _y * 2 + _z).int()
                    if dist < temp_dist[b, n, temp_idx]:
                        idx[b, n, temp_idx] = p
                        temp_dist[b, n, temp_idx] = dist
        return idx.int()

    def pointsift_select(self, radius, xyz):
        """
        code by python matrix logic
        :param radius:
        :param xyz:
        :return: idx
        """
        dev = xyz.device
        B, N, _ = xyz.shape
        judge_dist = radius ** 2
        idx = torch.arange(N).repeat(8, 1).permute(1, 0).contiguous().repeat(B, 1, 1).to(dev)
        for n in range(N):
            distance = torch.ones(B, N, 8).to(dev) * 1e10
            distance[:, n, :] = judge_dist
            centroid = xyz[:, n, :].view(B, 1, 3).to(dev)
            dist = torch.sum((xyz - centroid) ** 2, -1)  # shape: (B, N)
            subspace_idx = torch.sum((xyz - centroid + 1).int() * torch.tensor([4, 2, 1], dtype=torch.int, device=dev),
                                     -1)
            for i in range(8):
                mask = (subspace_idx == i) & (dist > 1e-10) & (dist < judge_dist)  # shape: (B, N)
                distance[..., i][mask] = dist[mask]
                idx[:, n, i] = torch.min(distance[..., i], dim=-1)[1]
        return idx

    def pointsift_group(self, radius, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        assert C == 3
        idx = self.pointsift_select(radius, xyz)  # B, N, 8

        grouped_xyz = self.index_points(xyz, idx)  # B, N, 8, 3
        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.index_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points, idx

    def pointsift_group_with_idx(self, idx, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        grouped_xyz = self.index_points(xyz, idx)  # B, N, 8, 3
        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.index_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points

class PointSIFT_res_module(PointSIFT_module_basic):

    def __init__(self, radius, output_channel, extra_input_channel=0, merge='add', same_dim=False):
        super(PointSIFT_res_module, self).__init__()
        self.radius = radius
        self.merge = merge
        self.same_dim = same_dim

        self.conv1 = nn.Sequential(
            conv_bn(3 + extra_input_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        )

        self.conv2 = nn.Sequential(
            conv_bn(3 + output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2], activation=None)
        )
        if same_dim:
            self.convt = nn.Sequential(
                nn.Conv1d(extra_input_channel, output_channel, 1),
                nn.BatchNorm1d(output_channel),
                nn.ReLU()
            )

    def forward(self, xyz, points):
        xyz = xyz.permute(0,2,1).contiguous()
        points = points.permute(0,2,1).contiguous()
        _, grouped_points, idx = self.pointsift_group(self.radius, xyz, points)  # [B, N, 8, 3], [B, N, 8, 3 + C]

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, C, N, 8
        new_points = self.conv1(grouped_points)
        new_points = new_points.squeeze(-1).permute(0, 2, 1).contiguous()

        _, grouped_points = self.pointsift_group_with_idx(idx, xyz, new_points)
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

        new_points = self.conv2(grouped_points)

        new_points = new_points.squeeze(-1)

        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
            # print(points.shape)
            if self.same_dim:
                points = self.convt(points)
            if self.merge == 'add':
                new_points = new_points + points
            elif self.merge == 'concat':
                new_points = torch.cat([new_points, points], dim=1)

        new_points = F.relu(new_points)
        new_points = new_points#.permute(0, 2, 1).contiguous()

        return xyz, new_points
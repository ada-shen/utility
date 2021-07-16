import torch
import torch.nn as nn

class add_noisy_by_point(nn.Module):
    def __init__(self, batch_size, point_num, sigma_init):
        super().__init__()
        self.sigma = nn.Parameter(torch.Tensor(batch_size, point_num), requires_grad=True)
        #self.register_parameter(name='sigma', param=self.sigma)
        nn.init.constant_(self.sigma, 0.03)
        
    def forward(self, point_cloud):
        B, N, C = point_cloud.shape
        epsional = torch.randn(point_cloud.shape).cuda()

        ### clip operation
        sigma_val = torch.clamp(self.sigma, -0.08, 0.08)
        sigma_val = sigma_val.unsqueeze(-1)

        ### no clip operation ###
        # sigma_val = self.sigma.unsqueeze(-1)
        
        sigma_val = sigma_val.repeat(1,1,C)
        noisy = sigma_val.mul(epsional)
        new_point = torch.add(point_cloud, noisy)
        point_merge = torch.cat((point_cloud, new_point), 0)
        return point_merge, self.sigma

class perturbation_point_xyz(nn.Module):
    def __init__(self, batch_size, point_num, channel):
        super().__init__()
        self.epsilon = nn.Parameter(torch.Tensor(batch_size, point_num, channel), requires_grad=True)
        # self.register_parameter(name='sigma', param=self.epsilon)
        nn.init.xavier_uniform_(self.epsilon)

    def forward(self, point_cloud):
        pert_pc = torch.add(point_cloud, self.epsilon)
        return pert_pc, self.epsilon


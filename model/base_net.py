import numpy as np
import torch
import torch.nn as nn
from model.st_gcn import GCN_TCN_Unit, UpSP_GCN_TCN_Unit

class ST_Gen(nn.Module):
    def __init__(self, in_channels, joint_num):
        super(ST_Gen, self).__init__()
        self.down_l1 = GCN_TCN_Unit(in_channels, 64, joint_num=joint_num, residual=False)
        self.down_l2 = GCN_TCN_Unit(64, 128, joint_num=joint_num)
        self.down_l3 = GCN_TCN_Unit(128, 256, joint_num=joint_num)

        self.up_l1 = GCN_TCN_Unit(256, 128, joint_num=joint_num)
        self.up_l2 = GCN_TCN_Unit(128, 64, joint_num=joint_num)
        self.up_l3 = GCN_TCN_Unit(64, in_channels, joint_num=joint_num, residual=False, activate=False, ifbn=False)
    
    def forward(self, x):
        x = self.down_l1(x)
        x = self.down_l2(x)
        x256 = self.down_l3(x)

        x = self.up_l1(x256)
        x = self.up_l2(x)
        x = self.up_l3(x)

        return x

class ST_Dis(nn.Module):
    def __init__(self, in_channels, joint_num):
        super(ST_Dis, self).__init__()
        self.l1 = GCN_TCN_Unit(in_channels, 64, joint_num=joint_num, residual=False)
        self.l2 = GCN_TCN_Unit(64, 64, joint_num=joint_num, stride=2)
        self.l3 = GCN_TCN_Unit(64, 128, joint_num=joint_num, stride=2)
        self.l4 = GCN_TCN_Unit(128, 128, joint_num=joint_num, residual=False, activate=False)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.mean(1)        # B, T, V
        return x



import numpy as np
import torch
import torch.nn as nn

class TCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, ifbn=True):
        super(TCN_Unit, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.reflect_pad = nn.ReflectionPad2d((0, 0, 2*pad, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1))
        if ifbn:
            self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.bn = lambda x: x
    
    def forward(self, x):
        # input shape B,C,T,V
        x = self.bn(self.conv(self.reflect_pad(x)))
        return x

class UpSP_TCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=6, stride=1, ifbn=True):
        super(UpSP_TCN_Unit, self).__init__()
        pad = int((kernel_size - stride) / 2)
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(pad, 0))
        if ifbn:
            self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.bn = lambda x: x
    
    def forward(self, x):
        x = self.bn(self.trans_conv(x))
        return x

class GCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, coff_embedding=4, joint_num=25, activate=True, ifbn=True):
        super(GCN_Unit, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        # self.inter_c = out_channels // coff_embedding if out_channels > 4 else in_channels // coff_embedding
        # self.PA = nn.Parameter(self.L1_norm(torch.randn((joint_num, joint_num), dtype=torch.float32)), requires_grad=True)
        self.PA = nn.Parameter(torch.randn((joint_num, joint_num), dtype=torch.float32), requires_grad=True)

        # self.conv_q = nn.Conv2d(in_channels, self.inter_c, 1)
        # self.conv_k = nn.Conv2d(in_channels, self.inter_c, 1)
        # self.soft = nn.Softmax(-2)
        self.conv_d = nn.Conv2d(in_channels, out_channels, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        if ifbn:
            self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.bn = lambda x: x
        if activate:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = lambda x:x
    
    def L1_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 1, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A
    
    def forward(self, x):
        N, C, T, V = x.size()

        # A1 = self.conv_q(x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
        # A2 = self.conv_k(x).view(N, self.inter_c * T, V)
        # TA = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
        # A = self.PA + TA
        A = self.PA
        gx = x.view(N, C * T, V)
        y = self.conv_d(torch.matmul(gx, A).view(N, C, T, V))
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class GCN_TCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, coff_embedding=4, joint_num=25, residual=True, activate=True, ifbn=True):
        super(GCN_TCN_Unit, self).__init__()
        self.gcn = GCN_Unit(in_channels, out_channels, coff_embedding=coff_embedding, joint_num=joint_num, activate=activate, ifbn=ifbn)
        self.tcn = TCN_Unit(out_channels, out_channels, stride=stride, ifbn=ifbn)
        
        if activate:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = lambda x:x

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TCN_Unit(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn(self.gcn(x)) + self.residual(x)
        y = self.relu(x)
        return y

class UpSP_GCN_TCN_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, coff_embedding=4, joint_num=25, residual=True, activate=True, ifbn=True):
        super(UpSP_GCN_TCN_Unit, self).__init__()
        self.gcn = GCN_Unit(in_channels, out_channels, coff_embedding=coff_embedding, joint_num=joint_num, activate=activate, ifbn=ifbn)
        self.tcn = UpSP_TCN_Unit(out_channels, out_channels, stride=stride, ifbn=ifbn)
        if activate:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = lambda x:x

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UpSP_TCN_Unit(in_channels, out_channels, kernel_size=2, stride=stride)

    def forward(self, x):
        x = self.tcn(self.gcn(x)) + self.residual(x)
        return self.relu(x)



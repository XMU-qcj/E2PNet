import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
from utils import *
import math

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.type = type
    def forward(self, x, y): 
        x_q = self.q_conv(y).permute(0,2,1)  # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-7 + attention.sum(dim=1, keepdims=True))
        x_r = (x_v @ attention) # b, c, n 
        x_r = (self.act(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class STA(nn.Module):
    def __init__(self, in_channel):
        super(STA, self).__init__()
        self.time = Attention(in_channel)
        self.space = Attention(in_channel)
        self.time_cor = Attention(in_channel)
        self.space_cor = Attention(in_channel)
        self.time_res = Attention(in_channel)
        self.space_res = Attention(in_channel)
        self.global_res = Attention(in_channel)
        self.output = nn.Conv1d(in_channel, in_channel, 1)
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x): 
        B, C, N = x[:,1].shape
        time_x = x[:,0]
        space_x = x[:,2]
        global_feat = self.global_res(x[:,1], x[:,1])
        time_attention = self.time(time_x, time_x)
        space_attention = self.space(space_x, space_x)

        time_attention_cor = self.time_cor(time_x, space_x)
        space_attention_cor = self.space_cor(space_x, time_x)

        time_sig = self.sig(time_attention_cor)
        space_sig = self.sig(space_attention_cor)
        time_residual = time_attention * time_sig
        space_residual = space_attention * space_sig

        time_update = self.tanh(time_residual)
        space_update = self.tanh(space_residual)

        time_global = global_feat
        space_global = global_feat
        time_global = self.sig(time_global)
        space_global = self.sig(space_global)

        time_global_residual = (time_update * time_global).reshape(x[:,1].shape)
        space_global_residual = (space_update * space_global).reshape(x[:,1].shape)

        global_residual = time_global_residual + space_global_residual
        global_residual = self.sig(global_residual)

        global_feat = self.tanh(global_feat)
        global_output = global_residual * global_feat
        return x[:,1] + global_output
    
class PointNetEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(PointNetEncoder, self).__init__()

        self.sa1 = PointNetSetAbstraction(radius= (12/256.) ** 2, nsample = 64, in_channel=3+4, mlp=[16, 32, 64], weight = [[0.1, 0.8], [0.5, 0.5], [0.8, 0.1]])
        self.fp1 = PointNetFeaturePropagation(64+4, [64, 64, 64])
        self.attention = STA(64)
        self.output = nn.Conv1d(64, embedding_size, 1)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.output2 = nn.Conv1d(embedding_size, 64, 1)
    def forward(self, xyt, center, t_list):
        
        B, _, _ = xyt.shape
        l0_xyt = xyt[:,:,:3] 
        l0_point = xyt
        mask_xyt = center[:,:,:3]
        # LA
        l1_xyt, l1_points = self.sa1(l0_xyt, l0_point, mask_xyt, t_list)
        # STA
        l1_points = self.attention(l1_points)
        # FP
        l0_points = self.fp1(l0_xyt.transpose(2,1), l1_xyt[:, 1, ...], l0_point, l1_points)
        x = F.relu(self.bn1(self.output(l0_points)))
        x = self.output2(x)
        return x
            
class PointNetAutoencoder(nn.Module):
    def __init__(
        self, embedding_size, input_channels=4, feat_compute = False, output_channels=4, normalize=True
    ):
        super(PointNetAutoencoder, self).__init__()
        self.normalize = normalize
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embedding_size = embedding_size
        self.encoder = PointNetEncoder(embedding_size)
        
        
    def forward(self, x, center, t_list):
        z = self.encode(x, center, t_list)
        return z

    def encode(self, x, center, t_list):
        z = self.encoder(x, center, t_list)
        
        
        if self.normalize:
            z = F.normalize(z)
        return z

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
sys.path.append("/media/XH-8T/qcj/cGAN_new_re/networks")
from utils_pointnet import *
sys.path.append("/media/XH-8T/qcj/Dataset")
from event_utils import *

class Attention_div(nn.Module):
    def __init__(self, channels, type = 'self'):
        super().__init__()
        self.q_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.trans_conv = nn.Conv2d(channels, channels, 1)
        self.after_norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.type = type
    def forward(self, x, y):
        if (self.type == 'self'):
            x_q = self.q_conv(y).permute(0,1,3,2) # b, n, c    
        else:
            x_q = self.q_conv(y)
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-7 + attention.sum(dim=1, keepdims=True))
        x_r = (x_v @ attention) # b, c, n 
        x_r = (self.act(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Attention_once(nn.Module):
    def __init__(self, channels, type = 'self'):
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
        if (self.type == 'self'):
            x_q = self.q_conv(y).permute(0,2,1) # b, n, c    
        else:
            x_q = self.q_conv(y)
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-7 + attention.sum(dim=1, keepdims=True))
        x_r = (x_v @ attention) # b, c, n 
        x_r = (self.act(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    
class attention(nn.Module):
    def __init__(self, in_channel):
        super(attention, self).__init__()
        self.time = Attention_div(in_channel)
        self.space = Attention_div(in_channel)
        self.time_cor = Attention_div(in_channel, 'cor')
        self.space_cor = Attention_div(in_channel, 'cor')
        self.time_res = Attention_div(in_channel)
        self.space_res = Attention_div(in_channel)
        self.output = nn.Conv1d(in_channel * 2, in_channel, 1)
    def forward(self, x):
        #print(x.shape) torch.Size([4, 32, 5120])
        B, C, N = x.shape
        time_x = x.reshape((B, C, N // 5, 5)).reshape((B, C, N // 5, 5)) # 4, 32, 1024, 5
        space_x = x.reshape((B, C, 5,  N // 5)).reshape((B, C,  5, N // 5)) # 4, 32, 5, 1024
        time_attention = self.time(time_x, time_x).reshape((B, C, N // 5, 5))
        space_attention = self.space(space_x, space_x).reshape((B, C, 5, N // 5))
        time_attention_cor = self.time_cor(time_x, space_x)
        space_attention_cor = self.space_cor(space_x, time_x)
        time_attention_res = self.time_res(time_attention_cor, time_attention).reshape(x.shape)
        space_attention_res = self.time_res(space_attention_cor, space_attention).reshape(x.shape)
        attention_res = torch.cat([time_attention_res, space_attention_res], 1)
        attention_res = self.output(attention_res)
#         attention_res = torch.cat([x, x], 1)
        return attention_res

class attention2(nn.Module):
    def __init__(self, in_channel):
        super(attention2, self).__init__()
        self.time = Attention_div(in_channel)
        self.space = Attention_div(in_channel)
        self.time_cor = Attention_div(in_channel, 'cor')
        self.space_cor = Attention_div(in_channel, 'cor')
        self.time_res = Attention_div(in_channel)
        self.space_res = Attention_div(in_channel)
        self.global_res = Attention_once(in_channel)
        self.output = nn.Conv1d(in_channel, in_channel, 1)
        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        #print(x.shape) torch.Size([4, 32, 5120])
        B, C, N = x.shape
        time_x = x.reshape((B, C, N // 5, 5)).reshape((B, C, N // 5, 5)) # 4, 32, 1024, 5
        space_x = x.reshape((B, C, 5,  N // 5)).reshape((B, C,  5, N // 5)) # 4, 32, 5, 1024
        global_feat = self.global_res(x, x)
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

        time_global = global_feat.reshape((B, C, N // 5, 5)).reshape((B, C, N // 5, 5))
        space_global = global_feat.reshape((B, C, 5,  N // 5)).reshape((B, C,  5, N // 5))
        time_global = self.sig(time_global)
        space_global = self.sig(space_global)

        time_global_residual = (time_update * time_global).reshape(x.shape)
        space_global_residual = (space_update * space_global).reshape(x.shape)

        global_residual = time_global_residual + space_global_residual
        global_residual = self.sig(global_residual)

        global_feat = self.tanh(global_feat)
        global_output = global_residual * global_feat
        # global_output = self.output(global_output)
        return x + global_output
import math

def normalize_event_volume(event_volume):
        event_volume_flat = event_volume.view(-1)  # 展成一维
        nonzero = torch.nonzero(event_volume_flat)  # 找出非零索引
        nonzero_values = event_volume_flat[nonzero]  # 取出非零
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.00001 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.99999 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            event_volume = torch.clamp(event_volume, -max_val, max_val)
            event_volume /= max_val
        return event_volume
def random_sample_events(events, k):
        events = events.transpose(1,0)
        if events.shape[1] >= k:
            choice_idx = np.random.choice(events.shape[1], k, replace=False)
        else:
            fix_idx = np.asarray(range(events.shape[1]))
            while events.shape[1] + fix_idx.shape[0] < k:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(events.shape[1]))), axis=0)
            random_idx = np.random.choice(events.shape[1], k - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        events = events[:, choice_idx]
        return events.transpose(1,0)
class PointNetEncoder(nn.Module):
    def __init__(self, embedding_size, input_channels=3, feat_compute = None, cycle_num = 5):
        super(PointNetEncoder, self).__init__()
        
        self.sa1 = PointNetSetAbstraction(npoint = 2048,radius= (12/255.) ** 2, nsample = 64, in_channel=3 + 1, mlp=[16, 32, 64], group_all=False)
        self.fp1 = PointNetFeaturePropagation(64+4, [128, 128, 256])
        self.cycle_num = cycle_num
        self.pcd_num = 4096
        self.feat_compute = feat_compute
        self.attention = attention2(64)
        self.attention2 = attention(64)
        if (self.feat_compute == False):
            self.output = nn.Conv1d(256, embedding_size, 1)
            self.bn1 = nn.BatchNorm1d(embedding_size)
            self.output2 = nn.Conv1d(embedding_size, embedding_size, 1)
        else:
            self.sa3 = PointNetSetAbstraction(radius=None, nsample=None, in_channel=256 + 4, mlp=[256, 256], group_all=True)
            self.bn1 = nn.BatchNorm1d(embedding_size)
            self.output = nn.Linear(256, embedding_size)
            self.output2 = nn.Linear(embedding_size, embedding_size)
    def forward(self, xyz, mask, t_list):
        # 1是原来的，2是pointnet++原版
        # for i in range(xyz.shape[0]):
        #     mask[i] = random_sample_events(xyz[i], 4805)
        B, _, _ = xyz.shape
        l0_xyz = xyz[:,:,:3]
        l0_point = xyz[:,:,3].unsqueeze(-1)
        l0_point_all = xyz
        mask_xyz = mask[:,:,:3]
        l1_xyz, l1_points = self.sa1(l0_xyz.transpose(2,1), l0_point.transpose(2,1))
        # l1_points = self.attention(l1_points)
        l0_points = self.fp1(l0_xyz.transpose(2,1), l1_xyz, l0_point_all.transpose(2,1), l1_points)
        
        if (self.feat_compute == False):
            x = F.relu(self.bn1(self.output(l0_points)))
            x = self.output2(x)
            return x
        else:
            l3_xyz, l3_points = self.sa3(l0_point_all, l0_points, mask_xyz, t_list)
            x = l3_points.view(B, 256)
            x = F.relu(self.bn1(self.output(x)))
            x = self.output2(x)
            # x = F.log_softmax(x, -1)
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

        self.encoder = PointNetEncoder(embedding_size, input_channels, feat_compute)
        
        
    def forward(self, x, flow, flow_ori, mask, t_list):
        flow_res = self.encoder(flow, mask, t_list)
        # flow_volume_temp = gen_event_volume(flow, flow_res.transpose(-2,-1),[x.shape[0], 6, x.shape[2], x.shape[3]])
        
        for i in range(flow.shape[0]):
            flow_input = flow_ori[i]
            flow_res_input = flow_res[i].transpose(-2,-1)

            flow_volume_output = (gen_discretized_event_volume(events = flow_input, val_plus = True, events_val = flow_res_input, vol_size = [6, x.shape[2], x.shape[3]]))
            flow_time_output = (per_event_timing_images(flow_input, flow_res_input, [2, x.shape[2], x.shape[3]]))
            flow_stacking_output = (per_stacking_events(flow_input, flow_res_input, [3, x.shape[2], x.shape[3]]))
            flow_counting_output = (per_event_counting_images(flow_input, flow_res_input, [2, x.shape[2], x.shape[3]]))

            flow_volume_output = normalize_event_volume(flow_volume_output)
            flow_time_output = normalize_event_volume(flow_time_output)
            flow_stacking_output = normalize_event_volume(flow_stacking_output)
            flow_counting_output = normalize_event_volume(flow_counting_output)

            flow_volume_output = flow_volume_output.unsqueeze(0)
            flow_time_output = flow_time_output.unsqueeze(0)
            flow_stacking_output = flow_stacking_output.unsqueeze(0)
            flow_counting_output = flow_counting_output.unsqueeze(0)
            if (i == 0):
                flow_time_res = flow_time_output
                flow_stacking_res = flow_stacking_output
                flow_counting_res = flow_counting_output
                flow_volume_res = flow_volume_output
            else:
                flow_time_res = torch.cat([flow_time_res, flow_time_output], 0)
                flow_stacking_res = torch.cat([flow_stacking_res, flow_stacking_output], 0)
                flow_counting_res = torch.cat([flow_counting_res, flow_counting_output], 0)
                flow_volume_res = torch.cat([flow_volume_res, flow_volume_output], 0)
        # import cv2
        # print(x.shape)
        # print(flow_stacking_res.shape)
        # print(flow_volume_res.shape)
        # print(flow_counting_res.shape)
        # print(flow_time_res.shape)
        # cv2.imwrite("./33.png", x[0, :3].permute(1,2,0).cpu().numpy() * 255)
        # cv2.imwrite("./44.png", flow_stacking_res[0].permute(1,2,0).cpu().numpy() * 255)
        # cv2.imwrite("./55.png", flow_volume_res[0,:3].permute(1,2,0).cpu().numpy() * 255)
        # cv2.imwrite("./66.png", flow_counting_res[0].permute(1,2,0)[...,0].cpu().numpy() * 255)
        # cv2.imwrite("./77.png", flow_time_res[0].permute(1,2,0)[...,0].cpu().numpy() * 255)
        # print(dsadsa)
        flow_res = torch.cat([flow_stacking_res, flow_volume_res, flow_counting_res, flow_time_res], 1)
        # SBT是只有G加了，SBT_plus是全加了
        return flow_res

    def encode(self, x, mask, t_list):
        z = self.encoder(x, mask, t_list)
        
        
        if self.normalize:
            z = F.normalize(z)
        return z

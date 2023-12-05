import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pointnet3 import PointNetAutoencoder
import sys
from event_utils import *
torch.set_printoptions(threshold=np.inf)
torch.set_printoptions(sci_mode=False)
def normalize_event_volume(event_volume):
        event_volume_flat = event_volume.view(-1)  # 展成一维
        nonzero = torch.nonzero(event_volume_flat)  # 找出非零索引
        nonzero_values = event_volume_flat[nonzero]  # 取出非零
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.1 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.9 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            max_val = max_val.item()
            event_volume = torch.clamp(event_volume, -max_val, max_val)
            event_volume /= max_val
        return event_volume

class EP2T(nn.Module):
    def __init__(self, pixel_size = (256, 256), x_offset = 12, y_offset = 12, t_offset = 0.1, pixel = (30, 30, 5), embedding_size=256):
        super(EP2T, self).__init__()
        self.pixel_size = pixel_size
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.t_offset = t_offset
        self.pixel = pixel
        self.flow_encoder = PointNetAutoencoder(embedding_size)
        
    def forward(self, flow):
        flow_ori = copy.deepcopy(flow)
        x = torch.linspace(self.x_offset, self.pixel_size[0] - self.x_offset , self.pixel[0])
        y = torch.linspace(self.y_offset, self.pixel_size[1] - self.y_offset, self.pixel[1])
        t_list = []
        for i in range(flow.shape[0]):
            tmp = torch.linspace(flow[i, :, 2].min(-1)[0] + self.t_offset, flow[i, :, 2].max(-1)[0] - self.t_offset, self.pixel[2])
            t_list.append(tmp)
        t_list = torch.stack(t_list, 0).cuda()
        # 归一化
        flow[:, :, :0] = flow[:, :, :0] / (self.pixel_size[0] - 1)
        flow[:, :, :1] = flow[:, :, :1] / (self.pixel_size[1] - 1)
        input_mean, input_max, input_min, input_std = torch.mean(flow[:,:,2], dim=1), torch.max(flow[:,:,2], dim=1)[0], torch.min(flow[:,:,2], dim=1)[0], torch.std(flow[:,:,2], dim=1)
        input_std = input_std.unsqueeze(1)
        input_max = input_max.unsqueeze(1)
        input_min = input_min.unsqueeze(1)
        flow[:, :, 2] = (flow[:, :, 2] - input_min) / (input_max - input_min)

        X, Y = torch.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        xy = torch.cat([X_flat.unsqueeze(1), Y_flat.unsqueeze(1)], 1)
        t = t_list.repeat_interleave(xy.shape[0]).view(flow.shape[0], -1, 1).cuda()
        xy = xy.unsqueeze(0).repeat(flow.shape[0], self.pixel[2], 1).cuda()
        center = torch.cat([xy, t], 2)
        flow_res = self.flow_encoder(flow, center, t_list)
        
        for i in range(flow.shape[0]):
            flow_input = flow_ori[i]
            flow_res_input = flow_res[i].transpose(-2,-1)
            flow_feature_output = (gen_feature_event(events = flow_input, events_val = flow_res_input, vol_size = [64, self.pixel_size[0], self.pixel_size[1]]))
            flow_volume_output = (gen_discretized_event_volume(events = flow_input, val_plus = True, events_val = flow_res_input, vol_size = [6, self.pixel_size[0], self.pixel_size[1]]))
            flow_time_output = (per_event_timing_images(flow_input, flow_res_input, [2, self.pixel_size[0], self.pixel_size[1]]))
            flow_stacking_output = (per_stacking_events(flow_input, flow_res_input, [3, self.pixel_size[0], self.pixel_size[1]]))
            flow_counting_output = (per_event_counting_images(flow_input, flow_res_input, [2, self.pixel_size[0], self.pixel_size[1]]))

            flow_volume_output = normalize_event_volume(flow_volume_output)
            flow_feature_output = normalize_event_volume(flow_feature_output)
            flow_time_output = normalize_event_volume(flow_time_output)
            flow_stacking_output = normalize_event_volume(flow_stacking_output)
            flow_counting_output = normalize_event_volume(flow_counting_output)

            flow_volume_output = flow_volume_output.permute(1,2,0).unsqueeze(0)
            flow_feature_output = flow_feature_output.permute(1,2,0).unsqueeze(0)
            flow_time_output = flow_time_output.permute(1,2,0).unsqueeze(0)
            flow_stacking_output = flow_stacking_output.permute(1,2,0).unsqueeze(0)
            flow_counting_output = flow_counting_output.permute(1,2,0).unsqueeze(0)
            if (i == 0):
                flow_time_res = flow_time_output
                flow_stacking_res = flow_stacking_output
                flow_counting_res = flow_counting_output
                flow_volume_res = flow_volume_output
                flow_feature_res = flow_feature_output
            else:
                flow_time_res = torch.cat([flow_time_res, flow_time_output], 0)
                flow_stacking_res = torch.cat([flow_stacking_res, flow_stacking_output], 0)
                flow_counting_res = torch.cat([flow_counting_res, flow_counting_output], 0)
                flow_volume_res = torch.cat([flow_volume_res, flow_volume_output], 0)
                flow_feature_res = torch.cat([flow_feature_res, flow_feature_output], 0)

        flow_res = torch.cat([flow_feature_res, flow_stacking_res, flow_volume_res, flow_counting_res, flow_time_res], -1)

        return flow_res


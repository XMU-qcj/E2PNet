import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import open3d
import torch
import argparse
import numpy as np


import sys

sys.path.append("/media/XH-8T/qcj/E2PNet/Dataset")
sys.path.append("../")
from data_loader_MVSEC_deepi2p import Data_MVSEC
from data_loader_VECtor import Data_VECtor
sys.path.append("../lcd")
from models.pointnet import *
from models.patchnet import *
from losses import *
sys.path.append("/media/XH-8T/qcj/E2PNet/lcd/lcd/models")
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--config", default = '/media/XH-8T/qcj/E2PNet/lcd/config_event.json', help="path to the log directory")
args = parser.parse_args()
config = args.config
config = json.load(open(config))

device = config["device"]


pointnet = PointNetAutoencoder_ori(config["embedding_size"],config["input_channels"],config["output_channels"],config["normalize"])
patchnet = PatchNetAutoencoder(config["embedding_size"], config["normalize"], config['input_channel'], config['output_channel'], config['feat_compute'])


if (config["load"]):
    fname = config["load"]
    pointnet.load_state_dict(torch.load(fname, map_location='cpu')["pointnet"], strict=False)
    patchnet.load_state_dict(torch.load(fname, map_location='cpu')["patchnet"], strict=False)
    
def normalize_point_cloud(pc):
    pc_input = copy.deepcopy(pc[:, :3])
    centroid = np.mean(pc_input, axis=0) # 求取点云的中心
    pc_input = pc_input - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc_input ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized = pc_input / m # 依据长轴将点云归一化到 (-1, 1)
    pc[:, :3] = pc_normalized
    return pc
pointnet.to(device)
pointnet.eval()
patchnet.to(device)
patchnet.eval()

testset3 = Data_VECtor("/media/XH-8T/EL_matching/",
                       "/media/XH-8T/EL_matching/VECtor_Benchmark/Large-scale", [0,], train=False,
                       stack_mode="SBN",pic_mode="DVS")
testloader = torch.utils.data.DataLoader(testset3, batch_size=4, shuffle=False,
                                             num_workers=8, pin_memory=True, drop_last = True)

import copy
mean_r = 0.
mean_t = 0.
import tqdm

criterion = {
    "mse": MSELoss(),
    "chamfer": ChamferLoss(config["output_channels"]),
    "triplet": HardTripletLoss(config["margin"], config["hardest"]),
}
criterion["mse"].to(device)
criterion["chamfer"].to(device)
criterion["triplet"].to(device)


with torch.no_grad():
    for i, data in tqdm.tqdm(enumerate(testloader), total = len(testloader)):
        img = torch.cat([data['event_stacking_images'].cuda().float(), data['event_volume'].cuda().float(), data['event_count_images'].cuda().float(), data['event_time_images'].cuda().float()], 1)
        pc = data['valid_points'].cpu().numpy()
        for pc_id in range(pc.shape[0]):
            pc[pc_id] = normalize_point_cloud(pc[pc_id])
        pc = torch.from_numpy(pc).cuda().float().transpose(2,1)
        flow = data['events_flow'].cuda().float()
        mid_img = data['mid_image'].cuda().float()
        mask = data['event_mask'].cuda().unsqueeze(1)
        P = data['cam_pose'].cuda().float()
        pixel_per_map = 31
        sampled_point = torch.zeros((flow.shape[0], pixel_per_map * pixel_per_map * 5, flow.shape[2])).to(flow.device)
        id_x = torch.ones((flow.shape[0], pixel_per_map * pixel_per_map)).to(flow.device)
        val = (torch.arange(8, 249, 8)).to(flow.device)
        x_val = (val.unsqueeze(-1).repeat(1, pixel_per_map)).reshape((-1))
        y_val = val.repeat(pixel_per_map).reshape((-1))
        mask = mask.reshape((mask.shape[0],-1))
        x_index = (torch.arange(8, 249, 8)).to(flow.device).repeat(pixel_per_map)
        y_index = (torch.arange(248 * 8, 248 * 249, 248 * 8)).to(flow.device).unsqueeze(-1).repeat(1, pixel_per_map).reshape((-1))
    
        mask_index = x_index + y_index
        mask_val = torch.index_select(mask, 1, mask_index)
        
        flow_pre = copy.deepcopy(flow)
        flow_ori = copy.deepcopy(flow)
        flow_ori_t, _ = flow_ori[:,:,2].sort(-1)

        flow[:, :, :2] = flow[:, :, :2] / 255.
        input_mean, input_max, input_min, input_std = torch.mean(flow[:,:,2], dim=1), torch.max(flow[:,:,2], dim=1)[0], torch.min(flow[:,:,2], dim=1)[0], torch.std(flow[:,:,2], dim=1)
        input_mean = input_mean.unsqueeze(1)
        input_std = input_std.unsqueeze(1)
        input_max = input_max.unsqueeze(1)
        input_min = input_min.unsqueeze(1)
        flow[:, :, 2] = (flow[:, :, 2] - input_min) / (input_max - input_min)
        flow_t, _ = flow[:,:,2].sort(-1)
        for id in range(5):
            id_t = (flow_t[:, id * (flow.shape[1] // 5)]  + ((flow_t[:, -1] - flow_t[:, 0]) / 10)).unsqueeze(-1)
            id_ori_t = (flow_ori_t[:, id * (flow_ori.shape[1] // 5)]  + ((flow_ori_t[:, -1] - flow_ori_t[:, 0]) / 10)).unsqueeze(-1)

            if (id == 0):
                t_list = flow_t[:, 0].unsqueeze(-1)
                t_list = torch.cat([t_list, id_t], -1)
                t_ori_list = id_ori_t
            else:
                t_list = torch.cat([t_list, id_t], -1)
                t_ori_list = torch.cat([t_ori_list, id_ori_t], -1)
            id_t = id_t.repeat(1, pixel_per_map * pixel_per_map)
            sampled_point[:,(id * (pixel_per_map * pixel_per_map)):((id+1) * (pixel_per_map * pixel_per_map)), 2] = id_t
            sampled_point[:,(id * (pixel_per_map * pixel_per_map)):((id+1) * (pixel_per_map * pixel_per_map)), 0] = x_val / 255.
            sampled_point[:,(id * (pixel_per_map * pixel_per_map)):((id+1) * (pixel_per_map * pixel_per_map)), 1] = y_val / 255.
            sampled_point[:,(id * (pixel_per_map * pixel_per_map)):((id+1) * (pixel_per_map * pixel_per_map)), 3] = mask_val
        t_list = torch.cat([t_list, flow_t[:, -1].unsqueeze(-1)], -1)

        input_event = img.permute(0,2,3,1)
        input_pc = pc.permute(0,2,1)
        output_event = mid_img
        # output_event = data['event_mask'].unsqueeze(1).to(device).float().permute(0,2,3,1)
        
        
        output_pc = input_pc[:,:,:3]
        mid_img = mid_img

        x = [input_pc, input_event]
        _, z0 = pointnet(x[0])
        y1, z1 = patchnet(x[1], flow, flow_pre, mid_img, sampled_point, t_list)

        loss_d = criterion["triplet"](z0, z1)

        if (i == 0):
            P_list = P.cpu()
            pc_list = z0.cpu()
            img_list = z1.cpu()
        else:
            P_list = torch.cat([P_list, P.cpu()], 0)
            pc_list = torch.cat([pc_list, z0.cpu()], 0)
            img_list = torch.cat([img_list, z1.cpu()], 0)
        # if (i == 3):
        #     break

import math
def get_P_diff(P_pred_np, P_gt_np):
    R_pred = P_pred_np[:,:3, :3]
    t_pred = P_pred_np[:,:3, 3]
    R_gt = P_gt_np[:,:3, :3]
    t_gt = P_gt_np[:,:3, 3]
    t_diff = translation_error(t_pred, t_gt)
    r_diff = rotation_error(R_pred, R_gt)

    return t_diff, r_diff
def rotation_error(R, R_gt):
	cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
	cos_theta = torch.clamp(cos_theta, -1, 1)
	return torch.acos(cos_theta) * 180 / math.pi
def translation_error(t, t_gt):
    return torch.norm(t - t_gt, dim = 1)

loss = HardTripletLoss(config["margin"], config["hardest"])
k = 20


def _pairwise_distance_squared(x, y):
    xx = torch.sum(torch.pow(x, 2), 1).view(-1, 1)
    yy = torch.sum(torch.pow(y, 2), 1).view(1, -1)
    pdist = xx + yy - 2.0 * torch.mm(x, torch.t(y))
    return pdist


# for i in tqdm.tqdm(range(img_list.shape[0])):
i = 580
pc_feature = pc_list[i].unsqueeze(0).repeat(pc_list.shape[0], 1)
dist = _pairwise_distance_squared(pc_feature, img_list)
dist = dist[0]
loss_d = dist.topk(k,  largest=False)[1]
now_P = P_list[loss_d, ...]
target_P = P_list[i].unsqueeze(0).repeat(k, 1, 1)
t_diff, r_diff = get_P_diff(now_P, target_P)
# all_diff = t_diff + r_diff
# index = all_diff.min(0)[1]
# r_min = r_diff[index]
# t_min = t_diff[index]
r_min = r_diff.topk(k,  largest=False)[1] #r_diff.min(0)[1]
# print(r_min) 
t_min = t_diff.topk(k,  largest=False)[1]
mean_r += r_min
mean_t += t_min
print(now_P[r_min])
print(now_P[t_min])


i = 581
pc_feature = pc_list[i].unsqueeze(0).repeat(pc_list.shape[0], 1)
dist = _pairwise_distance_squared(pc_feature, img_list)
dist = dist[0]
loss_d = dist.topk(k,  largest=False)[1]
now_P = P_list[loss_d, ...]
target_P = P_list[i].unsqueeze(0).repeat(k, 1, 1)
t_diff, r_diff = get_P_diff(now_P, target_P)
# all_diff = t_diff + r_diff
# index = all_diff.min(0)[1]
# r_min = r_diff[index]
# t_min = t_diff[index]
r_min = r_diff.topk(k,  largest=False)[1] #r_diff.min(0)[1]
# print(r_min) 
t_min = t_diff.topk(k,  largest=False)[1]
mean_r += r_min
mean_t += t_min
print(now_P[r_min])
print(now_P[t_min])
print(dsads)
mean_r = mean_r / img_list.shape[0]
mean_t = mean_t / img_list.shape[0]
print(mean_r, mean_t)

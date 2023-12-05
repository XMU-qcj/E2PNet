import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import open3d
import torch
import argparse
import numpy as np


import sys

sys.path.append("/media/XH-8T/qcj/E2PNet/Dataset")
sys.path.append("../")
from data_loader_MVSEC import Data_MVSEC
from data_loader_VECtor import Data_VECtor
sys.path.append("../lcd")
from models.pointnet import *
from models.patchnet import *
from losses import *
sys.path.append("/media/XH-8T/qcj/E2PNet/EP2T")
from utils import *
from EP2T import EP2T





parser = argparse.ArgumentParser()
parser.add_argument("--config", default = '/media/XH-8T/qcj/E2PNet/lcd/apps/config_event.json', help="path to the log directory")
args = parser.parse_args()
config = args.config
config = json.load(open(config))

device = config["device"]

pointnet = PointNetAutoencoder_ori(config["embedding_size"],config["input_channels"],config["output_channels"],config["normalize"])

patchnet = PatchNetAutoencoder(config["embedding_size"], config["normalize"], config['input_channel'], config['output_channel'], config['feat_compute'])
from torch.nn.parallel import DataParallel
import copy
patchnet = DataParallel(patchnet).cuda()
pointnet = DataParallel(pointnet).cuda()
EP2T_model = DataParallel(EP2T()).cuda()
if (config["load"]):
    fname = config["load"]
    pointnet.load_state_dict(torch.load(fname)["pointnet"])
    patchnet.load_state_dict(torch.load(fname)["patchnet"])

def normalize_point_cloud(pc):
    centroid = np.mean(pc, axis=0) # 求取点云的中心
    pc = pc - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized = pc / m # 依据长轴将点云归一化到 (-1, 1)
    return pc_normalized
pointnet.to(device)
pointnet.eval()
patchnet.to(device)
patchnet.eval()

testset2 = Data_MVSEC("/media/XH-8T/EL_matching/", [3,], train=False, stack_mode="SBN")
testloader = torch.utils.data.DataLoader(testset2, batch_size=1, shuffle=False,
                                             num_workers=2, pin_memory=True, drop_last = True)

mean_r = 0.
mean_t = 0.
import tqdm
with torch.no_grad():
    for i, data in tqdm.tqdm(enumerate(testloader), total = len(testloader)):
        img = torch.cat([data['event_stacking_images'].cuda().float(), data['event_volume'].cuda().float(), data['event_count_images'].cuda().float(), data['event_time_images'].cuda().float()], 1)
        pc = data['valid_points'].cpu().numpy()
        for pc_id in range(pc.shape[0]):
            pc[pc_id] = normalize_point_cloud(pc[pc_id])
        pc = torch.from_numpy(pc).cuda().float().transpose(2,1)

        flow = data['events_flow'].cuda().float()
        flow_pre = copy.deepcopy(flow)
        mid_img = data['mid_image'].cuda().float()
        mask = data['event_mask'].cuda().unsqueeze(1)
        P = data['cam_pose'].cuda().float()
        
        EP = EP2T_model(flow)
        input_event = EP
        mid_img = mid_img
        input_pc = pc.permute(0,2,1)
        output_pc = input_pc[:,:,:3]
        
        output_event = data['event_mask'].unsqueeze(1).to(device).float().permute(0,2,3,1)

        x = [input_pc, input_event]
        y0, z0 = pointnet(x[0])
        y1, z1 = patchnet(x[1])

        if (i == 0):
            P_list = P.cpu()
            pc_list = z0.cpu()
            img_list = z1.cpu()
        else:
            P_list = torch.cat([P_list, P.cpu()], 0)
            pc_list = torch.cat([pc_list, z0.cpu()], 0)
            img_list = torch.cat([img_list, z1.cpu()], 0)

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
k = 10

def _pairwise_distance_squared(x, y):
    xx = torch.sum(torch.pow(x, 2), 1).view(-1, 1)
    yy = torch.sum(torch.pow(y, 2), 1).view(1, -1)
    pdist = xx + yy - 2.0 * torch.mm(x, torch.t(y))
    return pdist

for i in tqdm.tqdm(range(img_list.shape[0])):
    pc_feature = pc_list[i].unsqueeze(0).repeat(pc_list.shape[0], 1)
    dist = _pairwise_distance_squared(pc_feature, img_list)
    dist = dist[0]
    loss_d = dist.topk(k,  largest=False)[1]
    now_P = P_list[loss_d, ...]
    target_P = P_list[i].unsqueeze(0).repeat(k, 1, 1)
    t_diff, r_diff = get_P_diff(now_P, target_P)

    r_min = r_diff.min(0)[0]
    t_min = t_diff.min(0)[0]
    mean_r += r_min
    mean_t += t_min
mean_r = mean_r / img_list.shape[0]
mean_t = mean_t / img_list.shape[0]
print(mean_r, mean_t)

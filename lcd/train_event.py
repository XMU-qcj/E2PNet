import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import json
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from collections import defaultdict
import tqdm
from lcd.dataset import CrossTripletDataset
from lcd.models.pointnet import *
from lcd.models.patchnet import *
from lcd.losses import *
import sys
import copy
sys.path.append("/media/XH-8T/qcj/E2PNet/Dataset")
from data_loader_MVSEC import Data_MVSEC
sys.path.append("/media/XH-8T/qcj/E2PNet/EP2T")
from EP2T import EP2T
def file_work(LOG_DIR):
    import glob
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    # backup some files
    py_file = glob.glob('/media/XH-8T/qcj/E2PNET/lcd/lcd/models/*.py')
    log_code = os.path.join(LOG_DIR, 'code')
    os.makedirs(log_code)
    for f in py_file:
        os.system('cp %s %s' % (f, log_code))

parser = argparse.ArgumentParser()
parser.add_argument("--config", default = 'config_event.json',help="path to the json config file")
parser.add_argument("--logdir", default = 'logs/LCD', help="path to the log directory")
parser.add_argument("--checkdir", default = 'lcd/checkpoint/', help="path to the log directory")
parser.add_argument("--root", help="path to the log directory")
args = parser.parse_args()

config = args.config
event_name = '_ep2t_vector'

logdir = args.logdir+event_name
checkpoint = args.checkdir+event_name[1:]
args = json.load(open(config))
import shutil
if os.path.isdir(logdir):
    shutil.rmtree(logdir)
file_work(logdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

fname = os.path.join(logdir, "config_"+event_name+".json")
with open(fname, "w") as fp:
    json.dump(args, fp, indent=4)

device = args["device"]
dataset2 = Data_MVSEC("/media/XH-8T/EL_matching/", [0,1,2], train=True, stack_mode="SBN")
loader = data.DataLoader(
    dataset2,
    batch_size=args["batch_size"],
    num_workers=args["num_workers"],
    pin_memory=True,
    shuffle=True,
    drop_last = True,
)

from torch.utils.data import DataLoader

testset2 = Data_MVSEC("/media/XH-8T/EL_matching/", [3,], train=False, stack_mode="SBN")
testloader = torch.utils.data.DataLoader(testset2, batch_size=args["batch_size"], shuffle=False,
                                             num_workers=args["num_workers"], pin_memory=True, drop_last = True)


patchnet = PatchNetAutoencoder(args["embedding_size"], args["normalize"], args['input_channel'], args['output_channel'], args['feat_compute'])
pointnet = PointNetAutoencoder_ori(
    args["embedding_size"],
    args["input_channels"],
    args["output_channels"],
    args["normalize"],
)
if (args['load']):
    patchnet.load_state_dict(torch.load(args['load'])["patchnet"])
    pointnet.load_state_dict(torch.load(args['load'])["pointnet"])
from torch.nn.parallel import DataParallel
patchnet = DataParallel(patchnet).cuda()
pointnet = DataParallel(pointnet).cuda()
EP2T_model = DataParallel(EP2T()).cuda()


parameters = list(patchnet.parameters()) + list(pointnet.parameters())
optimizer = optim.SGD(
    parameters,
    lr=args["learning_rate"],
    momentum=args["momentum"],
    weight_decay=args["weight_decay"],
)

criterion = {
    "mse": MSELoss(),
    "chamfer": ChamferLoss(args["output_channels"]),
    "triplet": HardTripletLoss(args["margin"], args["hardest"]),
}
criterion["mse"] = criterion["mse"].cuda()
criterion["chamfer"] = criterion["chamfer"].cuda()
criterion["triplet"] = criterion["triplet"].cuda()

import copy
def normalize_point_cloud(pc):
    pc_input = copy.deepcopy(pc[:, :3])
    centroid = np.mean(pc_input, axis=0) # 求取点云的中心
    pc_input = pc_input - centroid # 将点云中心置于原点 (0, 0, 0)
    m = np.max(np.sqrt(np.sum(pc_input ** 2, axis=1))) # 求取长轴的的长度
    pc_normalized = pc_input / m # 依据长轴将点云归一化到 (-1, 1)
    pc[:, :3] = pc_normalized
    return pc

best_loss = np.Inf
for epoch in range(args["epochs"]):
    start = datetime.datetime.now()
    scalars = defaultdict(list)
    pointnet.train()
    patchnet.train()
    for i, data in tqdm.tqdm(enumerate(loader), total = len(loader)):
        # continue
        img = torch.cat([data['event_stacking_images'].cuda().float(), data['event_volume'].cuda().float(), data['event_count_images'].cuda().float(), data['event_time_images'].cuda().float()], 1)
        pc = data['valid_points'].cpu().numpy()
        for pc_id in range(pc.shape[0]):
            pc[pc_id] = normalize_point_cloud(pc[pc_id])
        pc = torch.from_numpy(pc).cuda().float().transpose(2,1)

        flow = data['events_flow'].cuda().float()
        flow_pre = copy.deepcopy(flow)
        mid_img = data['mid_image'].cuda().float()
        mask = data['event_mask'].cuda().unsqueeze(1)

        
        EP = EP2T_model(flow)
        input_event = EP
        mid_img = mid_img
        input_pc = pc.permute(0,2,1)
        output_pc = input_pc[:,:,:3]
        
        output_event = data['event_mask'].unsqueeze(1).to(device).float().permute(0,2,3,1)

        x = [input_pc, input_event]
        y0, z0 = pointnet(x[0])
        y1, z1 = patchnet(x[1])
        loss_r = 0
        loss_d = 0

        loss_r += args["alpha"] * criterion["mse"](output_event, y1)
        loss_r += args["beta"] * criterion["chamfer"](output_pc, y0)
        loss_d += args["gamma"] * criterion["triplet"](z0, z1)
        loss = loss_d + loss_r

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(pointnet.parameters(), max_norm=1, norm_type=2)
        torch.nn.utils.clip_grad_norm_(patchnet.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        scalars["loss"].append(loss)
        scalars["loss_d"].append(loss_d)
        scalars["loss_r"].append(loss_r)

        now = datetime.datetime.now()
        log = "{} | Batch [{:04d}/{:04d}] | loss: {:.4f} | loss_d: {:.4f} | loss_r: {:.4f} |"
        log = log.format(now.strftime("%c"), i, len(loader), loss.item(), loss_d.item(), loss_r.item())
        if (i % 200 == 0):
            print(log)
        # break
    # Summary after each epoch
    
    if (epoch % 5 == 0):
        pointnet.eval()
        patchnet.eval()
        mean_loss_d = 0.
        mean_loss_r = 0.
        mean_loss = 0.
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

                EP = EP2T_model(flow)

                input_pc = pc.permute(0,2,1)
                output_event = data['event_mask'].unsqueeze(1).to(device).float().permute(0,2,3,1)
                input_event = EP

                output_pc = input_pc[:,:,:3]
                mid_img = mid_img

                x = [input_pc, input_event]

                y0, z0 = pointnet(x[0])
                y1, z1 = patchnet(x[1])

                loss_r = 0
                loss_d = 0


                loss_r += args["alpha"] * criterion["mse"](output_event, y1)
                loss_r += args["beta"] * criterion["chamfer"](output_pc, y0)
                loss_d += args["gamma"] * criterion["triplet"](z0, z1)
                loss = loss_d + loss_r


                scalars["loss"].append(loss)
                scalars["loss_d"].append(loss_d)
                scalars["loss_r"].append(loss_r)


                
                mean_loss_r += loss_r.item()
                mean_loss_d += loss_d.item()
                mean_loss += loss.item()
                now = datetime.datetime.now()
                
                # break
            print("--------------------------------------------------------------------------")
            mean_loss_r_res = mean_loss_r / len(testloader)
            mean_loss_d_res = mean_loss_d / len(testloader)
            mean_loss_res = mean_loss / len(testloader)
            log = "Test:{} | Batch [{:04d}/{:04d}] | loss: {:.4f} | loss_d: {:.4f} | loss_r: {:.4f} |"
            log = log.format(now.strftime("%c"), i, len(testloader), mean_loss_res, mean_loss_d_res, mean_loss_r_res)

            print(log)
            print("--------------------------------------------------------------------------")
            with open(fname, "a") as fp:
                fp.write(log + "\n")
        summary = {}
        now = datetime.datetime.now()
        duration = (now - start).total_seconds()
        log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
        log = log.format(now.strftime("%c"), epoch, args["epochs"], duration)
        for m, v in scalars.items():
            summary[m] = torch.stack(v).mean()
            log += " {}: {:.4f} |".format(m, summary[m].item())


        if summary["loss"] < best_loss:
            best_loss = summary["loss"]
            fname = os.path.join(checkpoint, "model_"+event_name+".pth")
            dir_name = checkpoint
            print("> Saving model to {}...".format(fname))
            model = {"pointnet": pointnet.state_dict(), "patchnet": patchnet.state_dict()}
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)
            torch.save(model, fname)
        log += " best: {:.4f} |".format(best_loss)

        fname = os.path.join(logdir, "train_"+event_name+".log")
        with open(fname, "a") as fp:
            fp.write(log + "\n")

        print(log)
        print("--------------------------------------------------------------------------")
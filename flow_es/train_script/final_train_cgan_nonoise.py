#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_cgan.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/6 15:39
# @Desc  :
import sys
import os
import cv2

sys.path.append(os.path.dirname(os.getcwd()))
import torch
from torch.optim import Adam
from networks.cGAN3 import Generator, Discriminator
# from networks.cGAN_patch import Generator, Discriminator
from Loss.loss_function import L1loss, BCEloss
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import argparse
import numpy as np
from Utils.result_visualization import flow2image
from Data.data_loader_SBT_SBN import MyData
from torch.utils.data import DataLoader
import random
import torch.nn as nn
from dataloader.h5 import H5Loader
from Loss.cons import *
sys.path.append("/media/XH-8T/qcj/E2PNet/EP2T")
from EP2T import EP2T

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(8886)
params = argparse.ArgumentParser()
params.add_argument("--ic", type=int, default=3 + 77, help='in channel')
params.add_argument("--oc", type=int, default=1, help='out channel')
params.add_argument("--epoch", type=int, default=160, help='training epochs')
params.add_argument("--batch", type=int, default=8, help='batch size')
params.add_argument('-gpu', type=str, default='3,2', help='device index')
params.add_argument("--log", type=str, default='./construct', help='log path')
args = params.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
plus = True
zoom_factor=1

G = Generator(args.ic , args.oc)  # 通道数要改
G = nn.DataParallel(G)
D = Discriminator(args.oc + 3 + 77)
D = nn.DataParallel(D)
EP2T_model = EP2T()
EP2T_model = nn.DataParallel(EP2T_model)

trainer_G = Adam(G.parameters(), 0.0002*3, betas=(0.5, 0.999), weight_decay=0.0005)
trainer_D = Adam(D.parameters(), 0.0002*3, betas=(0.5, 0.999), weight_decay=0.0005)
trainer_EP2T = Adam(EP2T_model.parameters(), 0.0002*3, betas=(0.5, 0.999), weight_decay=0.0005)

MSELoss=torch.nn.MSELoss()


train_data = MyData(7, train=True, engine_mask=True, stack_mode="SBT") 
train_loader = DataLoader(train_data, args.batch, drop_last=True, shuffle=True, num_workers=16)
print("train_data长度=",train_data.__len__())
train_data_frame_num=train_data.num_images

val_data = MyData(6, train=False, engine_mask=True, stack_mode="SBT")
val_loader = DataLoader(val_data, args.batch, drop_last=True, shuffle=True, num_workers=16)
print("val_data长度=",val_data.__len__())
val_data_frame_num=val_data.num_images
assert train_data.stack_mode == val_data.stack_mode
loss_judge = compute_infer_loss()

import tqdm
import copy
if __name__ == '__main__':
    summary = SummaryWriter(args.log)
    for epoch in range(args.epoch):
        D_Losses = []
        G_Losses = []
        EVAL = []

        d_fake =[]
        d_real = []
        d_shuffle =[]
        d_G= []

        MSE = [[], []]
        LPIPS = [[], []]
        SSIM = [[], []]
        log_gt = None
        log_pre = None
        for i, batch in tqdm.tqdm(enumerate(train_loader), total = 501):  # 训练循环
            # event = torch.randn(args.batch, args.ic, 256, 256)  # 替换
            # event = batch['event_volume'].float()
            event_stacking_images = batch['event_stacking_images'].float()
            gradient = batch['image_gradient'].float()
            mid_image = batch['mid_image'].float()
            # noise = torch.randn(event_stacking_images.size())
            event_volume=batch['event_volume'].float()
            event_time_images=batch['event_time_images'].float()
            event_mask = batch['event_mask'].long()
            event_count = batch['event_count_images'].float()
            flow_mask = batch['flow_mask'].long()
            full_mask = torch.mul(event_mask, flow_mask)
            Effective_Pixels_num = torch.sum(full_mask)
        
            time_scaling=batch['time_scaling'].float()
            time_scaling = time_scaling.reshape(-1, 1, 1)
            time_scaling = time_scaling.expand(time_scaling.shape[0], 256, 256)

            Full_Mask = torch.stack((full_mask, full_mask), dim=1)
            Flow_Mask = torch.stack((flow_mask, flow_mask), dim=1)
            Event_mask = torch.stack((event_mask, event_mask), dim=1)

            # event = torch.cat([event_stacking_images, gradient], 1)#5通道
            event = event_stacking_images#3通道
            event_img = torch.cat([event_stacking_images, event_volume, event_count, event_time_images], 1)
            # event = torch.cat([event_stacking_images, event_time_images], 1)#3+2通道
            # event = torch.cat([event_stacking_images, mid_image], 1)#4通道
            # event=event_volume#
            # 通道数要改
            noise = torch.randn(event.size())
            # real = torch.randn(args.batch, args.oc, 256, 256)  # 替换
            # real = batch['flow_frame'].float()

            real = mid_image.unsqueeze(1).float()
            real_label = torch.ones(args.batch, 1, 31, 31)
            fake_label = torch.zeros(args.batch, 1, 31, 31)

            raw_index = list(range(args.batch))
            shuffle_index = list(range(args.batch))
            np.random.shuffle(shuffle_index)
            flag = (np.array(shuffle_index) == np.array(raw_index))
            flag = torch.tensor(flag).float().view(-1, 1, 1, 1)
            shuffle_label = real_label * flag

            flow = batch['events_flow'].float()

            if torch.cuda.is_available():
                G = G.cuda()
                D = D.cuda()
                EP2T_model = EP2T_model.cuda()
                event = event.cuda()
                noise = noise.cuda()
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()
                shuffle_label = shuffle_label.cuda()
                real = real.cuda()
                Flow_Mask=Flow_Mask.cuda()
                Event_mask=Event_mask.cuda()
                Full_Mask = Full_Mask.cuda()
                time_scaling=time_scaling.cuda()
                flow = flow.cuda()
                event_img = event_img.cuda()
            # train D
            
            G.eval()
            D.train()
            
            trainer_D.zero_grad()
            
            for param in G.parameters():
                param.requires_grad = False

            for param in D.parameters():
                param.requires_grad = True
            
            if (plus == True):
                EP2T_model.train()
                trainer_EP2T.zero_grad()
                for param in EP2T_model.parameters():
                    param.requires_grad = True
            event_ori = copy.deepcopy(event)
            if (plus == True):
                EP = EP2T_model(flow)
                EP = EP.permute(0, 3, 1, 2)
                event = torch.cat([event_ori, EP], 1)
            else:
                event = event_ori
            fake_image = G(event)
            fake_image=zoom_factor*fake_image
            fake_pre = D(torch.cat([fake_image.detach(), event], 1))#Full_Mask
            # fake_pre = D(torch.cat([torch.mul(fake_image.detach(), Event_mask), event], 1))
            # fake_pre = D(torch.cat([torch.mul(fake_image.detach(), Flow_Mask), event], 1))
            # fake_pre = D(torch.cat([torch.mul(fake_image.detach(), Full_Mask), event], 1))#最好结果
            real_pre = D(torch.cat([real.detach(), event], 1))
            # real_pre = D(torch.cat([torch.mul(real.detach(), Event_mask), event], 1))
            # real_pre = D(torch.cat([torch.mul(real.detach(), Flow_Mask), event], 1))
            # real_pre = D(torch.cat([torch.mul(real.detach(), Full_Mask), event], 1))#最好结果
            shuffle_pre = D(torch.cat([real[shuffle_index, ...].detach(), event], 1))
            # shuffle_pre = D(torch.cat([torch.mul(real[shuffle_index, ...].detach(),Event_mask), event], 1))
            # shuffle_pre = D(torch.cat([torch.mul(real[shuffle_index, ...].detach(),Flow_Mask), event], 1))
            # shuffle_pre = D(torch.cat([torch.mul(real[shuffle_index, ...].detach(),Full_Mask), event], 1))#最好结果

            loss_d = (BCEloss(fake_pre, fake_label) + BCEloss(real_pre, real_label)
                      + BCEloss(shuffle_pre, shuffle_label)) / 3.0
            # loss_d = (BCEloss(fake_pre, fake_label) + BCEloss(real_pre, real_label)) / 2.0

            d_fake.append(float(BCEloss(fake_pre, fake_label)))
            d_real.append(float(BCEloss(real_pre, real_label)))
            d_shuffle.append(float(BCEloss(shuffle_pre, shuffle_label)))
                     
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=10, norm_type=2)
            
            trainer_D.step()
            if (plus == True):
                trainer_EP2T.step()
                torch.nn.utils.clip_grad_norm_(EP2T_model.parameters(), max_norm=10, norm_type=2)


            # train G
            D.eval()
            G.train()
            
            trainer_G.zero_grad()

            for param in D.parameters():
                param.requires_grad = False

            for param in G.parameters():
                param.requires_grad = True
            if (plus == True):
                EP2T_model.train()
                trainer_EP2T.zero_grad()
                for param in EP2T_model.parameters():
                    param.requires_grad = True
            
            if (plus == True):
                EP = EP2T_model(flow)
                EP = EP.permute(0, 3, 1, 2)
                event = torch.cat([event_ori, EP], 1)
            else:
                event = event_ori
            fake_image = G(event)
            fake_image=zoom_factor*fake_image

            fake_pre = D(torch.cat([fake_image.detach(), event], 1))
            # fake_pre = D(torch.cat([torch.mul(fake_image.detach(),Event_mask), event], 1))
            # fake_pre = D(torch.cat([torch.mul(fake_image.detach(),Flow_Mask), event], 1))
            # fake_pre = D(torch.cat([torch.mul(fake_image.detach(),Full_Mask), event], 1))#最好结果

            # eval = L1loss(fake_image, real)
            eval=MSELoss(fake_image, real)
            # eval = L1loss(torch.mul(fake_image, Event_mask), torch.mul(real, Event_mask))
            # eval = L1loss(torch.mul(fake_image, Flow_Mask), torch.mul(real, Flow_Mask))#最好结果
            # eval = L1loss(torch.mul(fake_image, Full_Mask), torch.mul(real, Full_Mask))

            # eval = L1loss(fake_image[Event_mask], real[Event_mask])
            loss_g = (1*eval + BCEloss(fake_pre, real_label)) / 2.0
            

            d_G.append(float(BCEloss(fake_pre, real_label)))

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=10, norm_type=2)
            
            trainer_G.step()
            if (plus == True):
                torch.nn.utils.clip_grad_norm_(EP2T_model.parameters(), max_norm=10, norm_type=2)
                trainer_EP2T.step()

            # print('epoch:{},Iteration:{},d_loss:{},g_loss:{},eval:{}'.format(epoch, i, float(loss_d), float(loss_g),
            #                                                                  float(eval)))
            if (i % 200 == 0):
                print("train epoh:",epoch,"+Interation:",i," d_loss=",'%.3f' %float(loss_d)," g_loss=",'%.3f' %float(loss_g),
                    " eval=",'%.3f' %float(eval))
                print( "predicted max：", '%.3f' % torch.max(fake_image), "  min:", '%.3f' % torch.min(fake_image))
                print("flow_frame  max:", '%.3f' % torch.max(real), "  min:", '%.3f' % torch.min(real))

            D_Losses.append(float(loss_d))
            G_Losses.append(float(loss_g))
            EVAL.append(float(eval))



            cons_loss = loss_judge.write_loss(fake_image.detach(), real.detach())

            lpips = cons_loss['mean_lpips']
            mse = cons_loss['mean_mse']
            ssim = cons_loss['mean_ssim']

            MSE[0].append(float(mse))
            SSIM[0].append(float(ssim))
            LPIPS[0].append(float(lpips))
            if (i % 200 == 0):
                print("train epoh:", epoch, " MSE:", '%.3f' % mse, "  lpips:", '%.3f' % lpips, "  ssim:", '%.3f' % ssim)
                
            if i>train_data_frame_num or i >500 or i == 2:
                log_gt = real
                log_pre = fake_image
                log_gt = make_grid(log_gt)
                log_pre = make_grid(log_pre)
                # break
        # write_log
        summary.add_scalar('d_loss', np.mean(D_Losses), epoch)
        summary.add_scalar('g_loss', np.mean(G_Losses), epoch)
        summary.add_scalar('L1', np.mean(EVAL), epoch)

        summary.add_scalar('d_fake', np.mean(d_fake), epoch)
        summary.add_scalar('d_real', np.mean(d_real), epoch)
        summary.add_scalar('d_shuffle', np.mean(d_shuffle), epoch)
        summary.add_scalar('d_G', np.mean(d_G), epoch)

        summary.add_image('train_gt', log_gt, epoch)
        summary.add_image('train_pre', log_pre, epoch)


        """下面是测试集，"""
        for i, batch in tqdm.tqdm(enumerate(val_loader), total = 501):  # 内部循环
            val_D_Losses = []
            val_G_Losses = []
            val_EVAL = []
            # event = torch.randn(args.batch, args.ic, 256, 256)  # 替换
            # event = batch['event_volume'].float()
            event_stacking_images = batch['event_stacking_images'].float()
            gradient = batch['image_gradient'].float()
            mid_image = batch['mid_image'].float()
            # noise = torch.randn(event_stacking_images.size())

            event_mask = batch['event_mask'].long()
            flow_mask = batch['flow_mask'].long()
            full_mask = torch.mul(event_mask, flow_mask)
            Effective_Pixels_num = torch.sum(full_mask)
            event_volume=batch['event_volume'].float()
            event_time_images=batch['event_time_images'].float()

            time_scaling = batch['time_scaling'].float()
            time_scaling = time_scaling.reshape(-1, 1, 1)
            time_scaling = time_scaling.expand(time_scaling.shape[0], 256, 256)

            Full_Mask = torch.stack((full_mask, full_mask), dim=1)
            Flow_Mask = torch.stack((flow_mask, flow_mask), dim=1)
            Event_mask = torch.stack((event_mask, event_mask), dim=1)

            # event = torch.cat([event_stacking_images, gradient], 1)#5通道
            event = event_stacking_images  # 3通道
            event_img = torch.cat([event_stacking_images, event_volume, event_count, event_time_images], 1)
            # event = torch.cat([event_stacking_images, event_time_images], 1)#3+2通道
            # event = torch.cat([event_stacking_images, mid_image], 1)#4通道
            # event=event_volume#
            noise = torch.randn(event.size())
            # 通道数要改
            # real = torch.randn(args.batch, args.oc, 256, 256)  # 替换
            # real = batch['flow_frame'].float()
            real = mid_image.unsqueeze(1).float()
            real_label = torch.ones(args.batch, 1, 31, 31)
            fake_label = torch.zeros(args.batch, 1, 31, 31)

            raw_index = list(range(args.batch))
            shuffle_index = list(range(args.batch))
            np.random.shuffle(shuffle_index)
            flag = (np.array(shuffle_index) == np.array(raw_index))
            flag = torch.tensor(flag).float().view(-1, 1, 1, 1)
            shuffle_label = real_label * flag

            flow = batch['events_flow'].float()
            if torch.cuda.is_available():
                G = G.cuda()
                D = D.cuda()
                event = event.cuda()
                event_img = event_img.cuda()
                noise = noise.cuda()
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()
                shuffle_label = shuffle_label.cuda()
                real = real.cuda()
                Flow_Mask=Flow_Mask.cuda()
                Event_mask=Event_mask.cuda()
                Full_Mask = Full_Mask.cuda()
                time_scaling=time_scaling.cuda()
                flow = flow.cuda()

            # 验证集，仅测试
            G.eval()
            D.eval()
            if (plus == True):
                EP2T_model.eval()
                for param in EP2T_model.parameters():
                    param.requires_grad = False
            for param in G.parameters():
                param.requires_grad = False

            for param in D.parameters():
                param.requires_grad = False
            
            
            event_ori = copy.deepcopy(event)
            
            if (plus == True):
                EP = EP2T_model(flow)
                EP = EP.permute(0, 3, 1, 2)
                event = torch.cat([event_ori, EP], 1)
            else:
                event = event_ori
            fake_image = G(event)
            fake_image=zoom_factor*fake_image
            
            val_D_Losses.append(float(loss_d))
            val_G_Losses.append(float(loss_g))
            val_EVAL.append(float(eval))

            cons_loss = loss_judge.write_loss(fake_image.detach(), real.detach())

            lpips = cons_loss['mean_lpips']
            mse = cons_loss['mean_mse']
            ssim = cons_loss['mean_ssim']

            MSE[1].append(float(mse))
            SSIM[1].append(float(ssim))
            LPIPS[1].append(float(lpips))
            if (i % 200 == 0):
                print("val epoh:", epoch, " MSE:", '%.3f' % mse, "  lpips:", '%.3f' % lpips, "  ssim:", '%.3f' % ssim)

                
            if i>val_data_frame_num or i >500 or i == 2:
                log_gt = real
                log_pre = fake_image

                # masked_gt = make_grid(torch.mul(log_gt, Full_Mask))
                # masked_pre = make_grid(torch.mul(log_pre, Full_Mask))
                log_gt = make_grid(log_gt)
                log_pre = make_grid(log_pre)
                # break

        # write_log
        summary.add_scalar('val_d_loss', np.mean(val_D_Losses), epoch)
        summary.add_scalar('val_g_loss', np.mean(val_G_Losses), epoch)
        summary.add_scalar('val_L1', np.mean(val_EVAL), epoch)
        summary.add_image('val_gt', log_gt, epoch)
        summary.add_image('val_pre', log_pre, epoch)
        # summary.add_image('val_masked_gt', masked_gt, epoch)
        # summary.add_image('val_masked_pre', masked_pre, epoch)

        summary.add_scalars('MSE', {
            'train': np.mean(MSE[0]),
            'val': np.mean(MSE[1]),
        }, epoch)
        summary.add_scalars('LPIPS', {
            'train': np.mean(LPIPS[0]),
            'val': np.mean(LPIPS[1]),
        }, epoch)
        summary.add_scalars('SSIM', {
            'train': np.mean(SSIM[0]),
            'val': np.mean(SSIM[1]),
        }, epoch)

    summary.close()
    print("训练完成")

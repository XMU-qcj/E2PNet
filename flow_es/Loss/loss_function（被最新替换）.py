#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2021/2/1 16:39
# @Author        : Xuesheng Bian
# @Email         : xbc0809@gmail.com
# @File          : loss_function.py
# @Description   : 

import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import tensorflow as tf

L1loss = nn.L1Loss()
BCEloss = nn.BCELoss()

def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = delta.pow(2)+epsilon**2
    loss=loss.pow(alpha)
    loss = torch.mean(loss)
    return loss

# def charbonnier_loss2(delta, alpha=0.45, epsilon=1e-3):
#     loss = tf.reduce_mean(tf.pow(tf.square(delta)+tf.square(epsilon), alpha))
#     return loss#原始TF实现，与上面复现的已验证结果一致

#根据输入的光流对图像进行重采样操作
def warp_images_with_flow(image, flow):
    height = flow.size(2)
    width = flow.size(3)
    flow_x, flow_y = torch.split(flow, 1, 1)
    coord_x ,coord_y = torch.meshgrid(torch.arange(0, width), torch.arange(0, height))
    if torch.cuda.is_available():
        coord_x = coord_x.cuda()
        coord_y = coord_y.cuda()
    pos_x = coord_x.unsqueeze(0).unsqueeze(0) + flow_x
    pos_y = coord_y.unsqueeze(0).unsqueeze(0) + flow_y
    # warped_points = torch.cat([(pos_x / width) * 2 - 1, (pos_y / height) * 2 - 1], 1)
    warped_points = torch.cat([ pos_y,pos_x], 1)
    warped_points=warped_points.permute([0, 2, 3, 1])
    warped_points=(warped_points/height)* 2 - 1
    # resample_image=F.grid_sample(image,warped_points , align_corners=False, padding_mode='border')
    resample_image = F.grid_sample(image, warped_points, mode="nearest",align_corners=False, padding_mode='border')
    #align_corners 为真时，对边缘更友好
    return resample_image

def compute_smoothness_loss(flow,margin_mask=None):#mask
    flow_ucrop = flow[:,:, 1:, :]
    flow_dcrop = flow[:,:, :-1, :]
    flow_lcrop = flow[:,:,:,1:]
    flow_rcrop = flow[:,:,:,:-1]

    flow_ulcrop = flow[:,:, 1:, 1:]
    flow_drcrop = flow[:,:, :-1, :-1]
    flow_dlcrop = flow[:,:, :-1, 1:,]
    flow_urcrop = flow[:,:, 1:, :-1,]
    if margin_mask==None:
        smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                          charbonnier_loss(flow_ucrop - flow_dcrop) + \
                          charbonnier_loss(flow_ulcrop - flow_drcrop) + \
                          charbonnier_loss(flow_dlcrop - flow_urcrop)
        smoothness_loss /= 4.
    else:#事件出现位置不参与平滑
        scalar_margin_mask = 1 - margin_mask#有事件的位置是1，但有事件的位置不参与计算，因此需要进行取反
        scalar_margin_mask=scalar_margin_mask.bool()
        scalar_margin_mask=torch.stack((scalar_margin_mask.squeeze(),scalar_margin_mask.squeeze()),dim=1)#光流2通道，复制堆叠

        mask_ucrop = scalar_margin_mask[:, :, 1:, :]#同flow图，对mask进行处理、对齐
        mask_dcrop = scalar_margin_mask[:, :, :-1, :]#与flow一一对应
        mask_lcrop = scalar_margin_mask[:, :, :, 1:]
        mask_rcrop = scalar_margin_mask[:, :, :, :-1]

        mask_ulcrop = scalar_margin_mask[:, :, 1:, 1:]
        mask_drcrop = scalar_margin_mask[:, :, :-1, :-1]
        mask_dlcrop = scalar_margin_mask[:, :, :-1, 1:, ]
        mask_urcrop = scalar_margin_mask[:, :, 1:, :-1, ]

        lrmask=torch.logical_and(mask_lcrop,mask_rcrop)#andshi
        udmask = torch.logical_and(mask_ucrop, mask_dcrop)
        uldrmask = torch.logical_and(mask_ulcrop, mask_drcrop)
        dlurmask = torch.logical_and(mask_dlcrop, mask_urcrop)

        smoothness_loss = charbonnier_loss((flow_lcrop - flow_rcrop)[lrmask]) + \
                          charbonnier_loss((flow_ucrop - flow_dcrop)[udmask]) + \
                          charbonnier_loss((flow_ulcrop - flow_drcrop)[uldrmask]) + \
                          charbonnier_loss((flow_dlcrop - flow_urcrop)[dlurmask])
        smoothness_loss /= 4.
    return smoothness_loss

def compute_photometric_loss(frame1, frame2, flow, flow_mask=None):

        Pixel_difference = warp_images_with_flow(frame2, flow)-frame1  #两个不同方向求差
        # Pixel_difference = frame1-warp_images_with_flow(frame2, flow) #两个不同方向求差
        # Pixel_difference = warp_images_with_flow(frame1, flow) - frame2  # 两个不同方向求差
        # Pixel_difference = frame2-warp_images_with_flow(frame1, flow) #两个不同方向求差

        # shownp = Pixel_difference.squeeze().detach().cpu().numpy()
        # shownp=shownp[1]#batch >1就要进行截取
        # cv2.imshow("photometric_loss", shownp)
        # cv2.waitKey(0)

        if flow_mask==None:
            photometric_loss=torch.sum(charbonnier_loss(Pixel_difference))
        # torch.sum(torch.pow(torch.pow(Pixel_difference, 2) + self.delta ** 2, self.alpha))
        else:
            mask=flow_mask[:,0,:,:]
            mask=mask.unsqueeze(1).bool()
            # test=Pixel_difference[mask]
            photometric_loss = torch.sum(charbonnier_loss(Pixel_difference[mask]))
        return photometric_loss

def PhotoMetric_Loss2(frame1, frame2, flow, flow_mask=None):
    """"""
    warped_images = warp_images_with_flow(frame2, flow)
    Pixel_difference=warped_images- frame1

    # Pixel_difference = warp_images_with_flow(frame2, flow) - frame1  # 两个不同方向求差
    # Pixel_difference = frame1-warp_images_with_flow(frame2, flow) #两个不同方向求差
    # Pixel_difference = warp_images_with_flow(frame1, flow) - frame2  # 两个不同方向求差
    # Pixel_difference = frame2-warp_images_with_flow(frame1, flow) #两个不同方向求差

    showpic = warped_images.squeeze().detach().cpu().numpy()
    showpic=showpic[0]
    cv2.imshow("warped_images", showpic)
    cv2.waitKey(0)

    shownp=Pixel_difference.squeeze().detach().cpu().numpy()
    shownp=shownp[0]
    cv2.imshow("photometric_loss", shownp)
    cv2.waitKey(0)

    frame_diffenrece=frame2-frame1
    frame_diffenrece=frame_diffenrece.squeeze().detach().cpu().numpy()
    frame_diffenrece=frame_diffenrece[0]
    cv2.imshow("frame_diffenrece", frame_diffenrece)
    cv2.waitKey(0)

    if flow_mask == None:
        photometric_loss = torch.sum(charbonnier_loss(Pixel_difference))
    else:
        mask = flow_mask[:, 0, :, :]
        mask = mask.unsqueeze(1).bool()
        photometric_loss = torch.sum(charbonnier_loss(Pixel_difference[mask]))
    # torch.sum(torch.pow(torch.pow(photometric, 2) + self.delta ** 2, self.alpha))
    # test=torch.sum(Pixel_difference)
    # tflag=torch.sum(frame1-frame2)
    return photometric_loss

def GT_Flow_loss(scalar_pred_flow,scalar_gtflow,scalar_outer_mask=None):
    Pixel_difference = scalar_pred_flow - scalar_gtflow
    if scalar_outer_mask==None:
        gtflow_loss=torch.sum(charbonnier_loss(torch.norm(Pixel_difference,dim=1)))
    else:
        mask = scalar_outer_mask[:, 0, :, :]
        mask = mask.bool()#这里是直接对比光流，光流有2通道
        flow_difference=torch.norm(Pixel_difference,dim=1)#uv光流计算向量距离

        # showpic=flow_difference/flow_difference.max()
        # showpic = showpic.squeeze().detach().cpu().numpy()
        # cv2.imshow("flow_difference", showpic)
        # cv2.waitKey(0)

        selected_pixel=flow_difference[mask]#对向量距离（模长）进行mask
        gtflow_loss = torch.sum(charbonnier_loss(selected_pixel))
    return gtflow_loss

# class Smoothness(nn.Module):
#     """"""
#
#     def __init__(self, kernel):
#         """Constructor for Smoothness"""
#         super(Smoothness, self).__init__()
#         self.kernel = torch.randn(1, 1, kernel, kernel)
#         self.kernel = self.kernel.cuda() if torch.cuda.is_available() else self.kernel
#         self.kernel[...] = -1.
#         self.kernel[:, :, kernel // 2, kernel // 2] = kernel * kernel - 1
#         self.kernel.requires_grad = False
#
#     def forward(self, u, v, margin_mask=1.0):
#         out = torch.sum(
#             # torch.pow(F.conv2d(u, self.kernel), 2) * margin_mask + torch.pow(F.conv2d(v, self.kernel), 2) * margin_mask
#             torch.pow(F.conv2d(u, self.kernel, padding=1), 2) * margin_mask
#             + torch.pow(F.conv2d(v, self.kernel, padding=1), 2) * margin_mask)
#         return out


class DeepSupervise(nn.Module):
    """"""
    def __init__(self, delta=0.001, alpha=0.45, kernel=3, gamma=0.5):
        """Constructor for DeepSupervise"""
        super(DeepSupervise, self).__init__()
        self.gamma = gamma
        # self.photometric = PhotoMetric(delta, alpha)
        # self.compute_smoothness_loss = Smoothness(kernel)
        self.w_photometric = 1
        self.w_MSE = 1
        self.w_smoothness = 1
        self.kernel_size = kernel
    def forward(self, image_frames, pred, gt_flow_frame, outer_mask=None, margin_mask=None):
        # outer_mask：完整的外部mask，屏蔽不需要进行梯度计算的位置（如没有事件的位置和没有光流GT的位置）
        # margin_mask：标记出现事件的边缘，事件边缘位置不参与平滑
        scalar_frames = [F.interpolate(image_frames, m.size()[2:], mode='bilinear',align_corners=True) for m in pred]  # 对图像进行下采样
        scalar_frames = [torch.split(frame, 1, 1) for frame in scalar_frames]  # 前后帧分开(光流mask的2通道一样)
        scalar_gtflow = [F.interpolate(gt_flow_frame, m.size()[2:], mode='bilinear',align_corners=True) for m in pred]  # 对gt光流多尺度下采样
        if outer_mask != None:  # 预测和GT先乘以mask后再计算loss
            outer_mask = outer_mask.float()
            scalar_outer_mask = [F.interpolate(outer_mask, m.size()[2:],mode='bilinear',align_corners=True) for m in pred]  # MASK进行对应尺寸的下采样
            scalar_outer_mask = [scalar_outer_mask[i].bool() for i in range(len(pred))]
            # GT监督的计算函数直接索引mask，不用再进行mask相乘了
            # masked_scalar_outer_frames = [torch.mul(scalar_frames[i], scalar_outer_mask[i]) for i in range(len(pred))]#不计算mask区域的损失
            # masked_scalar_outer_frames = [torch.split(frame, 1, 1) for frame in masked_scalar_outer_frames]#前后帧分开(光流mask的2通道一样)
            # masked_scalar_gtflow = [torch.mul(scalar_gtflow[i], scalar_outer_mask[i]) for i in range(len(pred))]
            # masked_pred_flow = [torch.mul(pred[i], scalar_outer_mask[i]) for i in range(len(pred))]
            # losses_photometric = [self.w_photometric * PhotoMetric(masked_scalar_outer_frames[i][0], masked_scalar_outer_frames[i][1]
            #                                                        , pred[i] ,scalar_outer_mask[i])#scalar_outer_mask for i in range(len(pred))]
            losses_photometric = [self.w_photometric * compute_photometric_loss(scalar_frames[i][0], scalar_frames[i][1]
                                                                                , pred[i], scalar_outer_mask[i])  # scalar_outer_mask
                                  for i in range(len(pred))]
            # test_losses_photometric = [self.w_photometric * PhotoMetric_Loss2(scalar_frames[i][0], scalar_frames[i][1]
            #                                                                     , pred[i], scalar_outer_mask[i])# scalar_outer_mask
            #                       for i in range(len(pred))]


            avg_photometric_loss=torch.sum(torch.stack(losses_photometric))/len(pred)#按照原文的方法
            # loss_gtflow_frames = [self.w_MSE * torch.sum(torch.pow(masked_pred_flow[i] - masked_scalar_gtflow[i], 2))
            #                       for i in  range(len(pred))]
            loss_gtflow_frames=[self.w_MSE * torch.sum(GT_Flow_loss(pred[i],scalar_gtflow[i],scalar_outer_mask[i]))
                                for i in  range(len(pred))]#光流GT直接监督
            avg_gtflow_loss = torch.sum(torch.stack(loss_gtflow_frames)) / len(pred)  # 按照原文的方法
            if margin_mask != None:#有事件位置作为mask，事件边缘不参与平滑
                margin_mask=margin_mask.unsqueeze(1).float()
                temp_size=pred[-1].size()[2:]
                temp_size2=[temp_size[i]*2 for i in range(2)]#尺寸为原来2倍
                temp_margin_mask=F.interpolate(margin_mask,temp_size2,mode='bilinear',align_corners=True)#上采样
                temp_margin_mask=temp_margin_mask.bool().int().float()#取整
                scalar_margin_mask=F.interpolate(temp_margin_mask,temp_size,mode='bilinear',align_corners=True)
                scalar_margin_mask=scalar_margin_mask.bool().int().float()#mask扩散到8个邻域中，对邻域也mask
                #有事件的位置是1，需要进行取反
                scalar_margin_mask = [F.interpolate(scalar_margin_mask, m.size()[2:],mode='bilinear')
                                      for m in pred]#边缘mask要对邻域也mask
                losses_smoothness = [self.w_smoothness * compute_smoothness_loss(pred[i], scalar_margin_mask[i]) for i
                    in range(len(pred))]
                avg_smoothness_loss = torch.sum(torch.stack(losses_smoothness)) / len(pred)
            else:#全图计算平滑损失
                # scalar_margin_mask = [torch.ones( m.size()[2:]) for m in pred]
                # scalar_margin_mask = [F.interpolate(image_frames, m.size()[2:]) for m in pred]
                losses_smoothness = [
                    self.w_smoothness * compute_smoothness_loss(pred[i]) for i in
                    range(len(pred))]
                avg_smoothness_loss = torch.sum(torch.stack(losses_smoothness)) / len(pred)
                # losses_smoothness = [self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
            #                      range(len(pred))]
        else:#没有mask，直接计算全部光流损失
            # scalar_frames = [F.interpolate(image_frames, m.size()[2:]) for m in pred]
            # scalar_frames = [torch.split(frame, 1, 1) for frame in scalar_frames]
            # scalar_gtflow = [F.interpolate(gt_flow_frame, m.size()[2:]) for m in pred]
            losses_photometric = [
                self.w_photometric * compute_photometric_loss(scalar_frames[i][0], scalar_frames[i][1], pred[i]) for i in
                range(len(pred))]
            avg_photometric_loss = torch.sum(torch.stack(losses_photometric)) / len(pred)  # 按照原文的方法(不同尺度下平均）)

            loss_gtflow_frames = [self.w_MSE * torch.sum(GT_Flow_loss(pred[i], scalar_gtflow[i]))
                                  for i in range(len(pred))]  # 光流GT直接监督
            avg_gtflow_loss = torch.sum(torch.stack(loss_gtflow_frames)) / len(pred)  # 按照原文的方法

            if margin_mask != None:
                # margin_mask = [F.interpolate(margin_mask, m.size()[2:]) for m in pred]
                # # losses_smoothness = [
                # #     self.w_smoothness * compute_smoothness_loss(pred[i] , margin_mask[i]) for i
                # #     in  range(len(pred))]
                margin_mask = margin_mask.unsqueeze(1).float()
                temp_size = pred[-1].size()[2:]
                temp_size2 = [temp_size[i] * 2 for i in range(2)]  # 尺寸为原来2倍
                temp_margin_mask = F.interpolate(margin_mask, temp_size2, mode='bilinear', align_corners=True)  # 上采样
                temp_margin_mask = temp_margin_mask.bool().int().float()  # 取整
                scalar_margin_mask = F.interpolate(temp_margin_mask, temp_size, mode='bilinear', align_corners=True)
                scalar_margin_mask = scalar_margin_mask.bool().int().float()  # mask扩散到8个邻域中，对邻域也mask
                # 有事件的位置是1，需要进行取反
                scalar_margin_mask = [F.interpolate(scalar_margin_mask, m.size()[2:], mode='bilinear')
                                      for m in pred]  # 边缘mask要对邻域也mask
                losses_smoothness = [self.w_smoothness * compute_smoothness_loss(pred[i], scalar_margin_mask[i]) for i
                                     in range(len(pred))]
                avg_smoothness_loss = torch.sum(torch.stack(losses_smoothness)) / len(pred)
            else:
                losses_smoothness = [
                    self.w_smoothness * compute_smoothness_loss(pred[i]) for i in
                    range(len(pred))]
                avg_smoothness_loss = torch.sum(torch.stack(losses_smoothness)) / len(pred)

            # losses_smoothness = [self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
            #                      range(len(pred))]
        # tsmoothness = Smoothness(kernel)
        # test1=compute_smoothness_loss(pred[0][:, 0:1, ...], pred[0][:, 1:, :])
        # test2=compute_smoothness_loss(pred[0])
        # test1=charbonnier_loss(pred[0])
        # with tf.Session()as sees:
        #     test2 = charbonnier_loss2(pred[0].detach().cpu().numpy())
        #     test2=sees.run(test2)

        zero=torch.zeros_like(pred[3])
        test1=PhotoMetric_Loss2(scalar_frames[3][0], scalar_frames[3][1], scalar_gtflow[3])
        test2 = PhotoMetric_Loss2(scalar_frames[3][0], scalar_frames[3][0], zero, scalar_outer_mask[3])
        test3 = PhotoMetric_Loss2(scalar_frames[3][0], scalar_frames[3][0], zero, scalar_outer_mask[3])

        # test_smooth=[compute_smoothness_loss(pred[i]) for i in range(len(pred))]
        print("photometric:", torch.sum(torch.stack(losses_photometric)).data
              , " smoothness:", torch.sum(torch.stack(losses_smoothness)).data
              , "GTframes", torch.sum(torch.stack(loss_gtflow_frames)).data)

        # test=torch.stack(losses_photometric + losses_smoothness + loss_gtflow_frames)
        # test2=torch.sum(test)
        # test3=avg_photometric_loss+avg_gtflow_loss+avg_smoothness_loss
        return avg_photometric_loss+avg_gtflow_loss+avg_smoothness_loss
        # return torch.sum(torch.stack(losses_photometric + losses_smoothness + loss_gtflow_frames))





if __name__ == '__main__':
    maps = [torch.rand(1, 2, 32 * (2 ** i), 40 * (2 ** i)) for i in range(0, 4)]
    fl = torch.rand(1, 2, 256, 320)
    In = torch.rand(1, 2, 256, 320)

    # warp_images_with_flow(In, fl)
    loss_f = DeepSupervise(3)

    loss = loss_f(In, maps, fl)
    print(loss)

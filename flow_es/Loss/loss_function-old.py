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

L1loss = nn.L1Loss()
BCEloss = nn.BCELoss()


class PhotoMetric(nn.Module):
    """"""

    def __init__(self, delta, alpha):
        """Constructor for FlowRecover"""
        super(PhotoMetric, self).__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, frame1, frame2, flow):
        photometric = frame1 - warp_images_with_flow(frame2, flow)
        return torch.sum(torch.pow(torch.pow(photometric, 2) + self.delta ** 2, self.alpha))


class Smoothness(nn.Module):
    """"""

    def __init__(self, kernel):
        """Constructor for Smoothness"""
        super(Smoothness, self).__init__()
        self.kernel = torch.randn(1, 1, kernel, kernel)
        self.kernel = self.kernel.cuda() if torch.cuda.is_available() else self.kernel
        self.kernel[...] = -1.
        self.kernel[:, :, kernel // 2, kernel // 2] = kernel * kernel - 1
        self.kernel.requires_grad = False

    def forward(self, u, v, margin_mask=1.0):
        out = torch.sum(
            # torch.pow(F.conv2d(u, self.kernel), 2) * margin_mask + torch.pow(F.conv2d(v, self.kernel), 2) * margin_mask
            torch.pow(F.conv2d(u, self.kernel, padding=1), 2) * margin_mask + torch.pow(
                F.conv2d(v, self.kernel, padding=1), 2) * margin_mask)
        return out


class DeepSupervise(nn.Module):
    """"""

    def __init__(self, delta=0.001, alpha=0.45, kernel=3, gamma=0.5):
        """Constructor for DeepSupervise"""
        super(DeepSupervise, self).__init__()
        self.gamma = gamma
        self.photometric = PhotoMetric(delta, alpha)
        self.smoothness = Smoothness(kernel)
        self.w_photometric = 0.5
        self.w_MSE = 1
        self.w_smoothness = 0.25
        self.kernel_size = kernel

    def forward(self, input_frames, pred, flow_frame, outer_mask=None, margin_mask=None):
        # outer_mask：完整的外部mask，屏蔽不需要进行梯度计算的位置（如没有事件的位置和没有光流GT的位置）
        # margin_mask：标记出现事件的边缘

        if outer_mask != None:  # 预测和GT先乘以mask后再计算loss
            outer_mask = outer_mask.float()
            # print(outer_mask.size())

            scalar_mask = [F.interpolate(outer_mask, m.size()[2:]) for m in pred]  # MASK进行对应尺寸的下采样

            scalar_frames = [F.interpolate(input_frames, m.size()[2:]) for m in pred]  # 对图像进行下采样
            scalar_margin_mask = [F.interpolate(input_frames, m.size()[2:]) for m in pred]
            # scalar_frames = [torch.split(frame, 1, 1) for frame in scalar_frames]
            masked_scalar_frames = [torch.mul(scalar_frames[i], scalar_mask[i]) for i in range(len(pred))]
            masked_scalar_frames = [torch.split(frame, 1, 1) for frame in masked_scalar_frames]

            scalar_gtflow = [F.interpolate(flow_frame, m.size()[2:]) for m in pred]  # 对gt光流下采样
            masked_scalar_gtflow = [torch.mul(scalar_gtflow[i], scalar_mask[i]) for i in
                                    range(len(pred))]
            masked_pred_flow = [torch.mul(pred[i], scalar_mask[i]) for i in
                                range(len(pred))]

            losses_photometric = [
                self.w_photometric * self.photometric(masked_scalar_frames[i][0], masked_scalar_frames[i][1], pred[i])
                for i in
                range(len(pred))]
            flow_frames_loss = [self.w_MSE * torch.sum(torch.pow(masked_pred_flow[i] - masked_scalar_gtflow[i], 2)) for
                                i in
                                range(len(pred))]
            if margin_mask != None:
                margin_mask = [F.interpolate(margin_mask, m.size()[2:]) for m in pred]
                losses_smoothness = [
                    self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask[i]) for i
                    in
                    range(len(pred))]
            else:
                margin_mask = 1.0
                losses_smoothness = [
                    self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
                    range(len(pred))]
            # losses_smoothness = [self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
            #                      range(len(pred))]

        else:
            scalar_frames = [F.interpolate(input_frames, m.size()[2:]) for m in pred]
            scalar_frames = [torch.split(frame, 1, 1) for frame in scalar_frames]
            scalar_gtflow = [F.interpolate(flow_frame, m.size()[2:]) for m in pred]

            losses_photometric = [
                self.w_photometric * self.photometric(scalar_frames[i][0], scalar_frames[i][1], pred[i]) for i in
                range(len(pred))]
            flow_frames_loss = [self.w_MSE * torch.sum(torch.pow(pred[i] - scalar_gtflow[i], 2)) for i in
                                range(len(pred))]
            if margin_mask != None:
                margin_mask = [F.interpolate(margin_mask, m.size()[2:]) for m in pred]
                losses_smoothness = [
                    self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask[i]) for i
                    in
                    range(len(pred))]
            else:
                margin_mask = 1.0
                losses_smoothness = [
                    self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
                    range(len(pred))]
            # losses_smoothness = [self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
            #                      range(len(pred))]

        return torch.sum(torch.stack(losses_photometric + losses_smoothness + flow_frames_loss))


def warp_images_with_flow(image, flow):
    height = flow.size(2)
    width = flow.size(3)
    flow_x, flow_y = torch.split(flow, 1, 1)

    coord_x, coord_y = torch.meshgrid(torch.arange(0, width), torch.arange(0, height))
    coord_x = coord_x.transpose(0, 1)
    coord_y = coord_y.transpose(0, 1)
    if torch.cuda.is_available():
        coord_x = coord_x.cuda()
        coord_y = coord_y.cuda()

    pos_x = coord_x.unsqueeze(0).unsqueeze(0) + flow_x
    pos_y = coord_y.unsqueeze(0).unsqueeze(0) + flow_y

    # pos_x = coord_x.expand(flow_x.size()) + flow_x
    # pos_y = coord_y.expand(flow_y.size()) + flow_y

    warped_points = torch.cat([(pos_x / width) * 2 - 1, (pos_y / height) * 2 - 1], 1)
    return F.grid_sample(image, warped_points.permute([0, 2, 3, 1]), align_corners=True, padding_mode='border')


if __name__ == '__main__':
    maps = [torch.rand(1, 2, 32 * (2 ** i), 40 * (2 ** i)) for i in range(0, 4)]
    fl = torch.rand(1, 2, 256, 320)
    In = torch.rand(1, 2, 256, 320)

    # warp_images_with_flow(In, fl)
    loss_f = DeepSupervise(3)

    loss = loss_f(In, maps, fl)
    print(loss)

import torch
import torch.nn.functional as F
import numpy as np


def resize_like(image, target):
    return torch.nn.functional.interpolate(image,
                                           size=target.shape[2:],
                                           mode='bilinear',
                                           align_corners=True)

#对图片根据光流进行变换【N*C*H*W】
def apply_flow(prev, flow, ignore_ooi=False):
    """ Warp prev to cur through flow
    I_c(x,y) = I_p(x+f_u(x,y), y+f_v(x,y))
    """
    batch_size, _, height, width = prev.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                                                height, 1).type_as(prev)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                                                 width, 1).transpose(1, 2).type_as(prev)

    x_shift = flow[:, 0, :, :]  # / flow.shape[-1]
    y_shift = flow[:, 1, :, :]  # / flow.shape[-1]
    flow_field = torch.stack((x_base + x_shift, y_base + y_shift), dim=3)

    output = F.grid_sample(prev, 2 * flow_field - 1, mode='bilinear',
                           padding_mode='zeros')

    if ignore_ooi:
        return output, output.new_ones(output.shape).byte()
    else:
        mask = (flow_field[..., 0] > -1.0) * (flow_field[..., 0] < 1.0) \
               * (flow_field[..., 1] > -1.0) * (flow_field[..., 1] < 1.0)

        return output, mask[:, None, :, :]

#XY方向梯度计算，错位相减
def gradient(I):
    """
    Arguments:
    - I - shape N1,...,Nn,C,H,W
    Returns:
    - dx - shape N1,...,Nn,C,H,W
    - dy - shape N1,...,Nn,C,H,W
    """

    dy = I.new_zeros(I.shape)
    dx = I.new_zeros(I.shape)

    dy[..., 1:, :] = I[..., 1:, :] - I[..., :-1, :]#位置0到位置-1之前的数，不是倒序
    dx[..., :, 1:] = I[..., :, 1:] - I[..., :, :-1]#错位相减

    return dx, dy

#xy方向上光流的平均（如果有mask的话就按照mask取）
def flow_smoothness(flow, mask=None):
    dx, dy = gradient(flow)

    if mask is not None:
        mask = mask.expand(-1, 2, -1, -1)
        loss = (charbonnier_loss(dx[mask]) \
                + charbonnier_loss(dy[mask])) / 2.
    else:
        loss = (charbonnier_loss(dx) \
                + charbonnier_loss(dy)) / 2.

    return loss

#(error ** 2. + 1e-5 ** 2.) ** alpha
def charbonnier_loss(error, alpha=0.45, mask=None):
    charbonnier = (error ** 2. + 1e-5 ** 2.) ** alpha
    if mask is not None:
        mask = mask.float()
        loss = torch.mean(torch.sum(charbonnier * mask, dim=(1, 2, 3)) / \
                          torch.sum(mask, dim=(1, 2, 3)))
    else:
        loss = torch.mean(charbonnier)
    return loss


def squared_error(input, target):
    return torch.sum((input - target) ** 2)


def generator_loss(loss_func, fake):
    if loss_func == "wgan":
        return -fake.mean()
    elif loss_func == "gan":
        return F.binary_cross_entropy_with_logits(input=fake, target=torch.ones_like(fake).cuda())
    elif loss_func == "lsgan":
        return squared_error(input=fake, target=torch.ones_like(fake).cuda()).mean()
    elif loss_func == "hinge":
        return -fake.mean()
    else:
        raise Exception("Invalid loss_function")


def discriminator_loss(loss_func, real, fake):
    if loss_func == "wgan":
        real_loss = -real.mean()
        fake_loss = fake.mean()
    elif loss_func == "gan":
        real_loss = F.binary_cross_entropy_with_logits(input=real,
                                                       target=torch.ones_like(real).cuda())
        fake_loss = F.binary_cross_entropy_with_logits(input=fake,
                                                       target=torch.zeros_like(fake).cuda())
    elif loss_func == "lsgan":
        real_loss = squared_error(input=real, target=torch.ones_like(real).cuda()).mean()
        fake_loss = squared_error(input=fake, target=torch.zeros_like(fake).cuda()).mean()
    elif loss_func == "hinge":
        real_loss = F.relu(1.0 - real).mean()
        fake_loss = F.relu(1.0 + fake).mean()
    else:
        raise Exception("Invalid loss_function")

    return real_loss + fake_loss


def multi_scale_flow_loss(prev_image, next_image, flow_list, valid_mask):
    # Multi-scale loss
    total_photo_loss = 0.
    total_smooth_loss = 0.
    pred_image_list = []

    for i, flow in enumerate(flow_list):
        # upsample the flow
        up_flow = F.interpolate(flow, size=(prev_image.shape[2], prev_image.shape[3]),
                                mode='nearest')
        # apply the flow to the current image
        pred_image, mask = apply_flow(prev_image, up_flow)

        scale = 2. ** (i - len(flow_list) + 1)
        total_photo_loss += charbonnier_loss(pred_image - next_image, mask=mask * valid_mask) * scale
        total_smooth_loss += flow_smoothness(flow) * scale

        pred_image_list.append(pred_image)

    return total_photo_loss, total_smooth_loss, pred_image_list
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Losses.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2020/10/12 16:09
# @Desc  :

import torch
import torch.nn as nn
from torch.nn import functional as F
# from utility.flow_refine_avg_timestamp_loss import cal_all_avg_timestamp


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
            torch.pow(F.conv2d(u, self.kernel,padding=1), 2) * margin_mask + torch.pow(F.conv2d(v, self.kernel,padding=1), 2) * margin_mask)
        return out


class DeepSupervise(nn.Module):
    """"""

    def __init__(self, delta=0.001, alpha=0.45, kernel=3, gamma=0.5):
        """Constructor for DeepSupervise"""
        super(DeepSupervise, self).__init__()
        self.gamma = gamma
        self.photometric = PhotoMetric(delta, alpha)
        self.smoothness = Smoothness(kernel)
        self.w_photometric=0.5
        self.w_MSE=1
        self.w_smoothness=0.25
        self.kernel_size=kernel

    def forward(self, input_frames, pred, flow_frame, outer_mask=None,margin_mask=None):
        #outer_mask：完整的外部mask，屏蔽不需要进行梯度计算的位置（如没有事件的位置和没有光流GT的位置）
        #margin_mask：标记出现事件的边缘

        if outer_mask!=None:#预测和GT先乘以mask后再计算loss
            outer_mask=outer_mask.float()
            scalar_mask=[F.interpolate(outer_mask, m.size()[2:]) for m in pred]#MASK进行对应尺寸的下采样

            scalar_frames = [F.interpolate(input_frames, m.size()[2:]) for m in pred]#对图像进行下采样
            scalar_margin_mask=[F.interpolate(input_frames, m.size()[2:]) for m in pred]
            # scalar_frames = [torch.split(frame, 1, 1) for frame in scalar_frames]
            masked_scalar_frames = [torch.mul(scalar_frames[i], scalar_mask[i]) for i in range(len(pred))]
            masked_scalar_frames=[torch.split(frame, 1, 1) for frame in masked_scalar_frames]

            scalar_gtflow = [F.interpolate(flow_frame, m.size()[2:]) for m in pred]#对gt光流下采样
            masked_scalar_gtflow=[torch.mul(scalar_gtflow[i], scalar_mask[i]) for i in
                                        range(len(pred))]
            masked_pred_flow = [torch.mul(pred[i], scalar_mask[i]) for i in
                                        range(len(pred))]

            losses_photometric = [self.w_photometric * self.photometric(masked_scalar_frames[i][0], masked_scalar_frames[i][1], pred[i]) for i in
                                  range(len(pred))]
            flow_frames_loss = [self.w_MSE * torch.sum(torch.pow(masked_pred_flow[i] - masked_scalar_gtflow[i], 2)) for i in
                                range(len(pred))]
            if margin_mask!=None:
                margin_mask = [F.interpolate(margin_mask, m.size()[2:] ) for m in pred]
                losses_smoothness = [
                    self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask[i]) for i
                    in
                    range(len(pred))]
            else:
                margin_mask=1.0
                losses_smoothness = [
                    self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
                    range(len(pred))]
            # losses_smoothness = [self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask) for i in
            #                      range(len(pred))]

        else:
            scalar_frames = [F.interpolate(input_frames, m.size()[2:]) for m in pred]
            scalar_frames = [torch.split(frame, 1, 1) for frame in scalar_frames]
            scalar_gtflow = [F.interpolate(flow_frame, m.size()[2:]) for m in pred]

            losses_photometric = [self.w_photometric * self.photometric(scalar_frames[i][0], scalar_frames[i][1], pred[i]) for i in
                                  range(len(pred))]
            flow_frames_loss = [self.w_MSE * torch.sum(torch.pow(pred[i] - scalar_gtflow[i], 2)) for i in
                                range(len(pred))]
            if margin_mask!=None:
                margin_mask = [F.interpolate(margin_mask, m.size()[2:]) for m in pred]
                losses_smoothness = [
                    self.w_smoothness * self.smoothness(pred[i][:, 0:1, ...], pred[i][:, 1:, :], margin_mask[i]) for i in
                    range(len(pred))]
            else:
                margin_mask=1.0
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


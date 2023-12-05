import os
import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure
from skimage.metrics import structural_similarity 
from numpy import *
import sys
sys.path.append("/media/XH-8T/qcj/cGAN_new_re/Loss")
from PerceptualSimilarity import models
class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', use_gpu=True, gpu_ids=[0]):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu, gpu_ids=gpu_ids)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return self.weight * dist.mean()
    
class l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)
class compute_infer_loss:
    def __init__(self, ):
        self.perceptual_loss_fn = perceptual_loss(net='alex')
        self.mse_loss_fn = l2_loss()
        self.ssim_loss_fn = structural_similarity
        self.loss = {'perceptual_loss': [],
                     'mse_loss': [],
                     'ssim_loss': []}

    def write_loss(self, pred_img, gt_img):
        self.loss['perceptual_loss'].append(self.perceptual_loss_fn(pred_img, gt_img).item())
        self.loss['mse_loss'].append(self.mse_loss_fn(pred_img, gt_img).item())
        self.loss['ssim_loss'].append(self.ssim_loss_fn(pred_img.squeeze().cpu().numpy(), 
                                                        gt_img.squeeze().cpu().numpy(), data_range=1.))
        mean_lpips = mean(self.loss['perceptual_loss'])
        mean_mse = mean(self.loss['mse_loss'])
        mean_ssim = mean(self.loss['ssim_loss'])
            
        return {'mean_lpips': mean_lpips, 'mean_mse': mean_mse, 'mean_ssim': mean_ssim}
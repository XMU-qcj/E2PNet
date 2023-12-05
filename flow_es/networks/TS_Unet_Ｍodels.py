#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Ｍodels.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/7/14 13:48
# @Desc  :

import copy
import torch
import torch.nn as nn
import numpy as np
from networks.vit_seg_modeling import VisionTransformer, CONFIGS, Encoder


def patch_tensor(img_tensor, mesh):
    img_tensor = img_tensor.view(img_tensor.size(0), img_tensor.size(1), mesh, img_tensor.size(2) // mesh,
                                 mesh, img_tensor.size(3) // mesh)
    img_tensor = img_tensor.permute(0, 2, 4, 1, 3, 5)
    img_tensor = img_tensor.contiguous()
    img_tensor = img_tensor.view(-1, img_tensor.size(-3), img_tensor.size(-2), img_tensor.size(-1))
    # img_tensor = img_tensor.view(img_tensor.size(0), img_tensor.size(1), -1)
    return img_tensor


def un_patch_tensor(img_tensor, mesh):
    # img_tensor = img_tensor.view(img_tensor.size(0), img_tensor.size(1), int(np.sqrt(img_tensor.size(-1))),
    #                              int(np.sqrt(img_tensor.size(-1))))
    img_tensor = img_tensor.view(-1, mesh, mesh, img_tensor.size(-3), img_tensor.size(-2), img_tensor.size(-1))
    img_tensor = img_tensor.permute(0, 3, 1, 4, 2, 5)
    img_tensor = img_tensor.contiguous()
    img_tensor = img_tensor.view(img_tensor.size(0), img_tensor.size(1), img_tensor.size(2) * img_tensor.size(3),
                                 img_tensor.size(4) * img_tensor.size(5))
    return img_tensor


class TSUnet(nn.Module):
    """"""

    def __init__(self, meshes=2):
        """Constructor for TSUnet"""
        super(TSUnet, self).__init__()

        S_config_vit = copy.deepcopy(CONFIGS['R50-ViT-B_16'])  # R50-ViT-B_16
        S_config_vit.n_classes = 2  # 9
        S_config_vit.n_skip = 3  # 3
        S_config_vit.custom = 16 // meshes

        self.S_part = VisionTransformer(S_config_vit, 128, 9)
        self.mesh = meshes
        self.fine = nn.Sequential(
            nn.Conv2d(6, 2, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        patch_size = x.size(-1) // self.mesh
        start = patch_size // 2
        moved_x = torch.cat([x.clone()[..., start:, :], x.clone()[..., 0:start, :]], -2)
        moved_y = torch.cat([x.clone()[..., start:], x.clone()[..., 0:start]], -1)

        if torch.cuda.is_available():
            moved_x = moved_x.cuda()
            moved_y = moved_y.cuda()

        moved_x_patch = patch_tensor(moved_x, self.mesh)
        moved_y_patch = patch_tensor(moved_y, self.mesh)

        out_move_x = self.S_part(moved_x_patch)
        out_move_y = self.S_part(moved_y_patch)

        out_x = patch_tensor(x, self.mesh)
        out_x = self.S_part(out_x)
        for i in range(len(out_x)):
            out_x[i] = un_patch_tensor(out_x[i], self.mesh)
            patch_size = out_x[i].size(-1) // self.mesh
            start = patch_size // 2
            out_move_x[i] = un_patch_tensor(out_move_x[i], self.mesh)
            out_move_x[i][..., -patch_size:, :] = 0 * out_move_x[i][..., -patch_size:, :]
            out_move_x[i] = torch.cat(
                [out_move_x[i][..., -start:, :], out_move_x[i][..., 0: - start, :]], - 2)

            out_move_y[i] = un_patch_tensor(out_move_y[i], self.mesh)
            out_move_y[i][..., -patch_size:] = 0 * out_move_y[i][..., -patch_size:]
            out_move_y[i] = torch.cat([out_move_y[i][..., -start:], out_move_y[i][..., 0:-start]], -1)
            out_x[i] = torch.cat([out_x[i], out_move_x[i], out_move_y[i]], 1)
            out_x[i] = self.fine(out_x[i])

        return out_x


if __name__ == '__main__':
    net = TSUnet(8)
    x = torch.randn(2, 10, 256, 256)
    y = net(x)
    print(y[-1].size())
    # print(y.size())

    '''
    import cv2
    import numpy as np

    img = cv2.imread('1.jpg')

    mesh = 4
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    print(img_tensor.size())
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    # img_tensor = img_tensor.view(img_tensor.size(0), img_tensor.size(1), mesh, img_tensor.size(2) // mesh,
    #                              mesh, img_tensor.size(3) // mesh)
    # print(img_tensor.size())
    #
    # img_tensor = img_tensor.permute(0, 2, 4, 1, 3, 5)
    # img_tensor = img_tensor.contiguous()
    # print(img_tensor.size())
    #
    # img_tensor = img_tensor.view(-1, img_tensor.size(-3), img_tensor.size(-2), img_tensor.size(-1))
    # print(img_tensor.size())
    #
    # img_tensor = img_tensor.view(-1, mesh, mesh, img_tensor.size(-3), img_tensor.size(-2), img_tensor.size(-1))
    # print(img_tensor.size())

    img_tensor = patch_tensor(img_tensor, mesh)

    # img_tensor = img_tensor.view(-1, mesh, mesh, img_tensor.size(-3), img_tensor.size(-2), img_tensor.size(-1))
    # print(img_tensor.size())

    # row = []
    # for i in range(mesh):
    #     column = []
    #     for j in range(mesh):
    #         patch = img_tensor[0, i, j, ...]
    #         patch = patch.permute(1, 2, 0)
    #         img = patch.numpy()
    #         column.append(img)
    #     column = np.concatenate(column, 1)
    #     row.append(column)
    # row = np.concatenate(row, 0)
    # cv2.imshow('merged', row)
    # cv2.waitKey()

    img_tensor = un_patch_tensor(img_tensor, 4)
    print(img_tensor.size())

    img_tensor = img_tensor.permute(0, 2, 3, 1)
    img = img_tensor.numpy()
    cv2.imshow('recover', img[0])
    cv2.waitKey()
    '''

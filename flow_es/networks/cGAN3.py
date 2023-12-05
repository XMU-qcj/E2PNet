#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : cGAN.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2021/8/6 14:48
# @Desc  :

import torch
import torch.nn as nn

class Conv(nn.Sequential):
    """"""

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dropout=0):
        """Constructor for Cpnv"""
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channel)
        self.ac = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return super(Conv, self).forward(x)


class DeConv(nn.Sequential):
    """"""

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, out_padding, dropout=0.):
        """Constructor for DeConv"""
        super(DeConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding,
                                       output_padding=out_padding)
        self.norm = nn.BatchNorm2d(out_channel)
        self.ac = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return super(DeConv, self).forward(x)


class Generator(nn.Module):
    """"""

    def __init__(self, in_channel=3, out_channel=1):
        """Constructor for Generator"""
        super(Generator, self).__init__()
        self.conv1 = Conv(in_channel, 64, 4, 2, 1)
        self.conv2 = Conv(64, 128, 4, 2, 1)
        self.conv3 = Conv(128, 256, 4, 2, 1)
        self.conv4 = Conv(256, 512, 4, 2, 1)
        self.conv5 = Conv(512, 512, 4, 2, 1)
        self.conv6 = Conv(512, 512, 4, 2, 1)
        self.conv7 = Conv(512, 512, 4, 2, 1)
        self.conv8 = Conv(512, 512, 4, 2, 1)

        self.deconv7 = DeConv(512, 512, 4, 2, 1, 0, 0.5)
        self.deconv6 = DeConv(512 * 2, 512, 4, 2, 1, 0, 0.5)
        self.deconv5 = DeConv(512 * 2, 512, 4, 2, 1, 0, 0.5)
        self.deconv4 = DeConv(512 * 2, 512, 4, 2, 1, 0, 0)
        self.deconv3 = DeConv(512 * 2, 256, 4, 2, 1, 0, 0)
        self.deconv2 = DeConv(256 * 2, 128, 4, 2, 1, 0, 0)
        self.deconv1 = DeConv(128 * 2, 64, 4, 2, 1, 0, 0)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, out_channel, 4, 2, 1, 0)
            # ,nn.Tanh()
        )
        for m in self.modules():
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    torch.nn.init.normal_(mm.weight, mean=0, std=0.02)
                if isinstance(mm, nn.BatchNorm2d):
                    torch.nn.init.normal_(mm.weight, mean=0, std=0.02)
    def forward(self, x,):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        out7 = self.conv7(out6)
        out = self.conv8(out7)

        out = self.deconv7(out)
        out = self.deconv6(torch.cat([out, out7], 1))
        out = self.deconv5(torch.cat([out, out6], 1))
        out = self.deconv4(torch.cat([out, out5], 1))
        out = self.deconv3(torch.cat([out, out4], 1))
        out = self.deconv2(torch.cat([out, out3], 1))
        out = self.deconv1(torch.cat([out, out2], 1))
        out = self.last(out)

        return out


class Discriminator(nn.Module):
    """"""

    def __init__(self, in_channel=3):
        """Constructor for Discriminator"""
        super(Discriminator, self).__init__()
        self.conv1 = Conv(in_channel, 64, 4, 2, 1, 0)
        self.conv2 = Conv(64, 128, 4, 2, 1, 0, )
        self.conv3 = Conv(128, 256, 4, 2, 1, 0, )
        self.conv4 = Conv(256, 512, 4, 1, 1, 0, )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        for m in self.modules():
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    torch.nn.init.normal_(mm.weight, mean=0, std=0.02)
                if isinstance(mm, nn.BatchNorm2d):
                    torch.nn.init.normal_(mm.weight, mean=0, std=0.02)

    def forward(self, x,):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
    
# class Discriminator(nn.Sequential):
#     """"""

#     def __init__(self, in_channel=3):
#         """Constructor for Discriminator"""
#         super(Discriminator, self).__init__()
#         self.conv1 = Conv(in_channel, 64, 4, 2, 1, 0)
#         self.conv2 = Conv(64, 128, 4, 2, 1, 0, )
#         self.conv3 = Conv(128, 256, 4, 2, 1, 0, )
#         self.conv4 = Conv(256, 512, 4, 1, 1, 0, )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(512, 1, 1, 1, 0),
#             nn.Sigmoid()
#         )
#         for m in self.modules():
#             for mm in m.modules():
#                 if isinstance(mm, nn.Conv2d):
#                     torch.nn.init.normal_(mm.weight, mean=0, std=0.02)
#                 if isinstance(mm, nn.BatchNorm2d):
#                     torch.nn.init.normal_(mm.weight, mean=0, std=0.02)

#     def forward(self, x):
#         return super(Discriminator, self).forward(x)


if __name__ == '__main__':
    x = torch.randn(2, 1, 256, 256)
    net = Discriminator(1)
    y = net(x)

    print(y.size())
    pass

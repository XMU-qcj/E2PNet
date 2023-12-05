#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2020/10/10 20:14
# @Desc  :

import torch


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    print(x1.size(), x2.size())
    return x1 + x2


if __name__ == '__main__':
    pass

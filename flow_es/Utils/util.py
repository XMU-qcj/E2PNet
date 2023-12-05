#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2021/2/1 14:37
# @Author        : Xuesheng Bian
# @Email         : xbc0809@gmail.com
# @File          : util.py
# @Description   : 

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


if __name__ == '__main__':
    pass

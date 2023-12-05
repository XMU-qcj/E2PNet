from torch.utils.data import Dataset
import h5py
import numpy as np
import time
import os
import random
import torch


def random_dropout_events(events,  pointgt_flow,dropout_ratio):
    # 对输入的n行二维数据，随机去除一定比例的行数
    # print("input=", np.shape(events)[0])
    if dropout_ratio == 0:
        return events, pointgt_flow
    dropout_num = int(dropout_ratio * np.shape(events)[0])
    full_index = list(range(np.shape(events)[0]))
    dropout_index = random.sample(full_index, dropout_num)
    remain_index = set(full_index) - set(dropout_index)  # 集合操作
    events_flow = events[list(remain_index), :]
    pointgt = pointgt_flow[list(remain_index), :]
    # print("outut=", np.shape(events_flow)[0])
    return events_flow, pointgt
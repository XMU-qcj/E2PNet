import h5py
import time
import os
import random
import torch
import cv2
import torch
import numpy as np
import matplotlib.colors as colors
"""对事件和光流进行可视化的功能性函数"""

#光流可视化
#输入事件，输出可视化图像（只是单纯的正负事件覆盖式更新）
def event2image_VECtor(event_x,event_y,event_p, image_size,start_index=0, end_index=-1):
    '''
    出现绿色 消失红色
    绿色x方向 红色y方向
    :param event:输入的事件流【x,y,t,p】
    :param start_index:事件流的截取起点
    :param end_index:事件流的截取终点
    :return:
    img_event:Event（+-1）的可视化图
    '''
    # img_event = np.zeros((260, 346, 3))
    img_event = np.zeros((image_size[0], image_size[1], 3))
    # img_flow = np.zeros((3,260, 346))
    # m, M = gt.min(), gt.max()
    # m, M = np.min(gt), np.max(gt)
    # print('mM', m, M)
    # gt = (gt - m) / (M - m + 1e-9)#这里做了标准化

    # if event_x.device != "cpu":
    #     event_x = event_x.cpu()
    #     event_y=event_y.cpu()
    #     event_p=event_p.cpu()


    y = event_y[start_index:end_index]  # copy.deepcopy(event[start_index:end_index, 0])
    x = event_x[start_index:end_index]
    p = event_p[start_index:end_index]
    p=p*2-1#把01极性变成+-1极性


    img_event[np.int32(y.flatten()).tolist(), np.int32(
        x.flatten()).tolist(), 2] = (p + 1) * 127.5  # 根据事件，在图像的对应位置上进行更新（最新的覆盖之前的像素）
    img_event[np.int32(y.flatten()).tolist(), np.int32(
        x.flatten()).tolist(), 0] = (1 - p) * 127.5
    img_event = img_event.astype('uint8')
    #红色为正事件

    # gt_[np.int32(x.flatten()).tolist(), np.int32(y.flatten()).tolist(), 0] = x_or
    # gt_[np.int32(x.flatten()).tolist(), np.int32(y.flatten()).tolist(), 1] = y_or

    # gt_=gt[0]
    # img_flow[1,:,:]=gt[0,:,:]/2#U
    # img_flow[2] = gt[1]/2#V

    #
    cv2.imshow("img_event", img_event)
    cv2.waitKey(0)
    # img_event = img_event.transpose([2, 0, 1])#转torch要用

    return img_event # image是Event（+-1）的可视化；gt_是光流图的可视化

#输入事件，输出可视化图像（只是单纯的正负事件覆盖式更新）
def event2image_MVSEC(event_flow, image_size,start_index=0, end_index=-1):
    '''
    出现绿色 消失红色
    绿色x方向 红色y方向
    :param event:输入的事件流【x,y,t,p】
    :param start_index:事件流的截取起点
    :param end_index:事件流的截取终点
    :return:
    img_event:Event（+-1）的可视化图
    '''
    event_x=event_flow[:,0]
    event_y = event_flow[:,1]
    event_p=event_flow[:,3]

    # img_event = np.zeros((260, 346, 3))
    img_event = np.ones((image_size[0], image_size[1], 3)) * 255
    # img_flow = np.zeros((3,260, 346))
    # m, M = gt.min(), gt.max()
    # m, M = np.min(gt), np.max(gt)
    # print('mM', m, M)
    # gt = (gt - m) / (M - m + 1e-9)#这里做了标准化

    y = event_y[start_index:end_index]  # copy.deepcopy(event[start_index:end_index, 0])
    x = event_x[start_index:end_index]
    p = event_p[start_index:end_index]

    # if event.device != "cpu":
    #     event = event.cpu()



    img_event[np.int32(y.flatten()).tolist(), np.int32(
        x.flatten()).tolist(), 2] = (p + 1) * 127.5  # 根据事件，在图像的对应位置上进行更新（最新的覆盖之前的像素）
    img_event[np.int32(y.flatten()).tolist(), np.int32(
        x.flatten()).tolist(), 0] = (1 - p) * 127.5
    img_event[np.int32(y.flatten()).tolist(), np.int32(
        x.flatten()).tolist(), 1] = 0
    img_event = img_event.astype('uint8')
    #红色为正事件

    # gt_[np.int32(x.flatten()).tolist(), np.int32(y.flatten()).tolist(), 0] = x_or
    # gt_[np.int32(x.flatten()).tolist(), np.int32(y.flatten()).tolist(), 1] = y_or

    # gt_=gt[0]
    # img_flow[1,:,:]=gt[0,:,:]/2#U
    # img_flow[2] = gt[1]/2#V

    #测试的时候可视化
    # cv2.imshow("img_event", img_event)
    # cv2.waitKey(0)

    # img_event = img_event.transpose([2, 0, 1])

    return img_event # image是Event（+-1）的可视化；gt_是光流图的可视化


def normalize_event_image(event_image, clamp_val=2., normalize_events=True):
    if not normalize_events:
        return event_image
    else:
        return torch.clamp(event_image, 0, clamp_val) / clamp_val  # + 1.) / 2.


def gen_event_images(event_volume, prefix, device="cuda", clamp_val=2., normalize_events=True):
    n_bins = int(event_volume.shape[1] / 2)#正负分开，所以是总数/2
    time_range = torch.tensor(np.linspace(0.1, 1, n_bins), dtype=torch.float32).to(device)
    time_range = torch.reshape(time_range, (1, n_bins, 1, 1))#把时间戳均匀离散化到0.1-1之间

    pos_event_image = torch.sum(
        event_volume[:, :n_bins, ...] * time_range / \
        (torch.sum(event_volume[:, :n_bins, ...], dim=1, keepdim=True) + 1e-5),
        dim=1, keepdim=True)
    neg_event_image = torch.sum(
        event_volume[:, n_bins:, ...] * time_range / \
        (torch.sum(event_volume[:, n_bins:, ...], dim=1, keepdim=True) + 1e-5),
        dim=1, keepdim=True)

    outputs = {
        '{}_event_time_image'.format(prefix): (pos_event_image + neg_event_image) / 2.,
        '{}_event_image'.format(prefix): normalize_event_image(
            torch.sum(event_volume, dim=1, keepdim=True)),
        '{}_event_image_x'.format(prefix): normalize_event_image(
            torch.sum(event_volume.permute((0, 2, 1, 3)), dim=1, keepdim=True),
            normalize_events=normalize_events),
        '{}_event_image_y'.format(prefix): normalize_event_image(
            torch.sum(event_volume.permute(0, 3, 1, 2), dim=1, keepdim=True),
            normalize_events=normalize_events)
    }
    return outputs

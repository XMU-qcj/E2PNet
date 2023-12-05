from torch.utils.data import Dataset
import h5py
import numpy as np
import time
import os
import random
import torch
import cv2
from torch import nn
"""利用输入的uv光流对事件进行投影前的位置矫正，之后采样投影，得到全局的时间戳作为LOSS"""
WIDTH=346
HEIGHT=260

# 输入开始和结束的时间位置，根据对应的时间戳，,返回对应时间戳内的Events起、止位置(下标)
def find_events_index(begin_ind, end_ind, events, gtstamp):
    '''

    :param begin_ind: 开始统计是时间戳
    :param end_ind: 结束统计是时间戳
    :param events: events原始数据
    :param gtstamp: GT值的时间戳
    :return: 最接近开始和结束GT时间戳的Events时间戳
    '''
    # 输入开始和结束的时间位置，根据对应的时间戳，,返回对应时间戳内的Events起、止位置(下标)
    # ！！！！要求begin<end
    assert begin_ind < end_ind, '起始下标应该小于结束'
    begin_stamp = gtstamp[begin_ind]
    end_stamp = gtstamp[end_ind]
    event_long = np.shape(events)[0]
    # global event_index
    # event_index = 0
    max_index = event_long  # 取Event序列头尾的索引（编号）
    # print(event_long)
    min_index = 0
    max_stamp = events[max_index - 1][2]  # events= (14071304, 4)
    min_stamp = events[min_index][2]  # 取Event序列的时间戳（绝对时间）
    # 下面先二分搜索begin_stamp
    while max_stamp > min_stamp:  # max与min交叉后二分查找完成
        mid_index = (max_index + min_index) // 2  # mid 取中间位置的指针
        mid_stamp = events[mid_index][2]
        if mid_stamp > begin_stamp:
            max_index = mid_index - 1
            max_stamp = events[max_index][2]
        elif mid_stamp < begin_stamp:
            min_index = mid_index + 1
            min_stamp = events[min_index][2]
        else:
            print("mid_stamp=", round(mid_stamp, 10), "begin_stamp=", round(begin_stamp, 10))
            nearest_begin_index = mid_index
            break
            # nearest_begin_stamp=mid_stamp
    if min_stamp - begin_stamp < max_stamp - begin_stamp:
        nearest_begin_index = min_index
        # nearest_begin_stamp=min_stamp
    else:
        nearest_begin_index = max_index
        # nearest_begin_stamp=max_stamp
        ########
    # 接下来查找end_index
    max_index = event_long  # 取Event序列头尾的索引（编号）
    min_index = nearest_begin_index  #
    max_stamp = events[max_index - 1][2]  # events= (14071304, 4)
    min_stamp = events[min_index][2]  # 取Event序列的时间戳（绝对时间）
    # 二分搜索end_ind
    while max_stamp > min_stamp:  # max与min交叉后二分查找完成
        mid_index = (max_index + min_index) // 2  # mid 取中间位置的指针
        mid_stamp = events[mid_index][2]
        if mid_stamp > end_stamp:
            max_index = mid_index - 1
            max_stamp = events[max_index][2]
        elif mid_stamp < end_stamp:
            min_index = mid_index + 1
            min_stamp = events[min_index][2]
        else:
            print("mid_stamp=", round(mid_stamp, 10), "begin_stamp=", round(end_stamp, 10))
            nearest_end_index = mid_index
            break
    if min_stamp - end_stamp < max_stamp - end_stamp:
        nearest_end_index = min_index
    else:
        nearest_end_index = max_index

    return (nearest_begin_index, nearest_end_index)

#从gtstamp中的search_ind索引，在img_timestamp中找到最接近的序号
def find_img_index(search_ind, gtstamp, img_timestamp):
    '''
    timestamp>>>>(img-ts)>>>>>index

    '''
    begin_stamp = gtstamp[search_ind]
    img_long = np.shape(img_timestamp)[0]

    max_index = img_long  # 取Event序列头尾的索引（编号）
    # print(img_long)
    min_index = 0
    max_stamp = img_timestamp[max_index - 1]  # events= (14071304, 4)
    min_stamp = img_timestamp[min_index] # 取Event序列的时间戳（绝对时间）
    # 下面先二分搜索begin_stamp
    while max_stamp > min_stamp:  # max与min交叉后二分查找完成
        mid_index = (max_index + min_index) // 2  # mid 取中间位置的指针
        mid_stamp = img_timestamp[mid_index]
        if mid_stamp > begin_stamp:
            max_index = mid_index - 1
            max_stamp = img_timestamp[max_index]
        elif mid_stamp < begin_stamp:
            min_index = mid_index + 1
            min_stamp = img_timestamp[min_index]
        else:
            print("mid_stamp=", round(mid_stamp, 10), "begin_stamp=", round(begin_stamp, 10))
            nearest_begin_index = mid_index
            break
            # nearest_begin_stamp=mid_stamp
    if min_stamp - begin_stamp < max_stamp - begin_stamp:
        nearest_begin_index = min_index
        # nearest_begin_stamp=min_stamp
    else:
        nearest_begin_index = max_index
        # nearest_begin_stamp=max_stamp
        ########
    # 接下来查找end_index


    return nearest_begin_index

#光流可视化
def flow2image(optflow):
    if optflow.device != "cpu":
        optflow = optflow.cpu()
    optflow = optflow.detach().numpy()
    h, w = optflow.shape[1:3]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(optflow[0, ...], optflow[1, ...])  # ang【0,2pi】（弧度角）；
    hsv[..., 0] = ang * 180 / np.pi / 2 # 对角度进行转换
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 对幅值进行归一化
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[..., 2] = 255  # HSV颜色模型：色调（H），饱和度（S），明度（V）
    img_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # img_flow=img_flow.transpose([1,2,0])
    """可视化"""
    cv2.imshow("img_flow", img_flow)
    cv2.waitKey(0)

    return img_flow

#输入事件，输出可视化图像
def event2image(event,  start_index=0, end_index=-1):
    '''
    出现绿色 消失红色
    绿色x方向 红色y方向
    :param event:输入的事件流【x,y,t,p】
    :param start_index:事件流的截取起点
    :param end_index:事件流的截取终点
    :return:
    img_event:Event（+-1）的可视化图
    '''
    img_event = np.zeros((260, 346, 3))
    # img_flow = np.zeros((3,260, 346))
    # m, M = gt.min(), gt.max()
    # m, M = np.min(gt), np.max(gt)
    # print('mM', m, M)
    # gt = (gt - m) / (M - m + 1e-9)#这里做了标准化
    if event.device != "cpu":
        event = event.cpu()


    y = event[start_index:end_index, 0]  # copy.deepcopy(event[start_index:end_index, 0])
    x = event[start_index:end_index, 1]
    p = event[start_index:end_index, 3]


    img_event[np.int32(x.flatten()).tolist(), np.int32(
        y.flatten()).tolist(), 2] = (p + 1) * 127.5  # 根据事件，在图像的对应位置上进行更新（最新的覆盖之前的像素）
    img_event[np.int32(x.flatten()).tolist(), np.int32(
        y.flatten()).tolist(), 0] = (1 - p) * 127.5
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

    return img_event # image是Event（+-1）的可视化；gt_是光流图的可视化

def calc_floor_ceil_delta(x):
    #输入的是归一化到N个时间片段中的时间序号（非整数）
    x_fl = torch.floor(x + 1e-8)#向下取整
    x_ce = torch.ceil(x - 1e-8)#向上取整
    x_ce_fake = torch.floor(x) + 1#向下取整再+1

    dx_ce = x - x_fl#x减去x的整数位，剩下小数位【x-torch.floor(x + 1e-8)】
    dx_fl = x_ce_fake - x#x的整数位-x，剩下（-x的小数位），再+1，=1-x的小数位
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]
# return【x的向下取整；1-x的小数位(就是实际的权重)】【x向上取整；x的小数位】
#返回的这个dx_就是权重，经过了1-R的计算

#根据输入的事件（事件的时间取出整数和小数），根据极性投影到各个时间片上，正负分开
def create_update(w,dw, h,dh, t,  p, vol_size):
    H=vol_size[1]
    W=vol_size[2]
    t=t.float()
    assert (w >= 0).byte().all() and (w <= W).byte().all()#all():张量tensor中所有元素都是True, 才返回True
    assert (h >= 0).byte().all() and (h <= H).byte().all()
    if w.is_cuda:
        vol_mul = torch.where(p > 0,
                              torch.ones(p.shape, dtype=torch.long).cuda(),
                              torch.zeros(p.shape, dtype=torch.long).cuda())
    else:
        vol_mul = torch.where(p > 0,
                              torch.ones(p.shape, dtype=torch.long),
                              torch.zeros(p.shape, dtype=torch.long))

    #torch.where(condition, x, y)  # condition是条件，满足条件就返回x，不满足就返回y
    #经测试是逐元素的操作
    #如果极性<0,返回1，否则返回零（负极性加到后一帧去？？？）
    #后面的索引是按照一维数组进行处理的，所以inds的计算是累加的操作【N*W*H】
    #          W    *      H       *（t的整数部分+现在）
    inds = (H*W) * (0 + vol_mul) \
           + W * (h-1) \
           + w

    vals_weight = dw*dh
    vals_weighted_t=vals_weight*t
    vals_weight.float()
    vals_weighted_t=vals_weighted_t.float()

    return inds, vals_weight,vals_weighted_t


#主函数，根据输入的光流，对事件进行校正后投影，计算平均时间损失
def cal_past_avg_timestamp(events,start_flow,volume_size):


    WIDTH=volume_size[1]
    HEIGHT=volume_size[0]

    #events=torch.from_numpy(events)
    start_time=events[0,2]
    events[:,2]=events[:,2]-start_time#对事件的时间戳进行帧内的归一化，从零开始
    w = events[:, 0].long()
    h = events[:, 1].long()
    t = events[:, 2]
    p=events[:, 3].long()
    events_num=events.shape[0]
    delt_t=t/t[-1]
    delt_t=delt_t.float()
    event_order=torch.arange(events_num).cuda()
    '''接下来根据事件的xy坐标取出对应位置的uv光流'''
    flatten_index=(HEIGHT*h+w)
    flatten_index=flatten_index.long()
    """对事件的w、h坐标根据光流进行矫正"""
    w_refined=w.float().put_(event_order,-delt_t*start_flow[0].view(-1)[flatten_index],accumulate=True)
    h_refined = h.float().put_(event_order, -delt_t * start_flow[1].view(-1)[flatten_index], accumulate=True)
    #上面获取了根据光流矫正后的事件wh坐标
    """接下来是将各个点进行投影采样，正负分开"""
    #每个像素位置有一个存数值的和计数的
    #取出xy坐标的整数和小数
    # refined_events_t_num = torch.zeros((2 * HEIGHT * WIDTH))
    w_floor,w_ceil=calc_floor_ceil_delta(w_refined)
    h_floor, h_ceil = calc_floor_ceil_delta(h_refined)#计算w、h轴上的相邻方向投影坐标以及权重
    """对矫正后位置超出像素边界的点进行处理"""
    w_floor[0] = torch.where(w_floor[0] < 0,torch.zeros(w_floor[0].shape, dtype=torch.long,).cuda(),w_floor[0])
    w_ceil[0] = torch.where(w_ceil[0] < 0, torch.zeros(w_ceil[0].shape, dtype=torch.long).cuda(), w_ceil[0])
    h_floor[0] = torch.where(h_floor[0] < 0, torch.zeros(h_floor[0].shape, dtype=torch.long).cuda(), h_floor[0])
    h_ceil[0] = torch.where(h_ceil[0] < 0, torch.zeros(h_ceil[0].shape, dtype=torch.long).cuda(), h_ceil[0])
    w_ceil[0] = torch.where(w_ceil[0] >= WIDTH, torch.zeros(w_ceil[0].shape, dtype=torch.long).cuda()+WIDTH-1, w_ceil[0])
    w_floor[0] = torch.where(w_floor[0] >= WIDTH, torch.zeros(w_floor[0].shape, dtype=torch.long).cuda() + WIDTH - 1, w_floor[0])
    h_ceil[0] = torch.where(h_ceil[0] >= HEIGHT, torch.zeros(h_ceil[0].shape, dtype=torch.long).cuda()+HEIGHT-1, h_ceil[0])
    h_floor[0] = torch.where(h_floor[0] >= HEIGHT, torch.zeros(h_floor[0].shape, dtype=torch.long).cuda() + HEIGHT - 1, h_floor[0])
    """开始累积投影的过程"""
    final_volume_size = [2, HEIGHT, WIDTH]
    refined_events_t_sum = torch.zeros(final_volume_size)  # 第多出个通道用于正负分开处理（分子的权重）
    refined_events_weight_sum = torch.zeros(final_volume_size)  # 记录分母的权重
    if events.is_cuda:
        refined_events_t_sum=refined_events_t_sum.cuda()
        refined_events_weight_sum=refined_events_weight_sum.cuda()

    inds_nn, weight_nn,weighted_t_nn=create_update(w_floor[0], w_floor[1], h_floor[0], h_floor[1], t, p, final_volume_size)
    refined_events_weight_sum.view(-1).put_(inds_nn,weight_nn,accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_nn, weighted_t_nn, accumulate=True)#对事件的左下邻居进行累积

    inds_np, weight_np, weighted_t_np = create_update(w_floor[0], w_floor[1], h_ceil[0], h_ceil[1], t, p,final_volume_size)
    refined_events_weight_sum.view(-1).put_(inds_np, weight_np, accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_np, weighted_t_np, accumulate=True)  # 对事件的左上邻居进行累积

    inds_pn, weight_pn, weighted_t_pn = create_update(w_ceil[0], w_ceil[1], h_floor[0], h_floor[1], t, p,final_volume_size)
    refined_events_weight_sum.view(-1).put_(inds_pn, weight_pn, accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_pn, weighted_t_pn, accumulate=True)  # 对事件的右下邻居进行累积

    inds_pp, weight_pp, weighted_t_pp = create_update(w_ceil[0], w_ceil[1], h_ceil[0], h_ceil[1], t, p,final_volume_size)
    refined_events_weight_sum.view(-1).put_(inds_pp, weight_pp, accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_pp, weighted_t_pp, accumulate=True)  # 对事件的右上邻居进行累积
    refined_event_t_img = refined_events_t_sum / (refined_events_weight_sum + 1e-8)
    """可视化对比（调试用）
    
    #原始的事件图像处理
    old_events_t_sum = torch.zeros(final_volume_size)  # 第多出个通道用于正负分开处理（分子的权重）
    old_events_weight_sum = torch.zeros(final_volume_size)
    temp_ones=torch.ones(w.shape, dtype=torch.long)
    inds_old, weight_old, weighted_t_old = create_update(w, temp_ones, h, temp_ones, t, p, final_volume_size)
    old_events_weight_sum.view(-1).put_(inds_old, weight_old, accumulate=True)
    old_events_t_sum.view(-1).put_(inds_old, weighted_t_old, accumulate=True)
    old_events_t_img=old_events_t_sum/(old_events_weight_sum+1e-8)
    old_img_t_avg = torch.sum(old_events_t_img)
    
    binary_refined_event_img = torch.where(refined_events_weight_sum > 0,
                                           torch.ones(refined_event_t_img.shape, dtype=torch.int8) * 127,
                                           torch.zeros(refined_event_t_img.shape, dtype=torch.int8))
    binary_old_event_img=torch.where(old_events_weight_sum>0,
                               torch.ones(old_events_t_img.shape, dtype=torch.int8) *127,
                               torch.zeros(old_events_t_img.shape, dtype=torch.int8))
    binary_old_event_img=binary_old_event_img.numpy()
    binary_refined_event_img=binary_refined_event_img.numpy()
    # binary_old_event_img[0].dtype="float32"
    old_events_t_img=old_events_t_img.numpy()
    refined_event_t_img=refined_event_t_img.numpy()
    old_events_t_img = cv2.normalize(old_events_t_img, None, 0, 1, cv2.NORM_MINMAX)  # 对幅值进行归一化
    refined_event_t_img=cv2.normalize(refined_event_t_img, None, 0, 1, cv2.NORM_MINMAX)
    old_t_img=cv2.resize(old_events_t_img[1],(0,0),fx=2,fy=2)
    refined_t_img=cv2.resize(refined_event_t_img[1],(0,0),fx=2,fy=2)
    cv2.imshow("old_t_img", old_t_img)
    cv2.waitKey(0)
    cv2.imshow("refined_t_img", refined_t_img)
    cv2.waitKey(0)"""

    """接下来计算整张图像的全部像素平均时间戳"""
    full_img_t_sum=torch.sum(refined_event_t_img)#正负两个通道各自的像素
    # full_img_t_sum.backward()
    return full_img_t_sum
#主函数，根据输入的光流，对事件进行校正后投影，计算平均时间损失(向后投影)
def cal_future_avg_timestamp(events, end_flow, volume_size):

    WIDTH=volume_size[1]
    HEIGHT=volume_size[0]
    # events=torch.from_numpy(events)
    start_time=events[0,2]
    events[:,2]=events[:,2]-start_time#对事件的时间戳进行帧内的归一化，从零开始
    w = events[:, 0].long()
    h = events[:, 1].long()
    t = events[:, 2]
    p=events[:, 3].long()
    events_num=events.shape[0]
    delt_t=1-(t/t[-1])
    delt_t=delt_t.float()
    event_order=torch.arange(events_num)
    '''接下来根据事件的xy坐标取出对应位置的uv光流'''
    flatten_index=(HEIGHT*h+w)
    flatten_index=flatten_index.long()
    """对事件的w、h坐标根据光流进行矫正"""
    if events.is_cuda:
        event_order=event_order.cuda()

    w_refined=w.float().put_(event_order, delt_t * end_flow[0].view(-1)[flatten_index], accumulate=True)
    h_refined = h.float().put_(event_order, delt_t * end_flow[1].view(-1)[flatten_index], accumulate=True)
    #上面获取了根据光流矫正后的事件wh坐标
    """接下来是将各个点进行投影采样，正负分开"""
    #每个像素位置有一个存数值的和计数的
    #取出xy坐标的整数和小数
    # refined_events_t_num = torch.zeros((2 * HEIGHT * WIDTH))
    w_floor,w_ceil=calc_floor_ceil_delta(w_refined)
    h_floor, h_ceil = calc_floor_ceil_delta(h_refined)#计算w、h轴上的相邻方向投影坐标以及权重
    """对矫正后位置超出像素边界的点进行处理"""
    w_floor[0] = torch.where(w_floor[0] < 0,torch.zeros(w_floor[0].shape, dtype=torch.long).cuda(),w_floor[0])
    w_ceil[0] = torch.where(w_ceil[0] < 0, torch.zeros(w_ceil[0].shape, dtype=torch.long).cuda(), w_ceil[0])
    h_floor[0] = torch.where(h_floor[0] < 0, torch.zeros(h_floor[0].shape, dtype=torch.long).cuda(), h_floor[0])
    h_ceil[0] = torch.where(h_ceil[0] < 0, torch.zeros(h_ceil[0].shape, dtype=torch.long).cuda(), h_ceil[0])
    w_ceil[0] = torch.where(w_ceil[0] >= WIDTH, torch.zeros(w_ceil[0].shape, dtype=torch.long).cuda()+WIDTH-1, w_ceil[0])
    w_floor[0] = torch.where(w_floor[0] >= WIDTH, torch.zeros(w_floor[0].shape, dtype=torch.long).cuda() + WIDTH - 1, w_floor[0])
    h_ceil[0] = torch.where(h_ceil[0] >= HEIGHT, torch.zeros(h_ceil[0].shape, dtype=torch.long).cuda()+HEIGHT-1, h_ceil[0])
    h_floor[0] = torch.where(h_floor[0] >= HEIGHT, torch.zeros(h_floor[0].shape, dtype=torch.long).cuda() + HEIGHT - 1, h_floor[0])
    """开始累积投影的过程"""
    final_volume_size = [2, HEIGHT, WIDTH]
    refined_events_t_sum = torch.zeros(final_volume_size)  # 第多出个通道用于正负分开处理（分子的权重）
    refined_events_weight_sum = torch.zeros(final_volume_size)  # 记录分母的权重
    inds_nn, weight_nn,weighted_t_nn=create_update(w_floor[0], w_floor[1], h_floor[0], h_floor[1], t, p, final_volume_size)
    if events.is_cuda:
        refined_events_t_sum=refined_events_t_sum.cuda()
        refined_events_weight_sum=refined_events_weight_sum.cuda()
    refined_events_weight_sum.view(-1).put_(inds_nn,weight_nn,accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_nn, weighted_t_nn, accumulate=True)#对事件的左下邻居进行累积

    inds_np, weight_np, weighted_t_np = create_update(w_floor[0], w_floor[1], h_ceil[0], h_ceil[1], t, p,final_volume_size)
    refined_events_weight_sum.view(-1).put_(inds_np, weight_np, accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_np, weighted_t_np, accumulate=True)  # 对事件的左上邻居进行累积

    inds_pn, weight_pn, weighted_t_pn = create_update(w_ceil[0], w_ceil[1], h_floor[0], h_floor[1], t, p,final_volume_size)
    refined_events_weight_sum.view(-1).put_(inds_pn, weight_pn, accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_pn, weighted_t_pn, accumulate=True)  # 对事件的右下邻居进行累积

    inds_pp, weight_pp, weighted_t_pp = create_update(w_ceil[0], w_ceil[1], h_ceil[0], h_ceil[1], t, p,final_volume_size)
    refined_events_weight_sum.view(-1).put_(inds_pp, weight_pp, accumulate=True)
    refined_events_t_sum.view(-1).put_(inds_pp, weighted_t_pp, accumulate=True)  # 对事件的右上邻居进行累积
    refined_event_t_img = refined_events_t_sum / (refined_events_weight_sum + 1e-8)
    """可视化对比（调试用）
    binary_refined_event_img = torch.where(refined_events_weight_sum > 0,
                                           torch.ones(refined_event_t_img.shape, dtype=torch.int8) * 127,
                                           torch.zeros(refined_event_t_img.shape, dtype=torch.int8))
    #原始的事件图像处理
    old_events_t_sum = torch.zeros(final_volume_size)  # 第多出个通道用于正负分开处理（分子的权重）
    old_events_weight_sum = torch.zeros(final_volume_size)
    temp_ones=torch.ones(w.shape, dtype=torch.long)
    inds_old, weight_old, weighted_t_old = create_update(w, temp_ones, h, temp_ones, t, p,
                                                      final_volume_size)
    old_events_weight_sum.view(-1).put_(inds_old, weight_old, accumulate=True)
    old_events_t_sum.view(-1).put_(inds_old, weighted_t_old, accumulate=True)
    old_events_t_img=old_events_t_sum/(old_events_weight_sum+1e-8)
    binary_old_event_img=torch.where(old_events_weight_sum>0,
                               torch.ones(old_events_t_img.shape, dtype=torch.int8) *127,
                               torch.zeros(old_events_t_img.shape, dtype=torch.int8))

    binary_old_event_img=binary_old_event_img.numpy()
    binary_refined_event_img=binary_refined_event_img.numpy()
    # binary_old_event_img[0].dtype="float32"
    old_events_t_img=old_events_t_img.numpy()
    refined_event_t_img=refined_event_t_img.numpy()
    old_events_t_img = cv2.normalize(old_events_t_img, None, 0, 1, cv2.NORM_MINMAX)  # 对幅值进行归一化
    refined_event_t_img=cv2.normalize(refined_event_t_img, None, 0, 1, cv2.NORM_MINMAX)
    old_t_img=cv2.resize(old_events_t_img[1],(0,0),fx=2,fy=2)
    refined_t_img=cv2.resize(refined_event_t_img[1],(0,0),fx=2,fy=2)
    cv2.imshow("old_t_img", old_t_img)
    cv2.waitKey(0)
    cv2.imshow("refined_t_img", refined_t_img)
    cv2.waitKey(0)"""

    """接下来计算整张图像的全部像素平均时间戳"""
    full_img_t_sum=torch.sum(refined_event_t_img)#正负两个通道各自的像素
    # full_img_t_sum.backward()

    return full_img_t_sum


def cal_all_avg_timestamp(events, events_num,start_flow, volume_size):
    """
    分别计算事件向前向后投影校正之后的平均时间戳
    :param events:
    :param start_flow:
    :param end_flow:
    :param volume_size:
    :return:
    """

    events=events[:events_num]
    past_tp_loss = cal_past_avg_timestamp(events, start_flow, volume_size)
    # past_tp_loss.backward()
    future_tp_loss=cal_future_avg_timestamp(events, start_flow, volume_size)
    # future_tp_loss.backward()
    all_avg_timestamp_loss=past_tp_loss +future_tp_loss
    # all_avg_timestamp_loss.backward()
    return all_avg_timestamp_loss

class refine_timestamp_loss(nn.Module):
    def __init__(self,volume_size):
        self.volume_size=volume_size
        super(refine_timestamp_loss,self).__init__()
        torch.autograd.set_detect_anomaly(True)

    def forward(self, events,events_num,opt_flow):
        batch_size=events.shape[0]
        loss=[cal_all_avg_timestamp(events[i], events_num[i], opt_flow[i], self.volume_size)for i in range(batch_size)]
        loss=torch.sum(torch.stack(loss))
        # loss.backward()
        return loss




if __name__ == '__main__':
    frame_index=922#从全序列中需要取的图片帧序号
    #966帧向左运动大
    frame_num=1#要连续取的窗口大小
    h5_data=h5py.File("D:\indoor_flying1_data.hdf5", 'r')
    h5_gt = h5py.File("D:\indoor_flying1_gt.hdf5", 'r')
    events = h5_data['davis']['left']['events']
    gt_flow=h5_gt['davis']['left']['flow_dist']
    gt_flow_ts=h5_gt['davis']['left']['flow_dist_ts']
    img_raw=h5_data['davis']['left']['image_raw']
    img_ts=h5_data['davis']['left']['image_raw_ts']
    # start_flow=gt_flow[frame_index]#;start_time=gt_flow_ts[frame_index]
    for i in range(frame_num):#取出连续帧的正反光流
        if i==0:
            start_flow = gt_flow[frame_index]
        else:
            start_flow=start_flow+gt_flow[frame_index+i]

    for i in range(frame_num):
        if i==0:
            end_flow = gt_flow[frame_index + frame_num]  # ;end_time=gt_flow_ts[frame_index+frame_num]
        else:
            end_flow=end_flow+gt_flow[frame_index+frame_num-i]
    """按照帧时间戳取出帧间事件"""
    event_start_index,event_end_index=find_events_index(frame_index, frame_index+frame_num, events, gt_flow_ts)
    events_num=event_end_index-event_start_index
    """根据flow帧的时间戳，找到最接近的图像帧"""
    img_index=find_img_index(frame_index, gt_flow_ts,img_ts)
    img_now=img_raw[img_index]
    # cv2.imshow("img_event", img_now)
    # cv2.waitKey(0)
    """根据光流对事件进行矫正后重新投影，计算平均时间戳作为LOSS"""
    input_events=torch.from_numpy(events[event_start_index:event_end_index])
    start_flow=torch.from_numpy(start_flow)
    # start_flow=torch.ones(start_flow.shape)*-20#测试用
    end_flow=torch.from_numpy(end_flow)
    e_img=event2image(input_events,  start_index=0, end_index=-1)
    # f_img=flow2image(start_flow)
    volume_size = [HEIGHT, WIDTH]
    """调用计算的主函数"""
    #avg_tp_loss=cal_all_avg_timestamp(events[event_start_index:event_end_index],start_flow,end_flow,volume_size)
    avg_tp_loss=cal_all_avg_timestamp(events[event_start_index:event_end_index],start_flow,volume_size)
    pass
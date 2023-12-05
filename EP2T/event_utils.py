import torch
import torch.nn as nn

import numpy as np

def none_safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if batch:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)
    else:
        return {}
def per_event_feature(events, events_val, vol_size):
    event_images = events.new_zeros(vol_size).permute(1,2,0)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].long()
    
    event_images[x, y, :] += events_val
    return event_images
def init_weights(m):
    """ Initialize weights according to the FlowNet2-pytorch from nvidia """
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.0001, b=0.0001)
        nn.init.xavier_uniform_(m.weight, gain=0.001)

    if isinstance(m, nn.Conv1d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

    if isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-.001, b=0.001)
        nn.init.xavier_uniform_(m.weight, gain=0.01)

def num_trainable_parameters(module):
    trainable_parameters = filter(lambda p: p.requires_grad,
                                  module.parameters())
    return sum([np.prod(p.size()) for p in trainable_parameters])


def num_parameters(network):
    n_params = 0
    modules = list(network.modules())

    for mod in modules:
        parameters = mod.parameters()
        n_params += sum([np.prod(p.size()) for p in parameters])
    return n_params

def calc_floor_ceil_delta(x):
    #输入的是归一化到N个时间片段中的时间序号（非整数）
    x_fl = torch.floor(x + 1e-8)#向下取整
    x_ce = torch.ceil(x - 1e-8)#向上取整

    dx_ce = x - x_fl#x减去x的整数位，剩下小数位【x-torch.floor(x + 1e-8)】
    dx_fl = torch.floor(x) + 1 - x#x的整数位-x，剩下（-x的小数位），再+1，=1-x的小数位
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]
# return【x的向下取整；1-x的小数位】【x向上取整；x的小数位】
#返回的这个就是权重，经过了1-R的计算

#根据输入的事件（事件的时间取出整数和小数），根据极性投影到各个时间片上，正负分开
def create_update(x, y, t, dt, p, vol_size):
    assert (x>=0).byte().all() and (x<vol_size[2]).byte().all()#all():张量tensor中所有元素都是True, 才返回True
    assert (y>=0).byte().all() and (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() and (t<vol_size[0] // 2).byte().all()
    #上面的/2是因为输入的大小之前已经*2后又//2了

    val = torch.div(torch.ones(p.shape, dtype=torch.long).to(x.device) * vol_size[0], 2, rounding_mode='trunc') # rounding_mode='trunc'
    # val = torch.trunc(val)
    # print(p.dtype)
    # print(val.dtype)
    vol_mul = torch.where(p < 0, val, torch.zeros(p.shape, dtype=torch.long).to(x.device))
    #torch.where(condition, x, y)  # condition是条件，满足条件就返回x，不满足就返回y
    #经测试是逐元素的操作
    #如果极性<0,返回1，否则返回零（负极性加到后一帧去？？？）
    #后面的索引是按yin照一维数组进行处理的，所以inds的计算是累加的操作【N*W*H】
    #          W    *      H       *（t的整数部分+现在）
    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals
#返回的inds是展成一维的索引，数值是各个索引对应位置上的更新数值（距离投影平面的（时间）距离）

def gen_event_volume(events, events_val, vol_size):
    # vol_size is [b, t, x, y]
    # events are BxNx4
    batch = events.shape[0]
    npts = events.shape[1]
    volume = events.new_zeros(vol_size)

    # Each is BxN
    x = events[..., 0].long()
    y = events[..., 1].long()
    t = events[..., 2]
    p = events[..., 3].long()
    # Dim is now Bx1
    t_min = t.min(dim=1, keepdim=True)[0]
    t_max = t.max(dim=1, keepdim=True)[0]
    
    t_scaled = (t-t_min) * ((vol_size[1]//2 - 1) / (t_max-t_min))

    xs_fl, xs_ce = calc_floor_ceil_delta(x)
    ys_fl, ys_ce = calc_floor_ceil_delta(y)
    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled)
    
    inds_fl, vals_fl = create_batch_update(xs_fl[0], xs_fl[1],
                                           ys_fl[0], ys_fl[1],
                                           ts_fl[0], ts_fl[1],
                                           events[..., 3],
                                           vol_size)
    events_val = events_val.reshape((-1, events_val.shape[-1])).max(-1)[0]
    vals_fl = events_val * vals_fl
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_batch_update(xs_ce[0], xs_ce[1],
                                           ys_ce[0], ys_ce[1],
                                           ts_ce[0], ts_ce[1],
                                           events[..., 3],
                                           vol_size)
    vals_ce = events_val * vals_ce
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume
def per_event_timing_images(events, events_val, vol_size, val_plus = False):
    event_time_images = events.new_zeros(vol_size)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].long()
    t_min = t.min()
    t_max = t.max()

    vol_mul = torch.where(p < 0,  # 正在前，负在后，负事件需要增加1*W*H的序号
                          torch.ones(p.shape, dtype=torch.long).to(events.device),
                          torch.zeros(p.shape, dtype=torch.long).to(events.device))
    # ind为位置索引
    inds = (vol_size[1] * vol_size[2]) * vol_mul \
           + (vol_size[2]) * y \
           + x
    vals = t / (t_max - t_min)
    if (val_plus == True):
        events_val = events_val.reshape((-1, events_val.shape[-1])).max(-1)[0]
        vals = vals * events_val
    event_time_images.view(-1).put_(inds.long(), vals, accumulate=False)  ##对应的位置更新为对应的时间戳，覆盖

    # cv2.imshow("img_event", img[0])
    # cv2.waitKey(0)
    # cv2.imshow("img_event", img[1])
    # cv2.waitKey(0)
    return event_time_images

def per_stacking_events(events, events_val, vol_size, val_plus = False):
    Stacking_num = vol_size[0]
    vol_size[0] = vol_size[0] * 2  # 正负分开存储再合并

    event_stacking_images_temp = events.new_zeros(vol_size)
    
    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].long()
    
    t_min = t.min()
    t_max = t.max()
    # val = torch.div(((t - t_min) / (t_max - t_min)), ((1 / Stacking_num) + 1e-7))
    # t_index = torch.floor(val)
    t_index = torch.div(((t - t_min) / (t_max - t_min)), ((1 / Stacking_num) + 1e-7), rounding_mode='floor')   # 时间归一化0~1，之后//片数
    
    
    vol_mul = torch.where(p < 0,  # 负在前，正在后，负事件需要增加1*W*H的序号
                          torch.ones(p.shape, dtype=torch.long).to(events.device),
                          torch.zeros(p.shape, dtype=torch.long).to(events.device))
    # ind为位置索引,
    inds = (vol_size[1] * vol_size[2]) * vol_mul * Stacking_num \
           + (vol_size[1] * vol_size[2]) * t_index + (vol_size[2]) * y \
           + x
    inds = inds.long()
    vals = torch.ones(p.shape, dtype=torch.float).to(events.device)  # 对应的位置累加1
    if (val_plus == True):
        events_val = events_val.reshape((-1, events_val.shape[-1])).max(-1)[0]
        vals = vals * events_val
    event_stacking_images_temp.view(-1).put_(inds, vals, accumulate=True)  ##对应的位置累加1
    event_stacking_images = -1 * event_stacking_images_temp[0:Stacking_num] \
                            + event_stacking_images_temp[Stacking_num:]

    imgmax = torch.max(event_stacking_images)
    imgmin = torch.min(event_stacking_images)
    imgmax = abs(max(imgmax, imgmin))
    img = (event_stacking_images / imgmax + 1) / 2  # 为了显示，归一化到【0,1】

    event_stacking_images = event_stacking_images / (imgmax + 1e-7)  # 归一化【-1，1】
    return event_stacking_images
def per_event_counting_images(events, events_val, vol_size, val_plus = False):
    event_count_images = events.new_zeros(vol_size)
    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].long()
    vol_mul = torch.where(p < 0,  # 正在前，负在后，负事件需要增加1*W*H的序号
                          torch.ones(p.shape, dtype=torch.long).to(events.device),
                          torch.zeros(p.shape, dtype=torch.long).to(events.device))
    # ind为位置索引
    inds = (vol_size[1] * vol_size[2]) * vol_mul \
           + (vol_size[2]) * y \
           + x
    vals = torch.ones(p.shape, dtype=torch.float).to(events.device)  # 对应的位置累加1
    if (val_plus == True):
        events_val = events_val.reshape((-1, events_val.shape[-1])).max(-1)[0]
        vals = vals * events_val
    event_count_images.view(-1).put_(inds.long(), vals, accumulate=True)  ##对应的位置累加1
    return event_count_images


def gen_feature_event(events, events_val, vol_size):
    event_feature = events.new_zeros(vol_size)
    x = events[:, 0].long()
    y = events[:, 1].long()
    inds = y * vol_size[1] + x
    events_val = events_val.permute(1,0)
    events_val += torch.torch.randn(events_val.shape).to(events_val.device)
    event_feature.view(vol_size[0], -1).scatter_(1, (inds).unsqueeze(0).expand_as(events_val), events_val)
    return event_feature
#输入
def gen_discretized_event_volume(events, vol_size, val_plus = False, events_val = None):#正负事件分两个通道
    # volume is [t, x, y]
    # events are Nx4
    
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 0]
    y = events[:, 1]
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_max1 = t_max
    t_scaled = (t-t_min) * ((vol_size[0] //2 -1) / (t_max-t_min))#之前输入部分*2，//2是向下取整，-1是因为从0开始编号
    #把时间归一化到vol_size（也就是体素的片数）
    t_max=t_scaled.max()
    
    # 文中公式（2）(向上下两个方向分别取整，以距离为权重累积)
    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    #【x的向下取整；1-x的小数位】【x向上取整；x的小数位】（小数位的数值已经是1-距离的，即越近权重越大）
    #因为采用距离的权重关联核【1-x】，所以上面的第二项是1-x之后的
    #对每一个事件分别在几个投影平面上进行更新操作（x向下的负方向）
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],#x的向下取整；1-x的小数位
                                     events[:, 3],
                                     vol_size)
    # put_(indices, tensor, accumulate=False) → Tensor
    #Tensor.put_(),前面输入的是原始数组，后面括号内输入索引和对应的张量，按照一维数组的序号进行索引，替换自身的函数
    # accumulate为真则不是替换而是累加
    #按照索引和更新数值对几个投影平面的张量进行更新
    if (val_plus != False):
        events_val = events_val.reshape((-1, events_val.shape[-1])).max(-1)[0]
        vals_fl = vals_fl * events_val
    volume.view(-1).put_(inds_fl.long(), vals_fl, accumulate=True)
    #正方向的更新叠加
    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],#【x向上取整；x的小数位】
                                     events[:, 3],
                                     vol_size)
    if (val_plus != False):
        vals_ce = vals_ce * events_val
    volume.view(-1).put_(inds_ce.long(), vals_ce, accumulate=True)
    return volume
#返回的是2*时间切片数，正负事件分开的投影结果
 
def create_batch_update(x, dx, y, dy, t, dt, p, vol_size):
    assert (x>=0).byte().all() and (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all() and (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() and (t<vol_size[0] // 2).byte().all()
    val = torch.div(torch.ones(p.shape, dtype=torch.long).to(x.device) * vol_size[1], 2, rounding_mode='floor')
    # val = torch.div(torch.ones(p.shape, dtype=torch.long).to(x.device) * vol_size[1], 2)
    # val = torch.floor(val)
    vol_mul = torch.where(p < 0,
                          val,
                          torch.zeros(p.shape, dtype=torch.long).to(x.device))
    
    batch_inds = torch.arange(x.shape[0], dtype=torch.long)[:, None]
    batch_inds = batch_inds.repeat((1, x.shape[1]))
    batch_inds = torch.reshape(batch_inds, (-1,)).to(x.device)

    dx = torch.reshape(dx, (-1,))
    dy = torch.reshape(dy, (-1,))
    dt = torch.reshape(dt, (-1,))
    x = torch.reshape(x, (-1,))
    y = torch.reshape(y, (-1,))
    t = torch.reshape(t, (-1,))
    vol_mul = torch.reshape(vol_mul, (-1,)).to(x.device)
    
    inds = vol_size[1]*vol_size[2]*vol_size[3] * batch_inds \
         + (vol_size[2]*vol_size[3]) * (t + vol_mul) \
         + (vol_size[3])*y \
         + x

    vals = dx * dy * dt
    return inds, vals


def gen_batch_discretized_event_volume(events, vol_size):
    # vol_size is [b, t, x, y]
    # events are BxNx4
    
    batch = events.shape[0]
    npts = events.shape[1]
    volume = events.new_zeros(vol_size)

    # Each is BxN
    x = events[..., 0].long()
    y = events[..., 1].long()
    t = events[..., 2]

    # Dim is now Bx1
    t_min = t.min(dim=1, keepdim=True)
    t_max = t.max(dim=1, keepdim=True)
    t_scaled = (t-t_min) * ((vol_size[1]//2 - 1) / (t_max-t_min))

    xs_fl, xs_ce = calc_floor_ceil_delta(x)
    ys_fl, ys_ce = calc_floor_ceil_delta(y)
    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled)
    
    inds_fl, vals_fl = create_batch_update(xs_fl[0], xs_fl[1],
                                           ys_fl[0], ys_fl[1],
                                           ts_fl[0], ts_fl[1],
                                           events[..., 3],
                                           vol_size)
        
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_batch_update(xs_ce[0], xs_ce[1],
                                           ys_ce[0], ys_ce[1],
                                           ts_ce[0], ts_ce[1],
                                           events[..., 3],
                                           vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)

    return volume
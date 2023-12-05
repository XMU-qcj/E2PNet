from torch.utils.data import Dataset
import h5py
import numpy as np
import time
import os
import random
import torch
from Data.event_utils import gen_discretized_event_volume
import cv2
from Utils.result_visualization import flow2image, event2image
from torch.nn import functional as F

# Longest_event = 75600


davis_list = ["./indoor_flying1_500_600.h5 ",  # /home/xhlinxm/MVSEC/indoor_flying/indoor_flying1_data.hdf5
              "./indoor_flying1_events_500_600_pointGT.h5",
              "/media/MVSEV/indoor_flying1_full_new.h5",
              "/media/MVSEV/indoor_flying2_full_new.h5",
              "/media/MVSEV/indoor_flying3_full_new.h5",
              "/media/MVSEV/indoor_flying4_full_new.h5",
              "/media/MVSEV/outdoor_day1_full_new.h5",
              "/media/MVSEV/outdoor_day2_full_new.h5",  # /home/lxh/MVSEC/
              "D://train_data",  # "/home/xhlinxm/MVSEC/train_data",#D://train_data;"/home/xhlinxm/MVSEC/test_data"
              "D://test_data",
              "D:\indoor_flying1_full_new.h5",  # 10
              "D:\indoor_flying2_full_new.h5",  # 11
              "D:\outdoor_day1_full_new.h5",  # 12
              "D:\outdoor_day2_full_new.h5",  # 13
              "E:\MVSEC HDF5\indoor_flying\indoor_flying2_data.hdf5",
              '../indoor_flying1_full_400_450.h5'
              ]

"""
Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow.
x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement.
"""


# def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
#     flow_x_interp = cv2.remap(x_flow,
#                               x_indices,
#                               y_indices,
#                               cv2.INTER_NEAREST)
#     # 重采样操作
#     flow_y_interp = cv2.remap(y_flow,
#                               x_indices,
#                               y_indices,
#                               cv2.INTER_NEAREST)
#
#     x_mask[flow_x_interp == 0] = False
#     y_mask[flow_y_interp == 0] = False
#
#     x_indices += flow_x_interp * scale_factor
#     y_indices += flow_y_interp * scale_factor
#
#     return


class MyData(Dataset):  # 这个类用于准备数据（取出所有数据）
    def __init__(self, file_index, train=True, engine_mask=False, stack_mode="SBT"):
        """
        总体思路，按照图片帧
        :param file_index: 取文件的序号
        :param train: 是训练还是测试
        :param frame_interval: 按照图像帧或者是GT帧分割输出帧
        :param engine_mask: 是否增加引擎盖的mask
        """
        assert stack_mode == "SBT" or stack_mode == "SBN"
        frame_interval = "img"
        self.file_path = davis_list[file_index]  # 文件路径
        self.dropout_ratio = 0.05  # 一定的概率丢弃事件，进行增强
        self.appearance_augmentation = False  # 对投影生成的体素图像进行视觉增强（伽马变换）
        self.normalize_events = True  # 规范化
        self.train = train  # 决定是否进行数据增强，以及取数据的间隔（验证时固定按照图像帧数取）
        self.top_left = [2, 13]  # Top left corner of crop - (d1,d2).
        self.image_size = [256, 256]  # "Final image size for predictions (HxW).
        self.n_time_bins = 5  # 要离散化的片数
        self.max_skip_frames = 10  # SBN时限制最大时长（去除过于稀疏数据）
        self.min_n_events = 2000  # 每帧事件的最小数量（SBT时限制）
        self.flip_x = 0  # "Probability of flipping the volume in x.",
        self.flip_y = 0  # "Probability of flipping the volume in y.",
        self.Longest_event = 95600  # 输出原始事件序列时的最大值
        self.frame_interval = frame_interval  # img 或者gt的间隔
        self.Stacking_num = 3  # 一帧图像帧再被分为n片
        self.SBT_time = 0.03  # 固定事件间隔，除以SBT_num就是每一片的时间（秒）
        self.SBN_num = 60000  # 固定数量的事件叠加成一帧
        self.stack_mode = stack_mode

        if train == True:
            super(MyData, self).__init__()
            print("训练使用数据集：", davis_list[file_index])
        else:
            print("测试使用数据集：", davis_list[file_index])
            self.flip_x = 0  # 测试的时候不翻转，以便可视化
            self.flip_y = 0
            self.frame_interval = "img"  # 测试时默认按照图像帧间隔分割

        self.engine = engine_mask
        self.load(True)
        # self.close()

    def load(self, only_length=False):
        self.full_file = h5py.File(self.file_path, 'r')
        self.events = self.full_file['davis']['left']['events']
        self.img_frame = self.full_file['davis']['left']['img_frame']
        self.img_ts = self.full_file['davis']['left']['image_ts']
        self.img_event_index = self.full_file['davis']['left']['image_event_index']
        self.gtflow_event_index = self.full_file['davis']['left']['gtflow_event_index']
        self.gt_flow_frame = self.full_file['davis']['left']['gtflow']
        self.gt_depth_img = self.full_file['davis']['left']['depth_img']
        self.gt_odometry = self.full_file['davis']['left']['odometry']
        self.gt_flow_ts = self.full_file['davis']['left']['gtflow_ts']
        # self.gt_pose = self.full_file['davis']['left']['pose']
        # self.pose_event_index = self.full_file['davis']['left']['pose_event_index']
        self.raw_image_size = self.img_frame.shape[1:]
        self.num_images = self.img_frame.shape[0]
        self.num_GT_frame = self.gt_flow_frame.shape[0]
        self.start_frame = 0  # 起始帧，全局的偏移量

        self.events_time = self.events[:, 2]

        assert self.gt_flow_ts.shape[0] == self.gt_flow_frame.shape[0]
        assert self.gt_flow_ts.shape[0] == self.gt_odometry.shape[0]
        assert self.gt_flow_ts.shape[0] == self.gt_depth_img.shape[0]
        assert self.gt_flow_ts.shape[0] == self.gtflow_event_index.shape[0]
        assert self.img_frame.shape[0] == self.img_ts.shape[0]

        if self.engine == True:
            self.engine_mask = 1 - cv2.imread('./masks/mask1.bmp', 0)  # 增加对引擎盖区域的mask
            # self.engine_mask=np.ones_like(self.img_frame[0])
            # self.engine_mask[190:,:]=0#190行以下都mask
        else:
            self.engine_mask = np.ones_like(self.img_frame[0])  # 如果不需要mask就设为1

        self.loaded = True

    def close(self):
        self.events = None
        self.image_to_event = None
        self.gtflow_event_index = None
        self.img_frame = None
        self.gt_flow_frame = None
        self.gt_depth_img = None
        self.gt_odometry = None
        # self.gt_pose = None
        self.pose_event_index = None

        self.full_file.close()
        self.loaded = False

    def __len__(self):
        if self.stack_mode == "SBT":  # 按照固定时间间隔
            # 每个事件都可以作为起点（之后加一个事件数量的筛选），尾部空出2帧
            # events_num=self.events.shape[0]
            end_time = self.events[-1, 2] - 2 * self.SBT_time
            self.events_time = self.events[:, 2]
            end_event_index = np.searchsorted(self.events_time, end_time, side='left')
            # return (self.num_images - self.start_frame - self.max_skip_frames - 1)
            return end_event_index
        else:  # 不是SBT就是SBN（按照固定事件数量累积）
            # 选定一个事件作为起点，直接加上固定数量就可以
            events_num = self.events.shape[0]
            return (events_num - self.SBN_num - 10)
            # return (self.num_GT_frame - self.start_frame - self.max_skip_frames - 1)

    def random_dropout_events(self, events, dropout_ratio):
        # 对输入的n行二维数据，随机去除一定比例的行数
        # print("input=", np.shape(events)[0])
        if dropout_ratio == 0:
            return events
        dropout_num = int(dropout_ratio * np.shape(events)[0])
        full_index = list(range(np.shape(events)[0]))
        dropout_index = random.sample(full_index, dropout_num)
        remain_index = set(full_index) - set(dropout_index)  # 集合操作
        events_flow = events[list(remain_index), :]
        # print("outut=", np.shape(events_flow)[0])
        return events_flow

    # 按照给定的序号，取对应图像帧（序号+固定偏置+随机的窗口大小）窗口（前后序号）
    # def get_prev_next_inds(self, ind):
    #     pind = self.start_frame + ind
    #     if self.train:
    #         cind = self.start_frame + ind + 1 + int((self.max_skip_frames - 1) * np.random.rand())  # 原来是+1
    #     else:
    #         cind = pind + 1  # 测试的时候就固定取窗口长度为1
    #     return pind, cind

    # 从大图中随机裁切一定大小的小图（抖动，数据增强）像素位置索引
    def get_box(self):
        top_left = self.top_left
        if self.train:
            top = int(np.random.rand() * (self.raw_image_size[0] - 1 - self.image_size[0]))
            left = int(np.random.rand() * (self.raw_image_size[1] - 1 - self.image_size[1]))
            top_left = [top, left]
        bottom_right = [top_left[0] + self.image_size[0],
                        top_left[1] + self.image_size[1]]

        return top_left, bottom_right

    def get_engine_mask(self, bbox):
        top_left, bottom_right = bbox
        engine_mask = self.engine_mask[top_left[0]:bottom_right[0],
                      top_left[1]:bottom_right[1]]
        return engine_mask

    # 按照给定的图像序号和mask位置取出图片和图片对应的事件序号
    # ！！！注意GT帧间隔模式下，取出的图像不可用于训练！！！
    def get_image(self, ind, bbox):  # SBN/SBT中ind是events序号
        top_left, bottom_right = bbox
        img_index = np.searchsorted(self.img_ts, self.events_time[ind])
        image = self.img_frame[img_index][top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]

        # image = image.transpose((2, 0, 1))  # （256,256,1）到（1,256,256）
        image = image.astype(np.float32) / 255.  # 转换成0~1
        image_frame_event_index = self.img_event_index[img_index]

        # cv_show_prev_image = self.img_frame[ind].astype(np.float32)/255
        # # cv_show_prev_image=cv_show_prev_image.transpose((2, 0, 1)).astype(np.float32) / 255.
        # # cv_show_prev_image-=0.5
        # # cv_show_prev_image *= 2.
        # cv2.imshow("prev_image", cv_show_prev_image)
        # cv2.waitKey(0)
        # cv2.imwrite("prev_image"+str(ind)+".bmp", cv_show_prev_image)
        return image, image_frame_event_index

    # 取出连续帧的光流（并且计算相应的GT）
    ##注意，根据时间跨度要进行GT的传播计算,这个是之前错误的
    def get_flow(self, start_ind, end_ind, bbox):
        top_left, bottom_right = bbox
        frame_num = end_ind - start_ind
        for i in range(frame_num):
            if i == 0:
                opt_flow = self.gt_flow_frame[start_ind][:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            else:
                opt_flow += self.gt_flow_frame[start_ind + i][:, top_left[0]:bottom_right[0],
                            top_left[1]:bottom_right[1]]
        return opt_flow

    def prop_flow(self, x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
        start_time = time.time()
        flow_x_interp = cv2.remap(x_flow,
                                  x_indices,
                                  y_indices,
                                  cv2.INTER_NEAREST)
        # 重采样操作
        # borderValue – Value used in case of a constant border. By default, it is 0.
        flow_y_interp = cv2.remap(y_flow,
                                  x_indices,
                                  y_indices,
                                  cv2.INTER_NEAREST)

        # test=flow_x_interp == 0
        x_mask[flow_x_interp == 0] = False
        y_mask[flow_y_interp == 0] = False

        x_indices += flow_x_interp * scale_factor
        y_indices += flow_y_interp * scale_factor
        # 上面2行注释掉，为了测试后面的代码
        """
        flow = torch.from_numpy(np.stack((x_flow, y_flow), axis=0))
        flow = torch.unsqueeze(flow, dim=0)
        index = torch.from_numpy(np.stack((y_indices, x_indices), axis=0))
        index = torch.unsqueeze(index, dim=0)
        h = x_flow.shape[0]
        w = x_flow.shape[1]
        index[:, 0] = (index[:, 0] / h)*2-1
        index[:, 1] = (index[:, 1] / w)*2-1
        index=index.permute(0,2,3,1)#(b,2,y,x)到(b,h,w,2),x是w（index 本来就是260*346）
        flow=flow.float()
        flow=flow.permute(0,1,3,2)#(b,2,x,y)到(b,2,h,w)

        flow_interp = F.grid_sample(flow, index, mode='bilinear',align_corners=True, padding_mode='zeros')
        flow_interp=flow_interp.squeeze().numpy()
        flow_x_interp = flow_interp[0]
        flow_y_interp = flow_interp[1]
        
        x_indices += flow_x_interp * scale_factor
        y_indices += flow_y_interp * scale_factor
        """
        # elapsed = (time.time() - start_time)
        # print(" prop:", elapsed)
        return

    def estimate_corresponding_gt_flow(self, start_index, end_index, bbox):
        gt_timestamps = self.gt_flow_ts[:]
        top_left, bottom_right = bbox

        event_frame_start_time = self.events_time[start_index]
        event_frame_end_time = self.events_time[end_index]
        gt_start_frame_index = np.searchsorted(gt_timestamps, event_frame_start_time,
                                               side='left') - 1  # 找到图像帧最近的GT帧索引左侧
        gt_end_frame_index = np.searchsorted(gt_timestamps, event_frame_end_time, side='right')
        gt_begin_time = gt_timestamps[gt_start_frame_index]
        gt_next_time = gt_timestamps[gt_start_frame_index + 1]
        gt_end_time = gt_timestamps[gt_end_frame_index]

        gt_dt = gt_next_time - gt_begin_time  # 计算GT帧之间的时间差
        event_dt = event_frame_end_time - event_frame_start_time  # 计算两帧图片的时间差（对比实验用图片的一帧时间间隔）

        x_flow = np.squeeze(self.gt_flow_frame[gt_start_frame_index, 0, :, :])
        y_flow = np.squeeze(self.gt_flow_frame[gt_start_frame_index, 1, :, :])

        # flow2image(torch.from_numpy(self.gt_flow_frame[gt_start_frame_index]))
        # cv2.imshow("img_frame", self.img_frame[start_index])
        # cv2.waitKey(0)

        # No need to propagate if the desired event_dt is shorter than the time between gt timestamps.
        if gt_next_time > event_frame_end_time:  # 所需的事件之间时间差在GT帧1帧范围内
            gt_flow_full = np.stack((x_flow * event_dt / gt_dt, y_flow * event_dt / gt_dt),
                                    axis=0)  # 假设时间够短，均匀，直接按比例截取时间
            cliped_gt_flow = gt_flow_full[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            return cliped_gt_flow, np.ones(cliped_gt_flow.shape, dtype=bool)  #

        # 否则要进行光流GT的传播
        x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]),
                                           np.arange(x_flow.shape[0]))
        x_indices = x_indices.astype(np.float32)
        y_indices = y_indices.astype(np.float32)

        orig_x_indices = np.copy(x_indices)
        orig_y_indices = np.copy(y_indices)

        # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
        x_mask = np.ones(x_indices.shape, dtype=bool)
        y_mask = np.ones(y_indices.shape, dtype=bool)

        scale_factor = (gt_timestamps[gt_start_frame_index + 1] - event_frame_start_time) / gt_dt
        total_dt = gt_timestamps[gt_start_frame_index + 1] - event_frame_start_time  # 累积已计算的时间长度

        self.prop_flow(x_flow, y_flow,
                       x_indices, y_indices,
                       x_mask, y_mask,
                       scale_factor=scale_factor)
        gt_start_frame_index += 1  # 上面对第一帧GT内的时间进行了计算，之后要根据第二帧GT计算

        while gt_timestamps[gt_start_frame_index + 1] < event_frame_end_time:  # 如果第二帧GT的结尾还在图像帧结束之前，则继续传播
            x_flow = np.squeeze(self.gt_flow_frame[gt_start_frame_index, 0, :, :])
            y_flow = np.squeeze(self.gt_flow_frame[gt_start_frame_index, 1, :, :])

            self.prop_flow(x_flow, y_flow,
                           x_indices, y_indices,
                           x_mask, y_mask)
            total_dt += gt_timestamps[gt_start_frame_index + 1] - gt_timestamps[gt_start_frame_index]
            gt_start_frame_index += 1

        final_dt = event_frame_end_time - gt_timestamps[gt_start_frame_index]
        total_dt += final_dt

        final_gt_dt = gt_timestamps[gt_start_frame_index + 1] - gt_timestamps[gt_start_frame_index]

        x_flow = np.squeeze(self.gt_flow_frame[gt_start_frame_index, 0, :, :])
        y_flow = np.squeeze(self.gt_flow_frame[gt_start_frame_index, 1, :, :])

        scale_factor = final_dt / final_gt_dt

        self.prop_flow(x_flow, y_flow,
                       x_indices, y_indices,
                       x_mask, y_mask,
                       scale_factor)

        x_shift = x_indices - orig_x_indices
        y_shift = y_indices - orig_y_indices
        x_shift[~x_mask] = 0
        y_shift[~y_mask] = 0  # 超出边界范围的被标记的位置，设为0（是否合理？还是设成边界值）
        gt_flow_full = np.stack((x_shift, y_shift), axis=0)
        cliped_gt_flow = gt_flow_full[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        mask_full = np.stack((x_mask, y_mask), axis=0)
        cliped_mask = mask_full[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        assert total_dt == event_dt  # 确保传播过程覆盖了全部的时间范围

        return cliped_gt_flow, cliped_mask

    # 取出深度帧（深度帧是与GT同步的，使用的是GT的帧）
    def get_depth(self, start_ind, end_ind, bbox):  # 2021年5月24日加

        top_left, bottom_right = bbox

        start_time = self.events_time[start_ind]
        end_time = self.events_time[end_ind]
        mid_time = (start_time + end_time) / 2

        left_gt_index = np.searchsorted(self.gt_flow_ts[:], mid_time, side='right') - 1
        right_gt_index = np.searchsorted(self.gt_flow_ts[:], mid_time, side='left')
        left_gt_time = self.gt_flow_ts[left_gt_index]
        right_gt_time = self.gt_flow_ts[right_gt_index]
        # searchsorted 是搜索插入位置，np.searchsorted([1,2,3,4,5], 3, side='right')=3
        # 所以side的设置只有在相等的情况下才会有差异

        k = (mid_time - left_gt_time) / (right_gt_time - left_gt_time)
        left_depth_img = self.gt_depth_img[left_gt_index][top_left[0]: bottom_right[0],
                         top_left[1]: bottom_right[1]]
        right_depth_img = self.gt_depth_img[right_gt_index][top_left[0]: bottom_right[0],
                          top_left[1]: bottom_right[1]]
        depth_img = (1 - k) * left_depth_img + k * right_depth_img

        depth_frame = torch.from_numpy(depth_img).float()
        depth_mask = torch.isnan(depth_frame)
        depth_frame[depth_mask] = 0.001
        depth_max = torch.max(depth_frame)
        depth_frame[0:64][depth_mask[0:64]] = depth_max + 1

        # depth_frame = depth_frame / depth_max
        # show_depth = depth_frame.numpy()
        # cv2.imshow("depth_frame", show_depth)
        # cv2.waitKey(0)

        depth_mask = torch.logical_not(depth_mask)  # depth_mask true是无效值，取反后true就有效值
        return depth_frame, depth_mask  # depth_mask 1是有效值，可直接相乘mask

    # 计算连续的几帧内对应了多少事件
    def count_events(self, pind, cind, frame_interval):
        assert frame_interval == "img" or frame_interval == "gt"
        if frame_interval == "gt":
            return self.gtflow_event_index[cind] - self.gtflow_event_index[pind]
        else:
            return self.img_event_index[cind] - self.img_event_index[pind]

    # 按照给的图像序号和mask位置，取出对应位置和跨越这几帧的事件
    def get_events(self, pind, cind, bbox):
        # 原来是按照固定间隔取，取完再截取，SBN模式下严格的说要改
        top_left, bottom_right = bbox
        peind = pind
        ceind = cind

        events = self.events[peind:ceind, :]
        mask = np.logical_and(np.logical_and(events[:, 1] >= top_left[0],
                                             events[:, 1] < bottom_right[0]),
                              np.logical_and(events[:, 0] >= top_left[1],
                                             events[:, 0] < bottom_right[1]))  # 获取位置范围内的事件
        # events原始是w*h*t*p
        events_masked = events[mask]
        # 对取出的事件xy坐标进行偏移校正
        events_shifted = events_masked
        events_shifted[:, 0] = events_masked[:, 0] - top_left[1]
        events_shifted[:, 1] = events_masked[:, 1] - top_left[0]

        events_shifted[:, 2] -= np.min(events_shifted[:, 2])  # 时间归一化

        # convolution expects 4xN
        # events_shifted = np.transpose(events_shifted).astype(np.float32)
        events_shifted = events_shifted.astype(np.float32)

        return events_shifted

    # 计算p、c范围内的事件数量
    def get_num_events(self, pind, cind, bbox, dataset):

        peind = self.image_to_event[dataset][pind]
        ceind = self.image_to_event[dataset][cind]

        events = self.events[dataset][peind:ceind, :]
        return events.shape[0]

    # 对事件累积图像的张量进行归一化（均值0，方差1）,加紧到0.98，丢弃前后2%的数值
    def normalize_event_volume(self, event_volume):
        event_volume_flat = event_volume.view(-1)  # 展成一维
        nonzero = torch.nonzero(event_volume_flat)  # 找出非零索引
        nonzero_values = event_volume_flat[nonzero]  # 取出非零
        if nonzero_values.shape[0]:
            lower = torch.kthvalue(nonzero_values,
                                   max(int(0.02 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            upper = torch.kthvalue(nonzero_values,
                                   max(int(0.98 * nonzero_values.shape[0]), 1),
                                   dim=0)[0][0]
            max_val = max(abs(lower), upper)
            event_volume = torch.clamp(event_volume, -max_val, max_val)
            event_volume /= max_val
        return event_volume

    # 后面两个函数是对图像的亮度进行伽马变换（指数非线性变换）
    def apply_illum_augmentation(self, image,
                                 gain_min=0.8, gain_max=1.2, gamma_min=0.8, gamma_max=1.2):
        random_gamma = gamma_min + random.random() * (gamma_max - gamma_min)
        random_gain = gain_min + random.random() * (gain_max - gain_min);
        image = self.transform_gamma_gain_np(image, random_gamma, random_gain)
        return image

    def transform_gamma_gain_np(self, image, gamma, gain):
        # apply gamma change and image gain.
        image = (1. + image) / 2.
        image = gain * np.power(image, gamma)
        image = (image - 0.5) * 2.
        return np.clip(image, -1., 1.)

    # 对事件进行按照出现次数累加的投影统计
    def event_counting_images(self, events, vol_size):
        event_count_images = events.new_zeros(vol_size)

        x = events[:, 0].long()
        y = events[:, 1].long()
        t = events[:, 2]
        p = events[:, 3].long()
        vol_mul = torch.where(p < 0,  # 正在前，负在后，负事件需要增加1*W*H的序号
                              torch.ones(p.shape, dtype=torch.long),
                              torch.zeros(p.shape, dtype=torch.long))
        # ind为位置索引
        inds = (vol_size[1] * vol_size[2]) * vol_mul \
               + (vol_size[2]) * y \
               + x
        vals = torch.ones(p.shape, dtype=torch.float)  # 对应的位置累加1
        event_count_images.view(-1).put_(inds, vals, accumulate=True)  ##对应的位置累加1

        img = event_count_images.numpy()
        # cv2.imshow("img_event", img[0])
        # cv2.waitKey(0)
        # cv2.imshow("img_event", img[1])
        # cv2.waitKey(0)
        return event_count_images

    # 按照最近（新）事件的时间戳生成事件的时间戳图像
    def event_timing_images(self, events, vol_size):
        event_time_images = events.new_zeros(vol_size)

        x = events[:, 0].long()
        y = events[:, 1].long()
        t = events[:, 2]
        p = events[:, 3].long()
        t_min = t.min()
        t_max = t.max()

        vol_mul = torch.where(p < 0,  # 正在前，负在后，负事件需要增加1*W*H的序号
                              torch.ones(p.shape, dtype=torch.long),
                              torch.zeros(p.shape, dtype=torch.long))
        # ind为位置索引
        inds = (vol_size[1] * vol_size[2]) * vol_mul \
               + (vol_size[2]) * y \
               + x
        vals = t / (t_max - t_min)
        event_time_images.view(-1).put_(inds, vals, accumulate=False)  ##对应的位置更新为对应的时间戳，覆盖

        img = event_time_images.numpy()
        # cv2.imshow("img_event", img[0])
        # cv2.waitKey(0)
        # cv2.imshow("img_event", img[1])
        # cv2.waitKey(0)
        return event_time_images

    def stacking_events(self, events, vol_size):
        # 这里是直接按照极性叠加
        events = torch.from_numpy(events)
        Stacking_num = vol_size[0]
        vol_size[0] = vol_size[0] * 2  # 正负分开存储再合并

        event_stacking_images_temp = events.new_zeros(vol_size)

        x = events[:, 0].long()
        y = events[:, 1].long()
        t = events[:, 2]
        p = events[:, 3].long()

        t_min = t.min()
        t_max = t.max()
        t_index = ((t - t_min) / (t_max - t_min)) // (1 / Stacking_num + 1e-7)  # 时间归一化0~1，之后//片数
        vol_mul = torch.where(p < 0,  # 负在前，正在后，负事件需要增加1*W*H的序号
                              torch.ones(p.shape, dtype=torch.long),
                              torch.zeros(p.shape, dtype=torch.long))
        # ind为位置索引,
        inds = (vol_size[1] * vol_size[2]) * vol_mul * Stacking_num \
               + (vol_size[1] * vol_size[2]) * t_index + (vol_size[2]) * y \
               + x
        inds = inds.long()
        vals = torch.ones(p.shape, dtype=torch.float)  # 对应的位置累加1
        event_stacking_images_temp.view(-1).put_(inds, vals, accumulate=True)  ##对应的位置累加1
        event_stacking_images = -1 * event_stacking_images_temp[0:Stacking_num] \
                                + event_stacking_images_temp[Stacking_num:]

        imgmax = torch.max(event_stacking_images)
        imgmin = torch.min(event_stacking_images)
        imgmax = abs(max(imgmax, imgmin))
        img = (event_stacking_images / imgmax + 1) / 2  # 为了显示，归一化到【0,1】

        event_stacking_images = event_stacking_images / imgmax  # 归一化【-1，1】

        # img=img.numpy()
        # cv2.imshow("img_event", img[0])
        # cv2.waitKey(0)
        # cv2.imshow("img_event", img[1])
        # cv2.waitKey(0)

        return event_stacking_images

    # 取数据的主函数
    def get_single_item(self, ind):
        start_time = time.time()
        # frame_start_event_index, frame_end_event_index = self.get_prev_next_inds(ind)  # 按照给定的序号，取对应图像帧前后序号
        if self.train:
            if self.stack_mode == "SBT":  # 按照固定时间间隔取
                # 保证等时间下取出的事件点大于一定数量
                n_events = -1
                n_iters = 0  # 测试迭代次数
                while n_events < self.min_n_events:
                    frame_start_time = self.events[ind, 2]  # 按照给定的事件序号，取对应事件的时间
                    frame_end_time = frame_start_time + self.SBT_time
                    frame_end_event_index = np.searchsorted(self.events_time, frame_end_time, side='left')
                    n_events = frame_end_event_index - ind  # self.count_events(frame_start_event_index, frame_end_event_index,self.frame_interval)  # 计算当前帧范围内事件总数

                    n_iters += 1
                    if n_events < self.min_n_events:  # or ind%5==0:  # 如果取到的帧事件数量太小，则跳过，随机再取一个帧
                        ind = random.randint(0, self.__len__())
            else:  # "SBN":#按照固定事件数量取
                frame_interval = 999 * self.SBT_time
                while frame_interval > self.max_skip_frames * self.SBT_time:
                    frame_end_event_index = ind + self.SBN_num
                    frame_start_time = self.events_time[ind]
                    frame_end_time = self.events_time[frame_end_event_index]
                    frame_interval = frame_end_time - frame_start_time
                    if frame_interval > self.max_skip_frames * self.SBT_time:  # 如果间隔太大，证明这段事件太稀疏
                        ind = random.randint(0, self.__len__())
                # 如果固定的数量覆盖的时间太长就舍弃

        else:  # 测试(代码同训练，后期可用于增加数据增强内容)
            if self.stack_mode == "SBT":  # 按照固定时间间隔取
                # 保证等时间下取出的事件点大于一定数量
                n_events = -1
                n_iters = 0  # 测试迭代次数
                while n_events < self.min_n_events:
                    frame_start_time = self.events[ind, 2]  # 按照给定的事件序号，取对应事件的时间
                    frame_end_time = frame_start_time + self.SBT_time
                    frame_end_event_index = np.searchsorted(self.events_time, frame_end_time, side='left')
                    n_events = frame_end_event_index - ind  # self.count_events(frame_start_event_index, frame_end_event_index,self.frame_interval)  # 计算当前帧范围内事件总数
                    n_iters += 1
                    if n_events < self.min_n_events:  # or ind%5==0:  # 如果取到的帧事件数量太小，则跳过，随机再取一个帧
                        ind = random.randint(0, self.__len__())
            else:  # "SBN":#按照固定事件数量取
                frame_interval = 999 * self.SBT_time
                while frame_interval > self.max_skip_frames * self.SBT_time:
                    frame_end_event_index = ind + self.SBN_num
                    frame_start_time = self.events_time[ind]
                    frame_end_time = self.events_time[frame_end_event_index]
                    frame_interval = frame_end_time - frame_start_time
                    if frame_interval > self.max_skip_frames * self.SBT_time:  # 如果间隔太大，证明这段事件太稀疏
                        ind = random.randint(0, self.__len__())

        bbox = self.get_box()  # 获取小图片的位置索引
        engine_mask = self.get_engine_mask(bbox)  # 增加对引擎盖区域的mask
        # 因为引擎盖反光，事件是错误的，因此需要去掉，其他位置的mask结果，不监督就好
        frame_start_event_index = ind
        # 按照序号和像素截取位置索引取出（图片+对应事件序号），仅用于debug可视化，不用于监督

        # prev_image, prev_image_frame_event_index = self.get_image(frame_start_event_index, bbox)
        # next_image, next_image_frame_event_index = self.get_image(frame_end_event_index, bbox)
        frame_mid_time = (self.events_time[frame_start_event_index] + self.events_time[frame_end_event_index]) / 2
        frame_mid_event_index = np.searchsorted(self.events_time, frame_mid_time)
        mid_image, mid_image_frame_event_index = self.get_image(frame_mid_event_index, bbox)

        # start_time = time.time()
        flow_frame, boundary_mask = self.estimate_corresponding_gt_flow(frame_start_event_index, frame_end_event_index,
                                                                        bbox)
        # elapsed = (time.time() - start_time)
        # print("GT用时:", elapsed)
        boundary_mask = np.logical_and(boundary_mask[0], boundary_mask[1])
        boundary_mask = np.uint8(boundary_mask)
        # 经验证此处boundary_mask多余了，因为mask掉的已经把光流改成了0，后面对0mask了

        # uv_equal=boundary_mask[0]==boundary_mask[1]
        # show_mask=torch.from_numpy((boundary_mask))
        # show_mask=show_mask.float()
        # flow2image(torch.from_numpy(flow_frame))
        # flow2image(show_mask)

        depth_frame, depth_mask = self.get_depth(frame_start_event_index, frame_end_event_index, bbox, )
        # test=depth_frame
        # test[np.isnan(test)]=0
        # max=np.max(test)
        # test=test/max
        # cv2.imshow("depth_frame", test)
        # cv2.waitKey(0)

        events_flow = self.get_events(frame_start_event_index, frame_end_event_index, bbox)  # 取出对应位置和跨越这几帧的事件

        if self.train:
            # 此模式下不太需要这个增强，SBN可能不应进行这个增强
            events_flow = self.random_dropout_events(events_flow, self.dropout_ratio)

        event_volume = gen_discretized_event_volume(torch.from_numpy(events_flow).cpu(),
                                                    [self.n_time_bins * 2,  # n_time_bins：体素的片数（正负分开处理）
                                                     self.image_size[0],  # W，H
                                                     self.image_size[1]])

        event_stacking_images = self.stacking_events(events_flow,
                                                     [self.Stacking_num,  # n_time_bins：体素的片数（正负累加处理）
                                                      self.image_size[0],  # W，H
                                                      self.image_size[1]])

        # 上面进行事件投影，获得事件正负独立，体素化离散投影的结果
        # 注意：1~N是正事件，N+1~2N是负事件体素。所以1和N+1是一组，N与2N是一组
        # 计算事件的数量累加图像
        # e_img=event2image(torch.from_numpy(events_frame).cpu())#可视化
        event_count_images = self.event_counting_images(torch.from_numpy(events_flow).cpu(),
                                                        [2,  # n_time_bins：体素的片数（正负分开处理）
                                                         self.image_size[0],  # W，H
                                                         self.image_size[1]])

        event_time_images = self.event_timing_images(torch.from_numpy(events_flow).cpu(),
                                                     [2,  # n_time_bins：体素的片数（正负分开处理）
                                                      self.image_size[0],  # W，H
                                                      self.image_size[1]])

        if self.normalize_events:
            event_volume = self.normalize_event_volume(event_volume)
            # 对事件累积图像的张量进行归一化（均值0，方差1）,加紧到0.98，丢弃前后2%的数值

        # prev_image_gt, next_image_gt = prev_image, next_image
        mid_image_gt = mid_image

        # f_img = flow2image(torch.from_numpy(flow_frame))  # 可视化光流图

        event_volume = event_volume * engine_mask  # 引擎盖mask
        event_count_images = event_count_images * engine_mask
        event_time_images = event_time_images * engine_mask
        # prev_image=prev_image* engine_mask
        mid_image = mid_image * engine_mask
        event_stacking_images = event_stacking_images * engine_mask
        # next_image=next_image* engine_mask

        # 计算位置掩码（出现事件的位置）
        WIDTH = self.image_size[1]  # 阵面宽
        HEIGHT = self.image_size[0]  # 阵面高
        events_num = events_flow.shape[0]

        # filled_events_flow = np.zeros((self.Longest_event, 4))
        # filled_events_flow[0:events_num] = events_flow#填充到等长

        # events_frame = event2image(torch.from_numpy(events_frame))

        event_mask = torch.zeros(HEIGHT, WIDTH, dtype=torch.int8)
        fill = torch.ones(events_num, dtype=torch.int8)
        index = (torch.from_numpy(events_flow[:, 1]).long(), torch.from_numpy(events_flow[:, 0]).long())
        event_mask = event_mask.index_put(index, fill)


        event_mask = event_mask * engine_mask  # 增加对引擎盖区域的mask

        # show_mask = event_mask.float().numpy()
        # cv2.imshow("event_mask", show_mask)
        # cv2.waitKey(0)

        # 训练则进行数据增强
        # 明天测试深度图的翻转，增加一个新的投影图用于cGAN
        if self.train:
            # 按照一定的概率翻转图像（事件叠加图，以及对应GT图左右或者上下翻转），数据增强
            # 这里是对图像进行增强，光流是否能翻转
            if np.random.rand() < self.flip_x:
                event_mask = torch.flip(event_mask, dims=[1])
                event_volume = torch.flip(event_volume, dims=[2])
                # prev_image = np.flip(prev_image, axis=2)
                mid_image = np.flip(mid_image, axis=2)
                # next_image = np.flip(next_image, axis=2)
                depth_frame = torch.flip(depth_frame, dims=[1])
                depth_mask = torch.flip(event_mask, dims=[1])

                flow_frame = np.flip(flow_frame, axis=2)  # 对光流的翻转可能存在问题，要试试看
                flow_frame[0] = -flow_frame[0]  # 翻转后光流左右相反，数值取反

                event_count_images = torch.flip(event_count_images, dims=[2])
                event_time_images = torch.flip(event_time_images, dims=[2])
                event_stacking_images = torch.flip(event_stacking_images, dims=[2])
                # f_img = flow2image(flow_frame)
                # depth_frame = np.flip(depth_frame, axis=1)
            if np.random.rand() < self.flip_y:
                event_mask = torch.flip(event_mask, dims=[0])
                event_volume = torch.flip(event_volume, dims=[1])
                # prev_image = np.flip(prev_image, axis=1)
                mid_image = np.flip(mid_image, axis=1)
                # next_image = np.flip(next_image, axis=1)
                depth_frame = torch.flip(depth_frame, dims=[0])
                depth_mask = torch.flip(event_mask, dims=[0])

                flow_frame = np.flip(flow_frame, axis=1)
                flow_frame[1] = -flow_frame[1]  # 翻转后光流上下相反，数值取反

                event_count_images = torch.flip(event_count_images, dims=[1])
                event_time_images = torch.flip(event_time_images, dims=[1])
                event_stacking_images = torch.flip(event_stacking_images, dims=[1])
            # prev_image_gt, next_image_gt = prev_image, next_image
            if self.appearance_augmentation:
                # prev_imag = self.apply_illum_augmentation(prev_image)
                mid_image = self.apply_illum_augmentation(mid_image)
                # next_image = self.apply_illum_augmentation(next_image)

        """下面是计算mask，有事件的位置为true"""
        # test=depth_frame
        # test[np.isnan(test)]=0
        # max=np.max(test)
        # test=test/max
        # cv2.imshow("prev_image", prev_image[0])
        # cv2.waitKey(0)
        # cv2.imshow("depth_frame", test[0])
        # cv2.waitKey(0)
        # f_img = flow2image(torch.from_numpy(flow_frame.copy()))  # 可视化光流图

        # test=outer_mask.nonzero()
        # cv2.imshow("outer_mask", outer_mask.numpy())
        # cv2.waitKey(0)
        # start_time = time.time()

        events_frame = event2image(torch.from_numpy(events_flow), self.image_size)
        events_frame = torch.from_numpy(events_frame)
        # events_frame=torch.unsqueeze(events_frame,dim=0)
        # events_frame=events_frame
        # 计算mask后的GT光流
        # opt_flow_frame = torch.from_numpy(opt_flow_frame)
        # pointgt_flow=opt_flow_frame.mul(torch.stack([outer_mask[0]],dim=0))
        # Event volume is t-y-x（T*H*W）
        # f_img = flow2image(flow_frame)
        # mask_img=outer_mask.numpy()*100
        # cv2.imshow("outer_mask", mask_img)
        # cv2.waitKey(0)

        # cv2.imshow("mid_image", mid_image)
        # cv2.waitKey(0)
        flow_mask_u = np.zeros_like(flow_frame[0])
        flow_mask_u[flow_frame[0].nonzero()] = 1
        flow_mask_v = np.zeros_like(flow_frame[0])
        flow_mask_v[flow_frame[1].nonzero()] = 1
        flow_mask = np.logical_or(flow_mask_u, flow_mask_v)
        flow_mask = np.uint8(flow_mask)
        # test = sum(sum(boundary_mask == flow_mask))  # 验证两个mask是否一致

        # flow_mask=np.float32(flow_mask)
        # cv2.imshow("flow_mask", flow_mask)
        # cv2.waitKey(0)
        # flow_mask = torch.from_numpy(flow_mask)
        # end_time = time.time()

        # print(end_time - start_time)

        # cv_show_prev_image = prev_image_gt[0]#prev_image
        # cv2.imshow("prev_image", cv_show_prev_image)
        # cv2.waitKey(0)

        # prev_image=torch.from_numpy(prev_image).float()
        # prev_image_gt = torch.from_numpy(prev_image_gt).float()

        gradient_x = cv2.Sobel(mid_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(mid_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.stack([gradient_x, gradient_y])

        mid_image = torch.from_numpy(mid_image).float()
        mid_image_gt = torch.from_numpy(mid_image_gt).float()
        # next_image = torch.from_numpy(next_image).float()
        # next_image_gt = torch.from_numpy(next_image_gt).float()
        flow_frame = torch.from_numpy(flow_frame).float()
        # depth_frame = torch.from_numpy(depth_frame).float()
        gradient = torch.from_numpy(gradient).float()
        gradient=torch.squeeze(gradient)

        output = {  # "prev_image": prev_image,  # 起始帧图像（经过裁剪，翻转等增强）
            # "prev_image_gt": prev_image_gt,  # 还是起始帧图像（未经过最后的亮度增强伽马变换等）
            # "prev_image_frame_event_index": prev_image_frame_event_index,  # 对应的起始事件序号
            # "next_image": next_image,  # 末尾帧图像（数据增强，）
            # "next_image_gt": next_image_gt,# 末尾帧图像（未经过最后的亮度增强伽马变换等）
            "mid_image_gt": mid_image_gt,
            "mid_image": mid_image,
            # "next_image_frame_event_index": next_image_frame_event_index,  # 对应的末尾事件序号
            "event_volume": event_volume,  # 投影得到的事件体素（T*H*W）
            "flow_frame": flow_frame,  # 光流GT值
            "event_mask": event_mask,  # 事件出现位置的掩码，后续算loss可能用得到(有事件的为1)
            "event_count_images": event_count_images,  # 根据各个像素位置累积的事件出现次数累积图像，前正后负
            "event_time_images": event_time_images,  # 按照最近（新）事件的时间戳生成的事件时间戳图像，前正后负
            "events_frame": events_frame,  # 事件序列可视化图像
            "events_num": events_num,  # 返回的事件序列有效长度(!!!!没有减去mask掉的位置)
            # "events_flow": filled_events_flow,  # 返回的原始时间序列，有效长度为events_num，其余位置填充0（774行）
            "flow_mask": flow_mask,  # 返回光流的mask，如果光流uv都为0则返回0，其余位置是1
            "depth_frame": depth_frame,  # 增加深度帧
            "depth_mask": depth_mask,  # 深度图的mask，已经过取反，true即有深度的位置，直接相乘即可
            "image_gradient": gradient,
            "event_stacking_images": event_stacking_images  # 堆叠的图
        }

        # 注意event_volume：1~N是正事件，N+1~2N是负事件体素。所以1和N+1是一组，N与2N是一组
        # elapsed = (time.time() - start_time)
        # print("取数据-Time used:", elapsed)
        return output

    def __getitem__(self, frame_index):
        start = time.process_time()
        if not self.loaded:
            self.load()

        # elapsed = (time.process_time() - start)
        # print("取数据-Time used:", elapsed)
        # print(" __getitem__取了", frame_index)
        return self.get_single_item(frame_index)
        # random_dropout_events(events, pointgt ,self.dropout_ratio)

        # 计算位置掩码（出现事件的位置）
        WIDTH = 346  # 阵面宽
        HEIGHT = 260  # 阵面高
        outer_mask = torch.zeros(2, HEIGHT, WIDTH, dtype=torch.int8)
        events_num = torch.zeros(2)
        events_num[0] = p_num
        events_num[1] = n_num  # 正、负事件对应的个数
        events_num = events_num.type(torch.int32)
        for pn_index in range(2):  # 对正负事件单独计算mask
            for i in range(events_num[pn_index]):
                outer_mask[pn_index][int(events_flow[pn_index, i, 1])][int(events_flow[pn_index, i, 0])] = 1
                pass
        # 之后计算mask后的GT光流
        # pointgt_flow_temp=point_gt[begin_index:end_index]
        # pointgt_flow = torch.zeros((2,2,HEIGHT, WIDTH))
        opt_flow_frame = torch.from_numpy(opt_flow_frame)
        # pointgt_flow[0]=opt_flow_frame.mul(torch.stack([outer_mask[0],outer_mask[0]],dim=0))
        # pointgt_flow[1] = opt_flow_frame.mul(torch.stack([outer_mask[1], outer_mask[1]], dim=0))


if __name__ == '__main__':
    data = MyData(12, train=True, engine_mask=False, frame_interval="img")
    t1 = data.__getitem__(6377)
    t2 = data.__getitem__(1865)
    # t3 = data.__getitem__(9607)#3166 1251
    # print(data.__len__())
    # print(data.__getitem__(5))
    # data = MyData(-1)

    from torch.utils.data import DataLoader

    loader = DataLoader(data, 1, drop_last=True)
    for i, batch in enumerate(loader):
        print(i, batch['event_count_images'])
        break

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

# Longest_event = 75600


davis_list = ["./indoor_flying1_500_600.h5 ",  # /home/xhlinxm/MVSEC/indoor_flying/indoor_flying1_data.hdf5
              "./indoor_flying1_events_500_600_pointGT.h5",
              "/media/MVSEV/indoor_flying1_full.h5",
              "/media/MVSEV/indoor_flying2_full.h5",
              "/media/MVSEV/indoor_flying3_full.h5",
              "/media/MVSEV/indoor_flying4_full.h5",
              "/media/MVSEV/outdoor_day1_full.h5",
              "/media/MVSEV/outdoor_day2_full.h5",  # /home/lxh/MVSEC/
              "D://train_data",  # "/home/xhlinxm/MVSEC/train_data",#D://train_data;"/home/xhlinxm/MVSEC/test_data"
              "D://test_data",
              "D:\indoor_flying1_full.h5",#10
              "D:\indoor_flying2_full.h5",#11
              "D:\outdoor_day1_full.h5",#12
              "D:\outdoor_day2_full.h5",#13
              "D:\indoor_flying2_data_pointGT.h5",
              "E:\MVSEC HDF5\indoor_flying\indoor_flying2_data.hdf5",
              '../indoor_flying1_full_400_450.h5'
              ]


class MyData(Dataset):  # 这个类用于准备数据（取出所有数据）
    def __init__(self, file_index, train=True, start_time=-1, sample=None):
        # (self, num_width, num_height, file_index, events_frame_time=1, dropout_ratio=0.01, insert_ratio=0,train=True)
        '''
        :param num_width: 宽方向的patch切分数量
        :param num_height: 高方向的patch切分数量
        :param file_index: 文件列表索引值
        :param dropout_ratio: 随机丢弃比例
        '''
        # self.folder_address=davis_list[file_index]#文件夹地址
        # self.full_list = os.listdir(self.folder_address)#文件夹内的文件列表
        # self.frame_num = np.shape(self.full_list)[0]#文件数量（列表长度）
        self.file_path = davis_list[file_index]  # 文件路径
        self.dropout_ratio = 0.02  # 一定的概率丢弃事件
        self.appearance_augmentation = True  # 对投影生成的体素图像进行视觉增强（伽马变换）
        self.normalize_events = True  # 规范化
        self.train = train
        self.top_left = [2, 13]  # Top left corner of crop - (d1,d2).
        self.image_size = [256, 256]  # "Final image size for predictions (HxW).
        self.n_time_bins = 3  # 要离散化的片数
        # if start_time == -1:
        #     self.start_time = 5#"Time to start reading from the dataset (s)."
        # else:
        #     self.start_time = start_time
        self.max_skip_frames = 2  # max_skip_frames最大可连续取的窗口大小
        self.flip_x = 0.1  # "Probability of flipping the volume in x.",
        self.flip_y = 0.1  # "Probability of flipping the volume in y.",
        self.Longest_event = 195600  # 输出原始事件序列时的最大值
        self.sample = sample
        self.SBT_num=3#Stacking Based on Time 划分帧的个数，在图像帧之间划分成几帧
        if train == True:
            super(MyData, self).__init__()
            print("训练使用数据集：", davis_list[file_index])
        else:
            print("测试使用数据集：", davis_list[file_index])
            self.flip_x = 0.5  # "Probability of flipping the volume in x.",
            self.flip_y = 0.5
        self.load(True)
        self.close()

    def load(self, only_length=False):
        self.full_file = h5py.File(self.file_path, 'r')
        self.events = self.full_file['davis']['left']['events']
        self.gt_event_index = self.full_file['davis']['left']['gt_event_index']
        self.img_frame = self.full_file['davis']['left']['img_frame']
        self.gt_flow_frame = self.full_file['davis']['left']['gt_frame']
        self.gt_depth = self.full_file['davis']['left']['depth_img']
        self.gt_odometry = self.full_file['davis']['left']['odometry']
        self.gt_pose = self.full_file['davis']['left']['pose']
        self.pose_event_index = self.full_file['davis']['left']['pose_event_index']

        self.raw_image_size = self.img_frame.shape[1:]
        self.num_images = self.img_frame.shape[0]
        self.start_frame = 0  # 起始帧，全局的偏移量
        self.loaded = True

    def close(self):
        self.events = None
        self.image_to_event = None
        self.gt_event_index = None
        self.img_frame = None
        self.gt_flow_frame = None
        self.gt_depth = None
        self.gt_odometry = None
        self.gt_pose = None
        self.pose_event_index = None

        self.full_file.close()
        self.loaded = False

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

    def get_prev_next_inds(self, ind):
        pind = self.start_frame + ind
        if self.train:
            cind = self.start_frame + ind + 0 + int((self.max_skip_frames - 1) * np.random.rand())  # 原来是+1
        else:
            cind = pind + 0  # 测试的时候就固定取窗口长度为1
        return pind, cind

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

    # 按照给定的图像序号和mask位置取出图片和图片对应的事件序号
    def get_image(self, ind, bbox):
        top_left, bottom_right = bbox
        image = self.img_frame[ind][top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1], None]

        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.
        image -= 0.5
        image *= 2.

        image_ts = self.gt_event_index[ind]
        return image, image_ts

    # 取出连续帧的光流（累加）
    def get_flow(self, start_ind, end_ind, bbox):
        top_left, bottom_right = bbox
        frame_num = end_ind - start_ind + 1
        for i in range(frame_num):
            if i == 0:
                opt_flow = self.gt_flow_frame[start_ind][:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            else:
                opt_flow += self.gt_flow_frame[start_ind + i][:, top_left[0]:bottom_right[0],
                            top_left[1]:bottom_right[1]]

        return opt_flow

    # 计算连续的几帧内对应了多少事件
    def count_events(self, pind, cind):
        return self.gt_event_index[cind, 1] - self.gt_event_index[pind, 0]

    # 按照给的图像序号和mask位置，取出对应位置和跨越这几帧的事件
    def get_events(self, pind, cind, bbox):
        top_left, bottom_right = bbox
        # peind = max(self.image_to_event[pind], 0)
        peind = int(self.gt_event_index[pind, 0])
        ceind = int(self.gt_event_index[cind, 1])

        events = self.events[peind:ceind, :]
        mask = np.logical_and(np.logical_and(events[:, 1] >= top_left[0],
                                             events[:, 1] < bottom_right[0]),
                              np.logical_and(events[:, 0] >= top_left[1],
                                             events[:, 0] < bottom_right[1]))  # 获取位置范围内的事件

        events_masked = events[mask]
        # 对取出的事件xy坐标进行偏移校正
        events_shifted = events_masked
        events_shifted[:, 0] = events_masked[:, 0] - top_left[1]
        events_shifted[:, 1] = events_masked[:, 1] - top_left[0]

        # subtract out min to get delta time instead of absolute
        # print(events_shifted.shape,pind,cind)
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

    ##取出的事件数量、前后索引、取出的像素位置索引、
    def select_valid_input(self, ind_in):

        for i in reversed(range(len(self.cum_num_images))):
            if ind_in >= self.cum_num_images[i]:
                dataset = i
                ind = ind_in - self.cum_num_images[i]
                break

        pind, cind = self.get_prev_next_inds(ind, dataset)  # 得到窗口的前后序号索引
        bbox = self.get_box(dataset)  # 从大图中随机裁切一定大小的小图（抖动，数据增强）像素位置索引
        return self.get_num_events(pind, cind, bbox, dataset), \
               cind, pind, bbox, dataset  # 取出的事件数量、前后索引、取出的像素位置索引、

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
    def apply_illum_augmentation(self, prev_image, next_image,
                                 gain_min=0.8, gain_max=1.2, gamma_min=0.8, gamma_max=1.2):
        random_gamma = gamma_min + random.random() * (gamma_max - gamma_min)
        random_gain = gain_min + random.random() * (gain_max - gain_min);
        prev_image = self.transform_gamma_gain_np(prev_image, random_gamma, random_gain)
        next_image = self.transform_gamma_gain_np(next_image, random_gamma, random_gain)
        return prev_image, next_image

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

    # 取数据的主函数
    def get_single_item(self, ind):
        # 保证取出的事件点大于一定数量
        start_time = time.time()
        if self.train:
            max_n_events = 400  # 50000 / 6
            n_events = -1
            n_iters = 0
            while n_events < max_n_events:
                n_events = self.count_events(ind, ind)  # (ind, ind + 1),改动了
                n_iters += 1
                if n_events < max_n_events:  # or ind%5==0:  # 如果取到的帧事件数量太小，则跳过，随机再取一个帧
                    ind = random.randint(0, self.__len__())
        else:  # 测试
            max_n_events = 400  # 50000 / 6
            n_events = -1
            n_iters = 0
            while n_events < max_n_events:
                n_events = self.count_events(ind, ind)  # (ind, ind + 1),改动了
                n_iters += 1
                if n_events < max_n_events:  # or ind % 5 != 0:  # 强制让测试集的序号为5的倍数
                    ind = random.randint(0, self.__len__())

        pind, cind = self.get_prev_next_inds(ind)  # 按照给定的序号，取对应图像帧前后序号
        bbox = self.get_box()  # 获取小图片的位置索引

        next_image, next_image_ts = self.get_image(cind, bbox)  # 按照序号和像素截取位置索引取出（图片+对应事件序号）
        prev_image, prev_image_ts = self.get_image(pind, bbox)
        flow_frame = self.get_flow(pind, cind, bbox)

        events_flow = self.get_events(pind, cind, bbox)  # 取出对应位置和跨越这几帧的事件

        if self.train:
            events_flow = self.random_dropout_events(events_flow, self.dropout_ratio)

        event_volume = gen_discretized_event_volume(torch.from_numpy(events_flow).cpu(),
                                                    [self.n_time_bins * 2,  # n_time_bins：体素的片数（正负分开处理）
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

        prev_image_gt, next_image_gt = prev_image, next_image

        f_img = flow2image(torch.from_numpy(flow_frame))  # 可视化光流图
        # 训练则进行数据增强
        if self.train:
            # 按照一定的概率翻转图像（事件叠加图，以及对应GT图左右或者上下翻转），数据增强
            # 这里是对图像进行增强，光流是否能翻转
            if np.random.rand() < self.flip_x:
                event_volume = torch.flip(event_volume, dims=[2])
                prev_image = np.flip(prev_image, axis=2)
                next_image = np.flip(next_image, axis=2)

                flow_frame = np.flip(flow_frame, axis=2)  # 对光流的翻转可能存在问题，要试试看
                flow_frame[0] = -flow_frame[0]  # 翻转后光流左右相反，数值取反
                # f_img = flow2image(flow_frame)
            if np.random.rand() < self.flip_y:
                event_volume = torch.flip(event_volume, dims=[1])
                prev_image = np.flip(prev_image, axis=1)
                next_image = np.flip(next_image, axis=1)

                flow_frame = np.flip(flow_frame, axis=1)
                flow_frame[1] = -flow_frame[1]  # 翻转后光流上下相反，数值取反
            prev_image_gt, next_image_gt = prev_image, next_image
            if self.appearance_augmentation:
                prev_image, next_image = self.apply_illum_augmentation(prev_image, next_image)

        """下面是计算mask，有事件的位置为true"""
        # 计算位置掩码（出现事件的位置）
        WIDTH = self.image_size[1]  # 阵面宽
        HEIGHT = self.image_size[0]  # 阵面高
        events_num = events_flow.shape[0]

        filled_events_flow = np.zeros((self.Longest_event, 4))
        filled_events_flow[0:events_num] = events_flow

        # events_frame = event2image(torch.from_numpy(events_frame))

        outer_mask = torch.zeros(HEIGHT, WIDTH, dtype=torch.int8)
        fill = torch.ones(events_num, dtype=torch.int8)
        index = (torch.from_numpy(events_flow[:, 1]).long(), torch.from_numpy(events_flow[:, 0]).long())
        outer_mask = outer_mask.index_put(index, fill)  #
        # test=outer_mask.nonzero()
        # cv2.imshow("outer_mask", outer_mask.numpy())
        # cv2.waitKey(0)
        # start_time = time.time()

        events_frame = event2image(torch.from_numpy(events_flow), self.image_size)
        # 计算mask后的GT光流
        # opt_flow_frame = torch.from_numpy(opt_flow_frame)
        # pointgt_flow=opt_flow_frame.mul(torch.stack([outer_mask[0]],dim=0))
        # Event volume is t-y-x（T*H*W）
        # f_img = flow2image(flow_frame)
        # mask_img=outer_mask.numpy()*100
        # cv2.imshow("outer_mask", mask_img)
        # cv2.waitKey(0)

        flow_mask_u = np.zeros_like(flow_frame[0])
        flow_mask_u[flow_frame[0].nonzero()] = 1
        flow_mask_v = np.zeros_like(flow_frame[0])
        flow_mask_v[flow_frame[1].nonzero()] = 1
        flow_mask = np.logical_or(flow_mask_u, flow_mask_v)
        flow_mask = np.uint8(flow_mask)
        # cv2.imshow("flow_mask", flow_mask)
        # cv2.waitKey(0)
        flow_mask = torch.from_numpy(flow_mask)
        end_time = time.time()

        # print(end_time - start_time)

        if self.sample is not None:
            # if self.sample == 0:
            #     event_count_images[0, ...] = 0
            # if self.sample == 1:
            #     event_count_images[1, ...] = 0
            # if self.sample == 2:
            #     event_time_images[0, ...] = 0
            # if self.sample == 3:
            #     event_time_images[1, ...] = 0

            if self.sample == 0:
                event_count_images[1, ...] = 0
                event_time_images[...] = 0
            if self.sample == 1:
                event_count_images[0, ...] = 0
                event_time_images[...] = 0
            if self.sample == 2:
                event_count_images[1, ...] = 0
                event_time_images[...] = 0
            if self.sample == 3:
                event_count_images[0, ...] = 0
                event_time_images[...] = 0

        output = {"prev_image": prev_image.copy(),  # 起始帧图像（经过裁剪，翻转等增强）
                  "prev_image_gt": prev_image_gt.copy(),  # 还是起始帧图像（未经过最后的亮度增强伽马变换等）
                  "prev_image_ts": prev_image_ts,  # 对应的起始事件序号
                  "next_image": next_image.copy(),  # 末尾帧图像（数据增强，）
                  "next_image_gt": next_image_gt.copy(),  # 末尾帧图像（未经过最后的亮度增强伽马变换等）
                  "next_image_ts": next_image_ts,  # 对应的末尾事件序号
                  "event_volume": event_volume,  # 投影得到的事件体素（T*H*W）
                  "flow_frame": flow_frame.copy(),  # 光流GT值
                  "outer_mask": outer_mask,  # 事件出现位置的掩码，后续算loss可能用得到(有事件的为1)
                  "event_count_images": event_count_images,  # 根据各个像素位置累积的事件出现次数累积图像，前正后负
                  "event_time_images": event_time_images,  # 按照最近（新）事件的时间戳生成的事件时间戳图像，前正后负
                  "events_frame": events_frame,  # 事件序列可视化图像
                  "events_num": events_num,  # 返回的事件序列有效长度
                  "events_flow": filled_events_flow,  # 返回的原始时间序列，有效长度为events_num，其余位置填充0
                  "flow_mask": flow_mask  # 返回光流的mask，如果光流uv都为0则返回0，其余位置是1
                  }

        # 注意event_volume：1~N是正事件，N+1~2N是负事件体素。所以1和N+1是一组，N与2N是一组
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

    def __len__(self):

        return (self.num_images - self.start_frame - self.max_skip_frames - 1)


if __name__ == '__main__':
    data = MyData(2)
    # print(data.__len__())
    # print(data.__getitem__(5))
    # data = MyData(-1)

    from torch.utils.data import DataLoader

    loader = DataLoader(data, 1, drop_last=True)
    for i, batch in enumerate(loader):
        print(i, batch['event_count_images'])
        break

from torch.utils.data import Dataset
import h5py
import numpy as np
import time
import os
import random
import torch
import cv2
from result_visualization import  event2image_MVSEC
from event_utils import  gen_discretized_event_volume
from ProjectPoints_MVSEC import Projection_model_MVSEC
import tqdm
import open3d as o3d
import copy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(555)

data_file_list=["indoor_flying1_matching.hdf5",  # /home/xhlinxm/MVSEC/indoor_flying/indoor_flying1_data.hdf5
                "indoor_flying2_matching.hdf5",
                "indoor_flying3_matching.hdf5",
                "indoor_flying4_matching.hdf5",
              ]

def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)

class Data_MVSEC(Dataset):  # 这个类用于准备数据（取出所有数据）
    def __init__(self, folder_path,data_file_index_list, train=True, stack_mode="SBT"):
        assert stack_mode == "SBT" or stack_mode == "SBN"
        assert data_file_index_list[-1]<=4 and len(data_file_index_list)>0#且序号必须由小到大
        frame_interval = "img"
        self.data_file_index_list=data_file_index_list
        self.data_file_list = []   # 文件夹路径
        for index_list in data_file_index_list:
            self.data_file_list.append(data_file_list[index_list])
        self.folder_path = folder_path

        self.dropout_ratio = 0.05  # 一定的概率丢弃事件，进行增强
        self.appearance_augmentation = False  # 对投影生成的体素图像进行视觉增强（伽马变换）
        self.normalize_events = True  # 事件规范化
        self.train = train  # 决定是否进行数据增强，以及取数据的间隔（验证时固定按照图像帧数取）
        self.top_left = [2, 13]  # 设定截取的事件位置边界左上角
        self.raw_image_height=260
        self.raw_image_width = 346
        self.target_image_height = 256  #输出的图像像素大小
        self.target_image_width = 256  # 输出的图像像素大小
        self.n_time_bins = 3  # 要体素化离散化的片数
        self.max_skip_frames = 3  # SBN时限制最大时长（去除过于稀疏数据）
        self.min_n_events = 10000  # 每帧事件的最小数量（SBT时限制）
        self.flip_x = 0 # "Probability of flipping the volume in x.",
        self.flip_y = 0  # "Probability of flipping the volume in y.",
        self.Longest_event = 95600  # 输出原始事件序列时的最大长度
        self.frame_interval = frame_interval  # img 或者gt的间隔
        self.Stacking_num = 3  # 一帧图像帧再被分为n片
        self.SBT_time = 0.06  # 固定事件间隔，除以SBT_num就是每一片的时间（秒）
        self.SBN_num = 8192  # 固定数量的事件叠加成一帧
        self.stack_mode = stack_mode
        self.pcd_num=2048#点云采样数量
        self.random_max_angle = 0.5
        self.random_max_trans = 0.5
        self.Enlarged_projection_field = 0 # 投影边界扩大为原有的2*X倍（左右对称）(0.5*(2^0.5 -1))
        self.Enlarge_projection = True
        if train == True:
            super(Data_MVSEC, self).__init__()
            print("训练使用数据集：", self.data_file_list)
        else:
            # print("测试使用数据集：", self.data_file_list)
            self.flip_x = 0  # 测试的时候不翻转，以便可视化
            self.flip_y = 0
            self.frame_interval = "img"  # 测试时默认按照图像帧间隔分割


        self.load()
        # self.close()
        pass

    def load(self):
        if len(self.data_file_list)==1:
            self.hdf5_file_patch = os.path.join(self.folder_path, self.data_file_list[0])
            H5file = h5py.File(self.hdf5_file_patch, 'r')
            pointcloud = H5file['velodyne/pointcloud']
            pointcloud_ts = H5file['velodyne']['pointcloud_ts']
            self.events = H5file['davis/left/events']#[w,h,t,p]
            self.events_time = self.events[:, 2]

            odometry_ts = H5file['davis']['left']['odometry_ts']
            odometry = H5file['davis/left/odometry']  # 这个是帧间位姿，与激光雷达联合获取深度与光流GT（随时间漂变）
            self.gt_flow_ts = odometry_ts#gtflow是根据odo生成的
            self.gt_flow_frame = H5file['davis']['left']['gtflow']
            # self.depth_image_raw = H5file['davis/left/depth_image_raw']
            self.depth_image_rect = H5file['davis/left/depth_image_rect']

            self.pose = H5file['davis/left/pose']  # 与上面是两种等效的表达方式，上面的更高效
            self.pose_ts = H5file['davis']['left']['pose_ts'][:]  # pose是全局对准的位姿

            self.image_frame = H5file['davis']['left']['img_frame']
            self.image_ts = H5file['davis']['left']['image_ts']
            self.image_event_index=H5file['davis']['left']['image_event_index']
            curr_scan_pts = o3d.io.read_point_cloud(self.folder_path+"map/Mvsec/"+str(self.data_file_index_list[0]+1)+".pcd")

            if np.asarray(curr_scan_pts.points).shape[0]==0:
                print("点云读取错误，未打开文件")
            self.full_LiDAR_map = []
            self.last_pose_index=[]
            self.last_pose_index.append(self.pose_ts.shape[0])
            self.full_LiDAR_map.append(np.asarray(curr_scan_pts.points))
            self.MVSEC_Projection_model = Projection_model_MVSEC(self.folder_path,"indoor_flying")

            self.loaded = True
        elif len(self.data_file_list)>1:
            flag = 0
            self.full_LiDAR_map=[]
            self.last_pose_index =[]
            for data_file in self.data_file_list:
                self.hdf5_file_patch = os.path.join(self.folder_path, data_file)
                H5file = h5py.File(self.hdf5_file_patch, 'r')
                if flag < 1:
                    self.events = H5file['davis/left/events'] [:] # [w,h,t,p]
                    self.events_time = self.events[:, 2]
                    self.gt_flow_ts = H5file['davis']['left']['odometry_ts'] [:]# gtflow是根据odo生成的
                    self.gt_flow_frame = H5file['davis']['left']['gtflow'][:]
                    self.depth_image_rect = H5file['davis/left/depth_image_rect'][:]
                    self.pose = H5file['davis/left/pose'] [:] # 与上面是两种等效的表达方式，上面的更高效
                    self.pose_ts = H5file['davis']['left']['pose_ts'][:]  # pose是全局对准的位姿
                    self.image_frame = H5file['davis']['left']['img_frame'][:]
                    self.image_ts = H5file['davis']['left']['image_ts'][:]
                    self.image_event_index = H5file['davis']['left']['image_event_index'][:]
                    # print(self.folder_path + "map/Mvsec/" + str(self.data_file_index_list[flag] + 1) + ".pcd")
                    curr_scan_pts = o3d.io.read_point_cloud(
                        self.folder_path + "map/Mvsec/" + str(self.data_file_index_list[flag] + 1) + ".pcd")
                    if np.asarray(curr_scan_pts.points).shape[0] == 0:
                        print("点云读取错误，未打开文件")
                    self.full_LiDAR_map.append(np.asarray(curr_scan_pts.points))
                    self.MVSEC_Projection_model = Projection_model_MVSEC(self.folder_path, "indoor_flying")
                    self.last_pose_index.append( self.pose_ts.shape[0])
                    flag+=1
                else:
                    self.events = np.concatenate((self.events,H5file['davis/left/events'][:]),axis=0)  # [w,h,t,p]
                    self.events_time = self.events[:, 2]
                    self.gt_flow_ts = np.concatenate((self.gt_flow_ts,H5file['davis']['left']['odometry_ts'][:] ),axis=0) # gtflow是根据odo生成的
                    self.gt_flow_frame = np.concatenate((self.gt_flow_frame,H5file['davis']['left']['gtflow'][:]),axis=0)
                    self.depth_image_rect = np.concatenate((self.depth_image_rect,H5file['davis/left/depth_image_rect'][:]),axis=0)
                    self.pose = np.concatenate((self.pose,H5file['davis/left/pose'][:]),axis=0)  # 与上面是两种等效的表达方式，上面的更高效
                    self.pose_ts = np.concatenate((self.pose_ts,H5file['davis']['left']['pose_ts'][:] ),axis=0) # pose是全局对准的位姿
                    self.image_frame = np.concatenate((self.image_frame,H5file['davis']['left']['img_frame'][:]),axis=0)
                    self.image_ts = np.concatenate((self.image_ts,H5file['davis']['left']['image_ts'][:]),axis=0)
                    self.image_event_index = np.concatenate((self.image_event_index,H5file['davis']['left']['image_event_index'][:]),axis=0)
                    curr_scan_pts = o3d.io.read_point_cloud(
                        self.folder_path + "map/Mvsec/" + str(self.data_file_index_list[flag] + 1) + ".pcd")
                    if np.asarray(curr_scan_pts.points).shape[0] == 0:
                        print("点云读取错误，未打开文件")
                    self.full_LiDAR_map.append(np.asarray(curr_scan_pts.points))
                    print("已完成数据集 ", flag+1, "的读取")
                    self.last_pose_index.append(self.pose_ts.shape[0])
                    flag+=1

            self.loaded = True

        else:
            print("数据集list有误")
    def __len__(self):#这里以pose的数量为基准
        if self.stack_mode == "SBT":  # 按照固定时间间隔
            # 每个事件都可以作为起点（之后加一个事件数量的筛选），尾部空出2帧
            # events_num=self.events.shape[0]
            end_time = self.events[-1, 2] - 3 * self.SBT_time
            # self.events_time = self.events[:, 2]
            end_pose_index = np.searchsorted(self.pose_ts, end_time, side='left')
            return end_pose_index
        else:  # 不是SBT就是SBN（按照固定事件数量累积）
            # 选定一个事件作为起点，直接加上固定数量就可以
            events_num = self.events.shape[0]
            last_valid_event=events_num-3*self.SBN_num
            end_time = self.events[last_valid_event, 2]
            end_pose_index = np.searchsorted(self.pose_ts, end_time, side='left')

            return end_pose_index
    #随机丢弃一些事件
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

    #随机丢弃一些事件
    def random_sample_events(self, events, k):
        events = events.transpose(1,0)
        if events.shape[1] >= k:
            choice_idx = np.random.choice(events.shape[1], k, replace=False)
        else:
            fix_idx = np.asarray(range(events.shape[1]))
            while events.shape[1] + fix_idx.shape[0] < k:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(events.shape[1]))), axis=0)
            random_idx = np.random.choice(events.shape[1], k - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        events = events[:, choice_idx]
        return events.transpose(1,0)
    

    # 从大图中随机裁切一定大小的小图（抖动，数据增强）像素位置索引
    def get_box(self):
        # top_left = self.top_left

        DVS_boundary = 0#超参，边界过滤
        top = np.random.randint(low=DVS_boundary,
                                high=self.raw_image_height - DVS_boundary - 1 - self.target_image_height,
                                dtype='int32')
        left = np.random.randint(low=DVS_boundary,
                                 high=self.raw_image_width - DVS_boundary - 1 - self.target_image_width,
                                 dtype='int32')

        # top = int(np.random.rand() * (self.raw_image_height - 1 - self.target_image_height))
        # left = int(np.random.rand() * (self.raw_image_width - 1 - self.target_image_width))
        top_left = [top, left]
        bottom_right = [top_left[0] + self.target_image_height,
                        top_left[1] + self.target_image_width]

        return top_left, bottom_right


    def get_image_by_event_index(self, evind, bbox):  # SBN/SBT中ind是events序号
        top_left, bottom_right = bbox
        img_index = np.searchsorted(self.image_ts, self.events_time[evind])
        image = self.image_frame[img_index][top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]

        # image = image.transpose((2, 0, 1))  # （256,256,1）到（1,256,256）
        image = image.astype(np.float32) / 255.  # 转换成0~1
        image_frame_event_index = self.image_event_index[img_index]

        cv_show_prev_image = self.image_frame[img_index].astype(np.float32)/255
        # cv_show_prev_image=cv_show_prev_image.transpose((2, 0, 1)).astype(np.float32) / 255.
        # cv_show_prev_image-=0.5
        # cv_show_prev_image *= 2.
        # cv2.imshow("prev_image", cv_show_prev_image)
        # cv2.waitKey(0)
        # cv2.imwrite("prev_image"+str(ind)+".bmp", cv_show_prev_image)
        return image, image_frame_event_index,cv_show_prev_image


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
        events_masked = events[mask]
        if events_masked.shape[0] <= self.target_events_num:
            return [], []
        # events原始是w*h*t*p
        events_masked = copy.deepcopy(events[mask])
        # 对取出的事件xy坐标进行偏移校正
        events_shifted = events_masked
        events_shifted[:, 0] = events_masked[:, 0] - top_left[1]
        events_shifted[:, 1] = events_masked[:, 1] - top_left[0]

        # test=np.min(events_shifted[:, 2])

        events_shifted[:, 2] -= np.min(events_shifted[:, 2])  # 时间归一化

        # convolution expects 4xN
        # events_shifted = np.transpose(events_shifted).astype(np.float32)
        events_shifted = events_shifted.astype(np.float32)

        events_full = events

        return events_shifted,events_full


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

        t_index = torch.div(((t - t_min) / (t_max - t_min)) , (1 / Stacking_num + 1e-7), rounding_mode='trunc')  # 时间归一化0~1，之后//片数
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

    def get_depth(self, start_ind, end_ind, bbox):  # 2021年5月24日加

        top_left, bottom_right = bbox

        start_time = self.events_time[start_ind]
        end_time = self.events_time[end_ind]
        mid_time = (start_time + end_time) / 2

        left_gt_index = np.searchsorted(self.gt_flow_ts, mid_time, side='right') - 1
        right_gt_index = np.searchsorted(self.gt_flow_ts, mid_time, side='left')
        left_gt_time = self.gt_flow_ts[left_gt_index]
        right_gt_time = self.gt_flow_ts[right_gt_index]
        # searchsorted 是搜索插入位置，np.searchsorted([1,2,3,4,5], 3, side='right')=3
        # 所以side的设置只有在相等的情况下才会有差异

        k = (mid_time - left_gt_time) / (right_gt_time - left_gt_time)
        left_depth_img = self.depth_image_rect[left_gt_index][top_left[0]: bottom_right[0],
                         top_left[1]: bottom_right[1]]
        right_depth_img = self.depth_image_rect[right_gt_index][top_left[0]: bottom_right[0],
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
    def enlarge_projection_field(self,bbox):
        enlarge_width_pixel=int(self.Enlarged_projection_field * self.target_image_width)
        enlarge_height_pixel = int(self.Enlarged_projection_field * self.target_image_height)
        enlarge_bbox=copy.deepcopy(bbox)
        enlarge_bbox[0][0]-=enlarge_height_pixel
        enlarge_bbox[0][1] -= enlarge_width_pixel
        enlarge_bbox[1][0] += enlarge_height_pixel
        enlarge_bbox[1][1] += enlarge_width_pixel

        return enlarge_bbox

    # 取数据的主函数
    def get_single_item(self, getdata_pose_index):#取出第ind笔(位姿)对应的数据
        start_time = time.time()
        start_time = time.time()
        # frame_start_event_index, frame_end_event_index = self.get_prev_next_inds(ind)  # 按照给定的序号，取对应图像帧前后序号
        if self.train:
            if self.stack_mode == "SBT":  # 按照固定时间间隔取
                # 保证等时间下取出的事件点大于一定数量
                n_events = 0
                n_iters = 0  # 测试迭代次数
                while n_events < self.min_n_events:
                    frame_pose_time=self.pose_ts[getdata_pose_index]
                    frame_start_time = frame_pose_time-0.5*self.SBT_time
                    frame_end_time = frame_pose_time+0.5*self.SBT_time
                    frame_start_event_index = np.searchsorted(self.events_time, frame_start_time, side='left')
                    frame_end_event_index = np.searchsorted(self.events_time, frame_end_time, side='left')
                    n_events = frame_end_event_index - frame_start_event_index # self.count_events(frame_start_event_index, frame_end_event_index,self.frame_interval)  # 计算当前帧范围内事件总数
                    n_iters += 1

                    final_frame_interval = self.SBT_time#取出的时间流时间长度
                    if n_events < self.min_n_events:  # 如果固定时间取到的帧事件数量太小，则跳过，随机再取一个帧
                        getdata_pose_index = random.randint(0, self.__len__())#重新给一个序号生成
                        n_events = -1
            elif self.stack_mode == "SBN" :  # "SBN":#按照固定事件数量取
                frame_interval = 99999 * self.SBT_time#设置一个较大的初始值
                while frame_interval > self.max_skip_frames * self.SBT_time:
                    frame_pose_time = self.pose_ts[getdata_pose_index]
                    frame_mid_event_index =np.searchsorted(self.events_time, frame_pose_time, side='left')
                    frame_start_event_index=frame_mid_event_index-self.SBN_num//2
                    frame_end_event_index =frame_mid_event_index+self.SBN_num//2

                    frame_start_time = self.events_time[frame_start_event_index]
                    frame_end_time = self.events_time[frame_end_event_index]
                    frame_interval = frame_end_time - frame_start_time
                    final_frame_interval = frame_interval
                    if frame_interval > self.max_skip_frames * self.SBT_time or frame_start_event_index<=0:  # 如果间隔太大，证明这段事件太稀疏
                        # test=self.__len__()[0]
                        getdata_pose_index = random.randint(0, int(self.__len__()))
                        frame_interval = 99999 * self.SBT_time
                # 如果固定的数量覆盖的时间太长就舍弃
            else:
                print("stack_mode 出错")

        else:  # 测试集(代码同训练，后期可在训练集增加数据增强内容)
            if self.stack_mode == "SBT":  # 按照固定时间间隔取
                # 保证等时间下取出的事件点大于一定数量
                n_events = 0
                n_iters = 0  # 测试迭代次数
                while n_events < self.min_n_events:
                    frame_pose_time = self.pose_ts[getdata_pose_index]
                    frame_start_time = frame_pose_time - 0.5 * self.SBT_time
                    frame_end_time = frame_pose_time + 0.5 * self.SBT_time
                    frame_start_event_index = np.searchsorted(self.events_time, frame_start_time, side='left')
                    frame_end_event_index = np.searchsorted(self.events_time, frame_end_time, side='left')
                    n_events = frame_end_event_index - frame_start_event_index  # self.count_events(frame_start_event_index, frame_end_event_index,self.frame_interval)  # 计算当前帧范围内事件总数
                    n_iters += 1

                    final_frame_interval = self.SBT_time  # 取出的时间流时间长度
                    if n_events < self.min_n_events:  # 如果固定时间取到的帧事件数量太小，则跳过，随机再取一个帧
                        getdata_pose_index = random.randint(0, self.__len__())  # 重新给一个序号生成
                        n_events = -1
            elif self.stack_mode == "SBN":  # "SBN":#按照固定事件数量取
                frame_interval = 99999 * self.SBT_time  # 设置一个较大的初始值
                while frame_interval > self.max_skip_frames * self.SBT_time:
                    frame_pose_time = self.pose_ts[getdata_pose_index]
                    frame_mid_event_index = np.searchsorted(self.events_time, frame_pose_time, side='left')
                    frame_start_event_index = frame_mid_event_index - self.SBN_num // 2
                    frame_end_event_index = frame_mid_event_index + self.SBN_num // 2

                    frame_start_time = self.events_time[frame_start_event_index]
                    frame_end_time = self.events_time[frame_end_event_index]
                    frame_interval = frame_end_time - frame_start_time
                    final_frame_interval = frame_interval
                    if frame_interval > self.max_skip_frames * self.SBT_time or frame_start_event_index<0:  # 如果间隔太大，证明这段事件太稀疏
                        getdata_pose_index = random.randint(0, self.__len__())
                        frame_interval = 99999 * self.SBT_time
                # 如果固定的数量覆盖的时间太长就舍弃
            else:
                print("stack_mode 出错")

        # bbox = self.get_box()  # 获取小图片的位置索引
        flag = False
        target_area = self.target_image_height * self.target_image_width
        full_area = self.raw_image_width * self.raw_image_height
        self.target_events_num = 0.8 * (target_area / full_area) * (frame_end_event_index - frame_start_event_index)

        while flag == False:
            bbox = self.get_box()  # 获取小图片的位置索引
            events_flow, events_flow_full = self.get_events(frame_start_event_index, frame_end_event_index, bbox)
            if len(events_flow) == 0:
                continue
            # 取出对应位置和跨越这几帧的事件(事件坐标进行了偏移，按照截取的范围为基准)
            if events_flow.shape[0] >= self.target_events_num:
                flag = True

        frame_mid_event_index = np.searchsorted(self.events_time, frame_pose_time)
        mid_image, mid_image_frame_event_index, mid_full_image = self.get_image_by_event_index(frame_mid_event_index,bbox)
        #上面的图像有除255

        # start_time = time.time()

        # elapsed = (time.time() - start_time)
        # print("GT用时:", elapsed)


        depth_frame, depth_mask = self.get_depth(frame_start_event_index, frame_end_event_index, bbox, )
        # cv2.imshow("depth_frame", depth_frame)
        # cv2.waitKey(0)

        # if self.train:
            # 此模式下不太需要这个增强，SBN可能不应进行这个增强
            # events_flow = self.random_dropout_events(events_flow, self.dropout_ratio)
        events_flow_input = self.random_sample_events(events_flow, k = 8192)
        # events_flow_input = events_flow
        event_volume = gen_discretized_event_volume(torch.from_numpy(events_flow).cpu(),
                                                    # [self.n_time_bins * 2,  # n_time_bins：体素的片数（正负分开处理）
                                                    [self.n_time_bins * 2,
                                                     self.target_image_height,  #
                                                     self.target_image_width])

        event_stacking_images = self.stacking_events(events_flow,
                                                     [self.Stacking_num,  # n_time_bins：体素的片数（正负累加处理）
                                                      self.target_image_height,
                                                      self.target_image_width])

        # 上面进行事件投影，获得事件正负独立，体素化离散投影的结果
        # 注意：1~N是正事件，N+1~2N是负事件体素。所以1和N+1是一组，N与2N是一组
        # 计算事件的数量累加图像
        # e_img=event2image(torch.from_numpy(events_RGB_frame).cpu())#可视化
        event_count_images = self.event_counting_images(torch.from_numpy(events_flow).cpu(),
                                                        [2,  # n_time_bins：体素的片数（正负分开处理）
                                                         self.target_image_height,  # W，H
                                                         self.target_image_width])

        event_time_images = self.event_timing_images(torch.from_numpy(events_flow).cpu(),
                                                     [2,  # n_time_bins：体素的片数（正负分开处理）
                                                      self.target_image_height,  # W，H
                                                      self.target_image_width])

        if self.normalize_events:# 对事件累积图像的张量进行归一化（均值0，方差1）,加紧到0.98，丢弃前后2%的数值

            event_volume = self.normalize_event_volume(event_volume)
            event_count_images=self.normalize_event_volume(event_count_images)#新增，非时间统计特征图都可使用
            event_stacking_images=self.normalize_event_volume(event_stacking_images)#新增，非时间统计特征图都可使用
            
        """下面是计算mask，有事件的位置为true"""
        # 计算位置掩码（出现事件的位置）
        WIDTH = self.target_image_width  # 阵面宽
        HEIGHT = self.target_image_height  # 阵面高
        events_num = events_flow.shape[0]

        # events_RGB_frame = event2image(torch.from_numpy(events_RGB_frame))

        event_mask = torch.zeros(HEIGHT, WIDTH, dtype=torch.int8)
        fill = torch.ones(events_num, dtype=torch.int8)
        index = (torch.from_numpy(events_flow[:, 1]).long(), torch.from_numpy(events_flow[:, 0]).long())
        event_mask = event_mask.index_put(index, fill)
        depth_mask = depth_mask.bool()

        # show_mask = event_mask.float().numpy()
        # cv2.imshow("event_mask", show_mask)
        # cv2.waitKey(0)

        # 训练则进行数据增强(翻转)
        # if self.train:
        #     # 按照一定的概率翻转图像（事件叠加图，以及对应GT图左右或者上下翻转），数据增强
        #     # 这里是对图像进行增强，光流是否能翻转
        #     if np.random.rand() < self.flip_x:
        #         event_mask = torch.flip(event_mask, dims=[1])
        #         event_volume = torch.flip(event_volume, dims=[2])
        #         # prev_image = np.flip(prev_image, axis=2)
        #         mid_image = np.flip(mid_image, axis=1)
        #         # next_image = np.flip(next_image, axis=2)
        #         depth_frame = torch.flip(depth_frame, dims=[1])
        #         depth_mask = torch.flip(event_mask, dims=[1])

        #         # flow_frame = np.flip(flow_frame, axis=2)  # 对光流的翻转可能存在问题，要试试看
        #         # flow_frame[0] = -flow_frame[0]  # 翻转后光流左右相反，数值取反

        #         event_count_images = torch.flip(event_count_images, dims=[2])
        #         event_time_images = torch.flip(event_time_images, dims=[2])
        #         event_stacking_images = torch.flip(event_stacking_images, dims=[2])
        #         # f_img = flow2image(flow_frame)
        #         # depth_frame = np.flip(depth_frame, axis=1)
        #     if np.random.rand() < self.flip_y:
        #         event_mask = torch.flip(event_mask, dims=[0])
        #         event_volume = torch.flip(event_volume, dims=[1])
        #         # prev_image = np.flip(prev_image, axis=1)
        #         mid_image = np.flip(mid_image, axis=0)
        #         # next_image = np.flip(next_image, axis=1)
        #         depth_frame = torch.flip(depth_frame, dims=[0])
        #         depth_mask = torch.flip(event_mask, dims=[0])

        #         # flow_frame = np.flip(flow_frame, axis=1)
        #         # flow_frame[1] = -flow_frame[1]  # 翻转后光流上下相反，数值取反

        #         event_count_images = torch.flip(event_count_images, dims=[1])
        #         event_time_images = torch.flip(event_time_images, dims=[1])
        #         event_stacking_images = torch.flip(event_stacking_images, dims=[1])
        #     # prev_image_gt, next_image_gt = prev_image, next_image
        #     if self.appearance_augmentation:
        #         # prev_imag = self.apply_illum_augmentation(prev_image)
        #         mid_image = self.apply_illum_augmentation(mid_image)
        #         # next_image = self.apply_illum_augmentation(next_image)

        events_RGB_frame = event2image_MVSEC(events_flow,[self.target_image_height,self.target_image_width])


        """开始取点云"""
        cam_pose = self.pose[getdata_pose_index]
        inv_cam_pose = np.linalg.inv(cam_pose)  # 世界坐标系转换为相机坐标系，需要取一个逆
        external_parameters = self.MVSEC_Projection_model.T_cam2_lidar  # 相机到雷达的变换，等价于雷达坐标系的点相机坐标系的变换
        K = self.MVSEC_Projection_model.K
        cam_pose_R = inv_cam_pose[0:3, 0:3]  # 相机位姿RT
        cam_pose_T = inv_cam_pose[0:3, 3]
        if len(self.last_pose_index)==1:
            dataset_index=0
        else:
            dataset_index=np.searchsorted(np.array(self.last_pose_index),getdata_pose_index)
        current_LiDAR_map=self.full_LiDAR_map[dataset_index]

        filted_points =self.MVSEC_Projection_model.project_point_filtering( current_LiDAR_map, inv_cam_pose, bbox, 1)
        # point2d_from_lidar, flag_points = self.MVSEC_Projection_model.Point2Pic(filted_points, cam_pose_R, cam_pose_T,
        #                                                                         bbox)
        #投影后2D点是【u，v】，也就是【w,h】
        enlarged_bbox = self.enlarge_projection_field(bbox)
        if self.Enlarge_projection == True:

            enlarged_point2d_from_lidar, enlarged_flag_points = self.MVSEC_Projection_model.Point2Pic(filted_points, cam_pose_R, cam_pose_T,
                                                                                    enlarged_bbox)
            enlarged_valid_points = enlarged_flag_points[enlarged_flag_points[:, 3] > 0]
            point2d_from_lidar, flag_points = self.MVSEC_Projection_model.Point2Pic(enlarged_valid_points, cam_pose_R,
                                                                                    cam_pose_T,
                                                                                    bbox)
        else :
            point2d_from_lidar, flag_points = self.MVSEC_Projection_model.Point2Pic(filted_points, cam_pose_R,
                                                                                    cam_pose_T,
                                                                                    bbox)
            enlarged_valid_points = np.empty(flag_points[0, 0:3])

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(filted_points[:, :3])  # 显示全部点云
        # o3d.visualization.draw_geometries([pcd])


        # compensated_point2d_from_lidar=point2d_from_lidar
        # compensated_point2d_from_lidar[:,0]=compensated_point2d_from_lidar[:,0]-bbox[0][1]
        # compensated_point2d_from_lidar[:, 1]=compensated_point2d_from_lidar[:, 1] - bbox[0][0]

        # mid_image_with_point = self.MVSEC_Projection_model.draw_point(mid_full_image, point2d_from_lidar)
        # mid_image_with_point = self.MVSEC_Projection_model.draw_point(mid_image, point2d_from_lidar)
        # events_RGB_frame_with_point = self.MVSEC_Projection_model.draw_point(events_RGB_frame,compensated_point2d_from_lidar)
        # cv2.imshow("tradition_image", mid_image)#调试可视化
        # cv2.waitKey(0)
        # cv2.imshow("mid_image_with_point", mid_image_with_point)#调试可视化
        # cv2.waitKey(0)
        # cv2.imshow("events_RGB_frame_with_point", events_RGB_frame_with_point)#调试可视化
        # cv2.waitKey(0)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(flag_points[:, :3])  # 显示全部点云
        # o3d.visualization.draw_geometries([pcd])#调试可视化

        valid_pcd = o3d.geometry.PointCloud()
        valid_points = copy.deepcopy(flag_points[flag_points[:, 3] > 0])
        
        # np.random.choice(X, 3, replace=False)
        # random.randint(0, self.__len__())
        if valid_points.shape[0]<self.pcd_num:
            return self.get_single_item(random.randint(0, self.__len__()))
        sample_index=np.random.choice(valid_points.shape[0], self.pcd_num, replace=True)
        valid_points=valid_points[sample_index]#点云随机采样为固定数量
        sample_index = np.random.choice(enlarged_valid_points.shape[0], self.pcd_num, replace=True)
        enlarged_valid_points = enlarged_valid_points[sample_index]
        if self.train:
            random_RT = random_pose(self.random_max_angle, self.random_max_trans)
            random_RT=torch.from_numpy(random_RT)
            # P(2D)=P(C2L)P(3d)=P(C2L)R` RP(3d)
            valid_points=torch.from_numpy(valid_points)
            enhance_points = copy.deepcopy(valid_points)

            enlarged_valid_points=torch.from_numpy(enlarged_valid_points)
            enhance_enlarged_points = copy.deepcopy(enlarged_valid_points)


            repat_pose = random_RT.expand(enhance_points.shape[0], -1, -1)
            enhance_points = torch.unsqueeze(enhance_points, dim=2)
            enhance_enlarged_points = torch.unsqueeze(enhance_enlarged_points, dim=2)
            #R*P(3d)
            enhanced_points = torch.bmm(repat_pose, enhance_points)  # 将点云从雷达坐标系转换到相机坐标系
            enhanced_points = torch.squeeze(enhanced_points)
            enhance_enlarged_points = torch.bmm(repat_pose, enhance_enlarged_points)  # 将点云从雷达坐标系转换到相机坐标系
            enhance_enlarged_points = torch.squeeze(enhance_enlarged_points)

            enhanced_cam_pose = copy.deepcopy(inv_cam_pose)  # inv_cam_pose是最终输出的位姿
            random_RT=random_RT.numpy()
            # P(C2L)R`(R逆)
            enhanced_cam_pose=np.matmul(enhanced_cam_pose,np.linalg.inv(random_RT))

            valid_points=enhanced_points
            enlarged_valid_points=enhance_enlarged_points
            inv_cam_pose=enhanced_cam_pose
        # if filted_points.shape[0]<2*self.pcd_num:
        #     return self.get_single_item(random.randint(0, self.__len__()))

        # valid_pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])  # 显示参与有效投影的点云
        

        # o3d.visualization.draw_geometries([valid_pcd])#调试可视化
        """取点云完成"""

        # cv_show_prev_image = prev_image_gt[0]#prev_image
        # cv2.imshow("prev_image", cv_show_prev_image)
        # cv2.waitKey(0)
        
        gradient_x = cv2.Sobel(mid_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(mid_image, cv2.CV_64F, 0, 1, ksize=3)
        image_gradient_map = np.stack([gradient_x, gradient_y])
        mid_image = torch.from_numpy(mid_image.copy()).float()

        # flow_frame = torch.from_numpy(flow_frame.copy()).float()
        # depth_frame = torch.from_numpy(depth_frame).float()
        image_gradient_map = torch.from_numpy(image_gradient_map).float()
        image_gradient_map = torch.squeeze(image_gradient_map)

        events_RGB_frame = torch.from_numpy(events_RGB_frame)


        elapsed = (time.time() - start_time)
        # print("取数据用时:", elapsed)
        
        output = {
            "mid_image": mid_image,#中间时刻的灰度图
            "event_volume": event_volume,  # 投影得到的事件体素（T*H*W）
            # "flow_frame": flow_frame,  # 光流GT值
            "event_mask": event_mask,  # 事件出现位置的掩码，后续算loss可能用得到(有事件的为1)
            "event_count_images": event_count_images,  # 根据各个像素位置累积的事件出现次数累积图像，前正后负
            "event_time_images": event_time_images,  # 按照最近（新）事件的时间戳生成的事件时间戳图像，前正后负
            "events_RGB_frame": events_RGB_frame,  # 事件序列可视化图像
            "events_num": events_num,  # 返回的事件序列有效长度(!!!!没有减去mask掉的位置)
            "depth_frame": depth_frame,  # 增加深度帧
            "depth_mask": depth_mask,  # 深度图的mask，已经过取反，true即有深度的位置，直接相乘即可
            "image_gradient": image_gradient_map,
            "event_stacking_images": event_stacking_images,  # 堆叠的图
            "valid_points":valid_points, #参与有效投影的点云
            # "filted_points":filted_points #投影半球方向内的所有点云
            "cam_pose":inv_cam_pose,
            "K":K,
            "events_flow":events_flow_input,
            "enlarged_valid_points":enlarged_valid_points,
            "box":bbox,
        }

        # 注意event_volume：1~N是正事件，N+1~2N是负事件体素。所以1和N+1是一组，N与2N是一组
        # elapsed = (time.time() - start_time)
        # print("取数据-Time used:", elapsed)
        return output

    def __getitem__(self, frame_index):
        # start = time.process_time()
        if not self.loaded:
            self.load()
        # elapsed = (time.process_time() - start)
        # print("取数据用时:", elapsed)
#         print(" __getitem__取了", frame_index)
        return self.get_single_item(frame_index)



if __name__ == '__main__':
    # data = Data_MVSEC("E:/事件匹配/Event_LiDAR_match/data/",1, train=True, stack_mode="SBT")
    data = Data_MVSEC("E:/事件匹配/Event_LiDAR_match/data/", 0, train=True, stack_mode="SBN")
    test_len=data.__len__()

    t2 = data.__getitem__(666)
    # t3 = data.__getitem__(9607)#3166 1251
    # print(data.__len__())
    # print(data.__getitem__(5))
    # data = MyData(-1)

    from torch.utils.data import DataLoader

    loader = DataLoader(data, batch_size=2, drop_last=True,shuffle=True,num_workers=0)

    for i, batch in enumerate(loader):
        print(i)
        # print(i, batch['event_count_images'])








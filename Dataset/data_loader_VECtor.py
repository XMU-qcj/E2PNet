from torch.utils.data import Dataset
import h5py
import hdf5plugin#必须保留
import numpy as np
import time
import os
import random
import torch
import cv2
from result_visualization import  event2image_VECtor,event2image_MVSEC
from event_utils import  gen_discretized_event_volume
from ProjectPoints_VECtor import Projection_model_VECtor,seven_to_pose
import torchvision.transforms as transforms
import tqdm
import open3d as o3d
import copy



dataset_file_list=["units_dolly",  # G:\VECtor Benchmark\Large-scale
                   "corridors_dolly",
                   "corridors_walk",
                   "units_scooter",
                   "school_dolly",
                   "school_scooter",
                    ]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(555)

def remove_hidden_point(points,cam_pose):
    # 加载点云
    inv_cam_pose = np.linalg.inv(cam_pose)
    cam_T=np.array([0,0,0,1])
    cam_T=np.matmul(inv_cam_pose,cam_T)
    cam_T=cam_T[0:3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd], window_name="原始点云", width=800, height=600)
    # diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    # radius = diameter * 100 # The radius of the sperical projection
    _, pt_map = pcd.hidden_point_removal(cam_T, 5000)  # 获取视点位置能看到的所有点的索引 pt_map
    # 可视点点云
    pcd_visible = pcd.select_by_index(pt_map)

    # pcd_visible.paint_uniform_color([0, 0, 1])  # 可视点为蓝色
    # print("可视点个数为：", pcd_visible)
    # # 隐藏点点云
    # pcd_hidden = pcd.select_by_index(pt_map, invert=True)
    # pcd_hidden.paint_uniform_color([1, 0, 0])  # 隐藏点为红色
    # print("隐藏点个数为：", pcd_hidden)
    # o3d.visualization.draw_geometries([pcd_visible, pcd_hidden], window_name="隐点移除", width=800, height=600)


    return np.asarray(pcd_visible.points)

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




class Data_VECtor(Dataset):  # 这个类用于准备数据（取出所有数据）
    def __init__(self, Parameter_path, Data_path, data_file_index_list, train=True, stack_mode="SBT",pic_mode="DVS"):
        assert stack_mode == "SBT" or stack_mode == "SBN"
        assert pic_mode == "DVS" or pic_mode == "IMG"

        self.corresponding_mode=pic_mode
        self.data_file_list=[]
        for index_list in data_file_index_list:
            self.data_file_list.append( dataset_file_list[index_list])

        self.Parameter_path = Parameter_path   # (内外参数)文件夹路径
        self.Data_path=Data_path#数据集根目录路径

        self.dropout_ratio = 0.05  # 一定的概率丢弃事件，进行增强
        self.appearance_augmentation = False  # 对投影生成的体素图像进行视觉增强（伽马变换）
        self.normalize_events = True  # 事件规范化
        self.train = train  # 决定是否进行数据增强，以及取数据的间隔（验证时固定按照图像帧数取）
        self.top_left = [40, 40]  # 设定截取的事件位置边界左上角
        self.raw_DVS_height=480
        self.raw_DVS_width = 640
        self.target_DVS_image_height = 256 #输出的事件图像像素大小
        self.target_DVS_image_width = 256# 输出的事件图像像素大小
        self.raw_IMG_height = 1024
        self.raw_IMG_width = 1224
        self.target_IMG_image_height = 512  # 输出的灰度图像像素大小（视野一致的话大小为2倍，再下采样）
        self.target_IMG_image_width = 512 # 输出的灰度图像像素大小
        self.n_time_bins = 3  # 要体素化离散化的片数
        self.max_skip_frames = 3  # SBN时限制最大时长（去除过于稀疏数据）
        self.min_n_events = 10000  # 每帧事件的最小数量（SBT时限制）
        self.flip_x = 0 # "Probability of flipping the volume in x.",
        self.flip_y = 0  # "Probability of flipping the volume in y.",
        self.Longest_event = 95600  # 输出原始事件序列时的最大长度
        self.Stacking_num = 3  # 一帧图像帧再被分为n片
        self.SBT_time = 0.06  # 固定事件间隔，除以SBT_num就是每一片的时间（秒）
        self.SBN_num = 8192  # 固定数量的事件叠加成一帧
        self.stack_mode = stack_mode
        self.VECtor_Projection_model=Projection_model_VECtor(self.Parameter_path,"Large-scale")
        self.pcd_num = 8192  # 点云采样数量
        self.Enlarged_projection_field=0.207#投影边界扩大为原有的2*X倍（左右对称）(0.5*(2^0.5 -1))
        self.Enlarge_projection=True#设置投影时是否随机扩大边界
        self.random_max_angle=0.5
        self.random_max_trans=0.5



        if train == True:
            super(Data_VECtor, self).__init__()
            print("训练使用数据集：", self.data_file_list)
        else:
            print("测试使用数据集：", self.data_file_list)
            self.flip_x = 0  # 测试的时候不翻转，以便可视化
            self.flip_y = 0

        self.load()
        pass

    def load(self):
        """读取载入各项文件"""
        """读取事件流的H5文件"""
        if len(self.data_file_list)>1:
            flag=0
            for data_file in self.data_file_list:
                hdf5_file_route = os.path.join(self.Data_path, data_file)
                hdf5_file_route=os.path.join(hdf5_file_route, data_file)
                hdf5_file_route=hdf5_file_route+"1.synced.left_event.hdf5"
                VECtor_events_hdf5 = h5py.File(hdf5_file_route, 'r')
                if flag<1:
                    events_x = VECtor_events_hdf5['events/x'][:]
                    events_y = VECtor_events_hdf5['events/y'][:]
                    events_t = VECtor_events_hdf5['events/t'][:]  # 单位微秒，转到秒要/1e6
                    events_p = VECtor_events_hdf5['events/p'][:]
                    events_ms_to_idx = VECtor_events_hdf5['ms_to_idx'][:]  # 毫秒级别索引对应的事件序号
                    # 即毫秒t为offset序号，得到的值为事件序号
                    events_t_offset = VECtor_events_hdf5['t_offset'][:]  # 事件流的起始绝对时间，上面的t是相对时间，单位微秒10e-6
                    events_t=(events_t_offset + events_t) / 1e6
                    flag+=1
                else:
                    events_x = np.concatenate((events_x,VECtor_events_hdf5['events/x'][:]),axis=0)
                    events_y = np.concatenate((events_y,VECtor_events_hdf5['events/y'][:]),axis=0)
                    events_p = np.concatenate((events_p,VECtor_events_hdf5['events/p'][:]),axis=0)
                    events_t_temp=VECtor_events_hdf5['events/t'][:]  # 单位微秒，转到秒要/1e6
                    events_ms_to_idx = VECtor_events_hdf5['ms_to_idx'][:]  # 毫秒级别索引对应的事件序号
                    # 即毫秒t为offset序号，得到的值为事件序号
                    events_t_offset = VECtor_events_hdf5['t_offset'][:]  # 事件流的起始绝对时间，上面的t是相对时间，单位微秒10e-6
                    events_t_temp = (events_t_offset + events_t_temp) / 1e6
                    assert events_t[0]<events_t_temp[0]#要求时间从小到大
                    events_t = np.concatenate((events_t,events_t_temp),axis=0)
                    flag += 1
        else:
            hdf5_file_route = os.path.join(self.Data_path, self.data_file_list[0])
            hdf5_file_route = os.path.join(hdf5_file_route, self.data_file_list[0])
            hdf5_file_route = hdf5_file_route + "1.synced.left_event.hdf5"
            VECtor_events_hdf5 = h5py.File(hdf5_file_route, 'r')
            events_x = VECtor_events_hdf5['events/x'][:]#
            events_y = VECtor_events_hdf5['events/y'][:]
            events_t = VECtor_events_hdf5['events/t'][:]  # 单位微秒，转到秒要/1e6
            events_p = VECtor_events_hdf5['events/p'][:]
            events_ms_to_idx = VECtor_events_hdf5['ms_to_idx'][:]  # 毫秒级别索引对应的事件序号
            # 即毫秒t为offset序号，得到的值为事件序号
            events_t_offset = VECtor_events_hdf5['t_offset'][:]  # 事件流的起始绝对时间，上面的t是相对时间，单位微秒10e-6
            events_t = (events_t_offset + events_t) / 1e6
        # test=events_x[-1]
        self.events_x=events_x#width
        self.events_y=events_y#h
        self.events_t=events_t
        self.events_p=events_p
        # test=self.events_x[:10000]
        self.events=np.stack([self.events_x,self.events_y,self.events_t,self.events_p],axis=1)
        print("读取事件流的文件完成")
        """读取事件流的文件完成"""

        """读取图像列表"""
        torch.set_printoptions(precision=6, sci_mode=False)
        if len(self.data_file_list)>1:
            flag=0
            image_dir_list = []
            for data_file in self.data_file_list:
                img_folde_dir = os.path.join(self.Data_path, data_file)
                img_folde_dir=os.path.join(img_folde_dir, data_file)
                img_folde_dir=img_folde_dir+"1.synced.left_camera"
                if flag<1:#首次循环的操作
                    image_file_list = os.listdir(img_folde_dir)
                    image_file_list.sort()
                    image_ts = torch.zeros(len(image_file_list))
                    image_ts = image_ts.to(dtype=torch.float64)
                    img_list_idx = 0
                    for name in image_file_list:
                        image_ts[img_list_idx] = float(name[:17])
                        image_dir_list.append(os.path.join(img_folde_dir, name))
                        img_list_idx += 1
                    if img_list_idx != len(image_dir_list):  # img_list_idx从0开始，len()从1开始
                        print("list长度不符", len(image_dir_list))
                    flag+=1
                else:#循环1以上时的操作
                    image_file_list = os.listdir(img_folde_dir)
                    image_file_list.sort()
                    image_ts_temp = torch.zeros(len(image_file_list))
                    image_ts_temp = image_ts_temp.to(dtype=torch.float64)
                    img_list_idx = 0
                    for name in image_file_list:
                        image_ts_temp[img_list_idx] = float(name[:17])
                        image_dir_list.append(os.path.join(img_folde_dir, name))
                        img_list_idx += 1

                    image_ts=torch.cat((image_ts,image_ts_temp),dim=0)

                    if image_ts.shape[0] != len(image_dir_list):  # img_list_idx从0开始，len()从1开始
                        print("ts与长度不符", image_ts.shape[0],len(image_dir_list))
                    flag+=1
        else:#list为1时的基础操作
            img_folde_dir = os.path.join(self.Data_path, self.data_file_list[0])
            img_folde_dir = os.path.join(img_folde_dir, self.data_file_list[0])
            img_folde_dir = img_folde_dir + "1.synced.left_camera"
            image_file_list = os.listdir(img_folde_dir)
            image_file_list.sort()

            image_ts = torch.zeros(len(image_file_list))
            image_ts = image_ts.to(dtype=torch.float64)
            image_dir_list=[]
            img_list_idx = 0
            for name in image_file_list:
                image_ts[img_list_idx] = float(name[:17])
                image_dir_list.append(os.path.join(img_folde_dir,name) )
                img_list_idx += 1
            if image_ts.shape[0] != len(image_dir_list):  # img_list_idx从0开始，len()从1开始
                print("ts与长度不符", image_ts.shape[0], len(image_dir_list))

        self.image_dir_list=image_dir_list
        self.image_ts=image_ts
        print("读取图像列表完成")
        """读取图像列表完成"""

        """开始读取点云"""
        torch.set_printoptions(precision=6, sci_mode=False)
        pcd_dir_list = []
        full_LiDAR_map = []
        if len(self.data_file_list)>1:
            flag=0

            for data_file in self.data_file_list:
                pcd_folde_dir = os.path.join(self.Data_path, data_file)
                pcd_folde_dir=os.path.join(pcd_folde_dir, data_file)
                pcd_folde_dir=pcd_folde_dir+"1.synced.lidar"
                if flag<1:
                    pcd_file_list = os.listdir(pcd_folde_dir)
                    pcd_file_list.sort()
                    pcd_ts = torch.zeros(len(pcd_file_list))
                    pcd_ts = pcd_ts.to(dtype=torch.float64)
                    full_lidar_map_index = torch.zeros(len(pcd_file_list))
                    pcd_list_idx = 0
                    for name in pcd_file_list:
                        pcd_ts[pcd_list_idx] = float(name[:17])
                        pcd_dir_list.append(os.path.join(pcd_folde_dir, name))
                        full_lidar_map_index[pcd_list_idx]=flag
                        pcd_list_idx += 1
                    if pcd_list_idx != len(pcd_dir_list):  # img_list_idx从0开始，len()从1开始
                        print("list长度不符", len(pcd_dir_list))
                    """读取点云全局地图"""
                    full_lidar_map_dir = os.path.join(self.Parameter_path, "map/Vector")
                    full_lidar_map_dir = os.path.join(full_lidar_map_dir, str(self.data_file_list[flag]))
                    full_lidar_map_dir = full_lidar_map_dir + ".pcd"
                    if os.path.exists(full_lidar_map_dir) != True:
                        print("读取点云错误")
                    full_LiDAR_pts = o3d.io.read_point_cloud(full_lidar_map_dir)
                    full_LiDAR_map.append(np.asarray(full_LiDAR_pts.points))
                    flag+=1#首次循环的操作
                else:#循环1以上时的操作
                    pcd_file_list = os.listdir(pcd_folde_dir)
                    pcd_file_list.sort()
                    pcd_ts_temp = torch.zeros(len(pcd_file_list))
                    full_lidar_map_index_temp=torch.zeros(len(pcd_file_list))
                    pcd_ts_temp = pcd_ts_temp.to(dtype=torch.float64)
                    pcd_list_idx = 0
                    for name in pcd_file_list:
                        pcd_ts_temp[pcd_list_idx] = float(name[:17])
                        pcd_dir_list.append(os.path.join(pcd_folde_dir, name))
                        full_lidar_map_index_temp[pcd_list_idx] = flag
                        pcd_list_idx += 1
                    pcd_ts=torch.cat((pcd_ts,pcd_ts_temp),dim=0)
                    full_lidar_map_index=torch.cat((full_lidar_map_index,full_lidar_map_index_temp),dim=0)
                    """读取点云全局地图"""
                    full_lidar_map_dir = os.path.join(self.Parameter_path, "map/Vector")
                    full_lidar_map_dir = os.path.join(full_lidar_map_dir, str(self.data_file_list[flag]))
                    full_lidar_map_dir = full_lidar_map_dir + ".pcd"
                    if os.path.exists(full_lidar_map_dir) != True:
                        print("读取点云错误")
                    full_LiDAR_pts = o3d.io.read_point_cloud(full_lidar_map_dir)
                    full_LiDAR_map.append(np.asarray(full_LiDAR_pts.points))
                    flag+=1
        else:#list为1时的基础操作
            pcd_folde_dir = os.path.join(self.Data_path, self.data_file_list[0])
            pcd_folde_dir = os.path.join(pcd_folde_dir, self.data_file_list[0])
            pcd_folde_dir = pcd_folde_dir + "1.synced.lidar"
            pcd_file_list = os.listdir(pcd_folde_dir)
            pcd_file_list.sort()
            pcd_ts = torch.zeros(len(pcd_file_list))
            pcd_ts = pcd_ts.to(dtype=torch.float64)
            pcd_dir_list=[]
            pcd_list_idx = 0
            for name in pcd_file_list:
                pcd_ts[pcd_list_idx] = float(name[:17])
                pcd_dir_list.append(os.path.join(pcd_folde_dir,name) )
                pcd_list_idx += 1
            if pcd_list_idx != len(pcd_dir_list):  # img_list_idx从0开始，len()从1开始
                print("list长度不符", len(pcd_dir_list))
            full_lidar_map_index=torch.zeros(len(pcd_file_list))
            """读取点云全局地图"""
            full_lidar_map_dir = os.path.join(self.Parameter_path, "map/Vector")
            full_lidar_map_dir = os.path.join(full_lidar_map_dir, str(self.data_file_list[0]))
            full_lidar_map_dir = full_lidar_map_dir + ".pcd"
            if os.path.exists(full_lidar_map_dir) != True:
                print("读取点云错误")
            full_LiDAR_pts = o3d.io.read_point_cloud(full_lidar_map_dir)
            full_LiDAR_map.append(np.asarray(full_LiDAR_pts.points))

        self.pcd_dir_list = pcd_dir_list
        self.pcd_ts = pcd_ts
        self.each_frame_dataset_index = full_lidar_map_index
        self.full_LiDAR_map = full_LiDAR_map
        """点云地图读取"""
        """单帧点云读取"""
        # pcd_index=torch.searchsorted(self.pcd_ts,pcd_ts[5])#pcd_ts[5]替换为目标时间戳
        # curr_scan_pts = o3d.io.read_point_cloud(self.pcd_dir_list[pcd_index])#单帧点云读取
        # single_LiDAR_frame = np.asarray(curr_scan_pts.points)#单帧点云
        #
        # dataset_index=int(self.each_frame_dataset_index[pcd_index])
        # full_lidar_map_dir = os.path.join(self.Parameter_path,"map/Vector")
        # full_lidar_map_dir=os.path.join(full_lidar_map_dir,str(self.data_file_list[dataset_index]))
        # full_lidar_map_dir=full_lidar_map_dir+".pcd"
        # if os.path.exists(full_lidar_map_dir)!=True:
        #     print("读取点云错误")
        # self.full_LiDAR_pts = o3d.io.read_point_cloud(full_lidar_map_dir)
        # full_LiDAR_map = np.asarray(self.full_LiDAR_pts.points)
        # self.full_LiDAR_map=full_LiDAR_map

        print("读取点云完成")
        """读取点云完成"""



        """开始读取位姿"""
        if len(self.data_file_list)>1:
            flag=0
            for data_file in self.data_file_list:
                pose_dir = os.path.join(self.Data_path, data_file)
                pose_dir=os.path.join(pose_dir, data_file)
                pose_dir=pose_dir+"1.synced.gt.txt"
                # test=os.path.exists(pose_dir)
                if flag<1:#首次循环的操作
                    with open(pose_dir, "r") as pose_file:#先统计总的行数
                        pose_num = len(pose_file.readlines())
                        pose_num = pose_num - 2
                        pose_ts = torch.zeros(pose_num)
                        pose_ts = pose_ts.to(dtype=torch.float64)
                        pose_dict = torch.zeros((pose_num, 7))
                        pose_dict = pose_dict.to(dtype=torch.float64)
                    pose_index = 0
                    with open(pose_dir, "r") as pose_file:
                        for line in pose_file.readlines(pose_index):
                            if pose_index < 2:
                                pose_index += 1
                                continue  # 前两行是数据格式备注，跳过
                            tokens = line.split(" ")
                            pose_ts[pose_index - 2] = float(tokens[0])  # 前两条跳过，这里要扣减
                            pose = tokens[1:]
                            pose_dict[pose_index - 2, 0] = float(pose[0])
                            pose_dict[pose_index - 2, 1] = float(pose[1])
                            pose_dict[pose_index - 2, 2] = float(pose[2])
                            pose_dict[pose_index - 2, 3] = float(pose[3])
                            pose_dict[pose_index - 2, 4] = float(pose[4])
                            pose_dict[pose_index - 2, 5] = float(pose[5])
                            pose_dict[pose_index - 2, 6] = float(pose[6])
                            pose_index += 1
                    flag+=1
                else:#循环1以上时的操作
                    with open(pose_dir, "r") as pose_file:#先统计总的行数
                        pose_num_temp = len(pose_file.readlines())
                        pose_num_temp = pose_num_temp - 2
                        pose_ts_temp = torch.zeros(pose_num_temp)
                        pose_ts_temp = pose_ts_temp.to(dtype=torch.float64)
                        pose_dict_temp = torch.zeros((pose_num_temp, 7))
                        pose_dict_temp = pose_dict_temp.to(dtype=torch.float64)
                    pose_index = 0
                    with open(pose_dir, "r") as pose_file:
                        for line in pose_file.readlines(pose_index):
                            if pose_index < 2:
                                pose_index += 1
                                continue  # 前两行是数据格式备注，跳过
                            tokens = line.split(" ")
                            pose_ts_temp[pose_index - 2] = float(tokens[0])  # 前两条跳过，这里要扣减
                            pose = tokens[1:]
                            pose_dict_temp[pose_index - 2, 0] = float(pose[0])
                            pose_dict_temp[pose_index - 2, 1] = float(pose[1])
                            pose_dict_temp[pose_index - 2, 2] = float(pose[2])
                            pose_dict_temp[pose_index - 2, 3] = float(pose[3])
                            pose_dict_temp[pose_index - 2, 4] = float(pose[4])
                            pose_dict_temp[pose_index - 2, 5] = float(pose[5])
                            pose_dict_temp[pose_index - 2, 6] = float(pose[6])
                            pose_index += 1
                    pose_num=pose_num+pose_num_temp
                    pose_ts=torch.cat((pose_ts,pose_ts_temp),dim=0)
                    pose_dict=torch.cat((pose_dict,pose_dict_temp),dim=0)
                    flag += 1
            if pose_num != self.pcd_ts.shape[0]:
                print("pose与pcd_ts长度不一致", pose_num, self.pcd_ts.shape[0])

        else:#list为1时的基础操作
            pose_dir = os.path.join(self.Data_path, self.data_file_list[0])
            pose_dir = os.path.join(pose_dir, self.data_file_list[0])
            pose_dir = pose_dir + "1.synced.gt.txt"
            with open(pose_dir, "r") as pose_file:  # 先统计总的行数
                pose_num = len(pose_file.readlines())
                pose_num = pose_num - 2
                pose_ts = torch.zeros(pose_num)
                pose_ts = pose_ts.to(dtype=torch.float64)
                pose_dict = torch.zeros((pose_num, 7))
                pose_dict = pose_dict.to(dtype=torch.float64)
            pose_index = 0
            with open(pose_dir, "r") as pose_file:
                for line in pose_file.readlines(pose_index):
                    if pose_index < 2:
                        pose_index += 1
                        continue  # 前两行是数据格式备注，跳过
                    tokens = line.split(" ")
                    pose_ts[pose_index - 2] = float(tokens[0])  # 前两条跳过，这里要扣减
                    pose = tokens[1:]
                    pose_dict[pose_index - 2, 0] = float(pose[0])
                    pose_dict[pose_index - 2, 1] = float(pose[1])
                    pose_dict[pose_index - 2, 2] = float(pose[2])
                    pose_dict[pose_index - 2, 3] = float(pose[3])
                    pose_dict[pose_index - 2, 4] = float(pose[4])
                    pose_dict[pose_index - 2, 5] = float(pose[5])
                    pose_dict[pose_index - 2, 6] = float(pose[6])
                    pose_index += 1
        self.pose_ts=pose_ts
        self.pose=pose_dict
        """读取特定位姿"""
        # frame_pose_index=5#pose的index
        # lidar_pose = self.pose[frame_pose_index]  # 世界坐标系下相机的位姿Pose，也是相机坐标系转为世界坐标系的矩阵
        # lidar_pose = seven_to_pose(lidar_pose.numpy())
        # cam_pose = np.matmul(self.VECtor_Projection_model.T_cam0_lidar, np.linalg.inv(lidar_pose))  # T_cam0_lidar*lidar_pose
        # DVS_pose = np.matmul(np.linalg.inv(self.VECtor_Projection_model.T_cam0_camDVS), cam_pose)

        print("读取位姿完成")
        """读取位姿完成"""

        """完成了所有内容的加载"""
        self.loaded = True

    def __len__(self):  # 这里以pose的数量为基准
        if self.stack_mode == "SBT":  # 按照固定时间间隔
            # 每个事件都可以作为起点（之后加一个事件数量的筛选），尾部空出2帧
            # events_num=self.events.shape[0]
            end_time = self.events[-1, 2] - 3 * self.SBT_time
            # self.events_t = self.events[:, 2]
            end_pose_index = np.searchsorted(self.pose_ts, end_time, side='left')
            return end_pose_index
        else:  # 不是SBT就是SBN（按照固定事件数量累积）
            # 选定一个事件作为起点，直接加上固定数量就可以
            events_num = self.events.shape[0]
            last_valid_event = events_num - 3 * self.SBN_num
            end_time = self.events[last_valid_event, 2]
            end_pose_index = np.searchsorted(self.pose_ts, end_time, side='left')

            return end_pose_index

    def get_img_by_image_index(self,image_index):
        """读取特定图像"""
        img_dir=self.image_dir_list[image_index]
        img = cv2.imread(img_dir, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # print(img.shape)  # numpy数组格式为（H,W,C）
        # img_tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
        # print(img_tensor.size())
        # cv2.imshow("test_image", img)
        # cv2.waitKey(0)

        # input_tensor = img_tensor.clone().detach().to(torch.device('cpu'))  # 到cpu
        return img

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

    # 从大图中随机裁切一定大小的小图（抖动，数据增强）像素位置索引
    def get_box(self,pic_mode):
        top_left = self.top_left

        if pic_mode=="DVS":
            DVS_boundary=70
            top=np.random.randint(low=DVS_boundary, high=self.raw_DVS_height-DVS_boundary- 1 - self.target_DVS_image_height, dtype='int32')
            left = np.random.randint(low=DVS_boundary,high=self.raw_DVS_width - DVS_boundary - 1 - self.target_DVS_image_width,dtype='int32')
            # top = int(np.random.rand() * (self.raw_DVS_height - 1 - self.target_DVS_image_height))
            # left = int(np.random.rand() * (self.raw_DVS_width - 1 - self.target_DVS_image_width))
            top_left = [top, left]
            bottom_right = [top_left[0] + self.target_DVS_image_height,
                            top_left[1] + self.target_DVS_image_width]
        elif pic_mode=="IMG":#IMG
            top = int(np.random.rand() * (self.raw_IMG_height - 1 - self.target_IMG_image_height))
            left = int(np.random.rand() * (self.raw_IMG_width - 1 - self.target_IMG_image_width))
            top_left = [top, left]
            bottom_right = [top_left[0] + self.target_IMG_image_height,
                            top_left[1] + self.target_IMG_image_width]



        return top_left, bottom_right


    def get_image_by_event_index(self, evind, bbox):  # SBN/SBT中ind是events序号
        top_left, bottom_right = bbox
        img_index = np.searchsorted(self.image_ts, self.events_t[evind])
        full_image=self.get_img_by_image_index(img_index)
        image = full_image[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]

        # image = image.transpose((2, 0, 1))  # （256,256,1）到（1,256,256）
        image = image.astype(np.float32) / 255.  # 转换成0~1


        cv_show_prev_image = full_image.astype(np.float32)/255
        # cv_show_prev_image=cv_show_prev_image.transpose((2, 0, 1)).astype(np.float32) / 255.
        # cv_show_prev_image-=0.5
        # cv_show_prev_image *= 2.
        # cv2.imshow("prev_image", cv_show_prev_image)
        # cv2.waitKey(0)
        # cv2.imwrite("prev_image"+str(ind)+".bmp", cv_show_prev_image)
        return image,cv_show_prev_image


    # 按照给的图像序号和mask位置，取出对应位置和跨越这几帧的事件
    def get_events(self, pind, cind, bbox):
        # 原来是按照固定间隔取，取完再截取，SBN模式下严格的说要改

        top_left, bottom_right = bbox
        # top_left=[70,70]
        # bottom_right=[self.raw_DVS_height-70,self.raw_DVS_width-70]

        peind = pind
        ceind = cind

        events = self.events[peind:ceind, :]
        # event2image_MVSEC(events, [self.raw_DVS_height, self.raw_DVS_width], start_index=0, end_index=-1)
        mask = np.logical_and(np.logical_and(events[:, 1] >= top_left[0],
                                             events[:, 1] < bottom_right[0]),
                              np.logical_and(events[:, 0] >= top_left[1],
                                             events[:, 0] < bottom_right[1]))  # 获取位置范围内的事件

        events_masked = events[mask]
        # event2image_MVSEC(events_masked, [self.raw_DVS_height, self.raw_DVS_width], start_index=0, end_index=-1)
        if events_masked.shape[0]<=1:
            return [],[]
        # events原始是w*h*t*p

        # 对取出的事件xy坐标进行偏移校正
        events_shifted = events_masked
        events_shifted[:, 0] = events_masked[:, 0] - top_left[1]
        events_shifted[:, 1] = events_masked[:, 1] - top_left[0]
        # events_shifted[:, 2] -= np.min(events_shifted[:, 2])  # 时间归一化
        time_shift=events_shifted[0, 2]
        events_shifted[:, 2] = events_shifted[:, 2]- time_shift # 时间归一化
        events_shifted[:, 3]= events_shifted[:, 3]*2-1

        # convolution expects 4xN
        # events_shifted = np.transpose(events_shifted).astype(np.float32)
        events_shifted = events_shifted.astype(np.float32)


        events_full=events

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

    def enlarge_projection_field(self,bbox):
        if self.corresponding_mode== "DVS":
            enlarge_width_pixel=int(self.Enlarged_projection_field * self.target_DVS_image_width)
            enlarge_height_pixel = int(self.Enlarged_projection_field * self.target_DVS_image_height)
            enlarge_bbox=copy.deepcopy(bbox)
            enlarge_bbox[0][0]-=enlarge_height_pixel
            enlarge_bbox[0][1] -= enlarge_width_pixel
            enlarge_bbox[1][0] += enlarge_height_pixel
            enlarge_bbox[1][1] += enlarge_width_pixel
        else :
            enlarge_width_pixel = int(self.Enlarged_projection_field * self.target_IMG_image_width)
            enlarge_height_pixel = int(self.Enlarged_projection_field * self.target_IMG_image_height)
            enlarge_bbox=copy.deepcopy(bbox)
            enlarge_bbox[0][0] -= enlarge_height_pixel
            enlarge_bbox[0][1] -= enlarge_width_pixel
            enlarge_bbox[1][0] += enlarge_height_pixel
            enlarge_bbox[1][1] += enlarge_width_pixel

        return enlarge_bbox





    # 取数据的主函数
    def get_single_item(self, getdata_pose_index):#取出第ind笔(位姿)对应的数据
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
                    frame_start_event_index = np.searchsorted(self.events_t, frame_start_time, side='left')
                    frame_end_event_index = np.searchsorted(self.events_t, frame_end_time, side='left')
                    n_events = frame_end_event_index - frame_start_event_index # self.count_events(frame_start_event_index, frame_end_event_index,self.frame_interval)  # 计算当前帧范围内事件总数
                    n_iters += 1
                    final_frame_interval = self.SBT_time#取出的时间流时间长度

                    # 由于事件相机镜头存在边缘光晕，要滤除
                    flag = False
                    try_count = 0
                    target_area = self.target_DVS_image_height * self.target_DVS_image_width
                    full_area = self.raw_DVS_width * self.raw_DVS_height
                    self.target_events_num = 0.6 * (target_area / full_area) * (
                            frame_end_event_index - frame_start_event_index)
                    while flag == False:
                        DVS_bbox = self.get_box("DVS")  # 获取小图片的位置索引
                        # 这里不管什么模式，都是要取DVS的数据
                        events_flow, events_flow_full = self.get_events(frame_start_event_index, frame_end_event_index,
                                                                        DVS_bbox)
                        if try_count >= 6:
                            # getdata_pose_index = random.randint(0, self.__len__())
                            # print("应该重新选取data_pose_index")
                            break
                        if len(events_flow) < self.target_events_num:
                            try_count += 1
                            # print("重新取DVS_box")
                            continue
                        # 取出对应位置和跨越这几帧的事件(事件坐标进行了偏移，按照截取的范围为基准)
                        if events_flow.shape[0] >= self.target_events_num:
                            flag = True

                    if n_events < self.min_n_events  or flag == False:  # 如果固定时间取到的帧事件数量太小，则跳过，随机再取一个帧
                        getdata_pose_index = random.randint(0, self.__len__())#重新给一个序号生成
                        n_events = -1
            elif self.stack_mode == "SBN" :  # "SBN":#按照固定事件数量取
                frame_interval = 99999 * self.SBT_time#设置一个较大的初始值
                while frame_interval > self.max_skip_frames * self.SBT_time:
                    frame_pose_time = self.pose_ts[getdata_pose_index]
                    frame_mid_event_index =np.searchsorted(self.events_t, frame_pose_time, side='left')
                    frame_start_event_index=frame_mid_event_index-self.SBN_num//2
                    frame_end_event_index =frame_mid_event_index+self.SBN_num//2

                    frame_start_time = self.events_t[frame_start_event_index]
                    frame_end_time = self.events_t[frame_end_event_index]
                    frame_interval = frame_end_time - frame_start_time
                    final_frame_interval = frame_interval

                    # 由于事件相机镜头存在边缘光晕，要滤除
                    flag = False
                    try_count = 0
                    target_area = self.target_DVS_image_height * self.target_DVS_image_width
                    full_area = self.raw_DVS_width * self.raw_DVS_height
                    self.target_events_num = 0.6 * (target_area / full_area) * (
                            frame_end_event_index - frame_start_event_index)
                    while flag == False:
                        DVS_bbox = self.get_box("DVS")  # 获取小图片的位置索引
                        # 这里不管什么模式，都是要取DVS的数据
                        events_flow, events_flow_full = self.get_events(frame_start_event_index, frame_end_event_index,
                                                                        DVS_bbox)
                        if try_count >= 6:
                            # getdata_pose_index = random.randint(0, self.__len__())
                            # print("应该重新选取data_pose_index")
                            break
                        if len(events_flow) < self.target_events_num:
                            try_count += 1
                            # print("重新取DVS_box")
                            continue
                        # 取出对应位置和跨越这几帧的事件(事件坐标进行了偏移，按照截取的范围为基准)
                        if events_flow.shape[0] >= self.target_events_num:
                            flag = True

                    if frame_interval > self.max_skip_frames * self.SBT_time \
                            or frame_start_event_index<0 \
                            or flag == False:  # 如果间隔太大，证明这段事件太稀疏
                        # test=self.__len__()[0]
                        getdata_pose_index = random.randint(0, int(self.__len__()))
                        print("重新取data_pose_index")
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
                    frame_start_event_index = np.searchsorted(self.events_t, frame_start_time, side='left')
                    frame_end_event_index = np.searchsorted(self.events_t, frame_end_time, side='left')
                    n_events = frame_end_event_index - frame_start_event_index  # self.count_events(frame_start_event_index, frame_end_event_index,self.frame_interval)  # 计算当前帧范围内事件总数
                    n_iters += 1
                    final_frame_interval = self.SBT_time  # 取出的时间流时间长度

                    # 由于事件相机镜头存在边缘光晕，要滤除
                    flag = False
                    try_count = 0
                    target_area = self.target_DVS_image_height * self.target_DVS_image_width
                    full_area = self.raw_DVS_width * self.raw_DVS_height
                    self.target_events_num = 0.6 * (target_area / full_area) * (
                            frame_end_event_index - frame_start_event_index)
                    while flag == False:
                        DVS_bbox = self.get_box("DVS")  # 获取小图片的位置索引
                        # 这里不管什么模式，都是要取DVS的数据
                        events_flow, events_flow_full = self.get_events(frame_start_event_index, frame_end_event_index,
                                                                        DVS_bbox)
                        if try_count >= 6:
                            # getdata_pose_index = random.randint(0, self.__len__())
                            # print("应该重新选取data_pose_index")
                            break
                        if len(events_flow) < self.target_events_num:
                            try_count += 1
                            # print("重新取DVS_box")
                            continue
                        # 取出对应位置和跨越这几帧的事件(事件坐标进行了偏移，按照截取的范围为基准)
                        if events_flow.shape[0] >= self.target_events_num:
                            flag = True

                    if n_events < self.min_n_events  or flag==False:  # 如果固定时间取到的帧事件数量太小，则跳过，随机再取一个帧
                        getdata_pose_index = random.randint(0, self.__len__())  # 重新给一个序号生成
                        n_events = -1
            elif self.stack_mode == "SBN":  # "SBN":#按照固定事件数量取
                frame_interval = 99999 * self.SBT_time  # 设置一个较大的初始值
                while frame_interval > self.max_skip_frames * self.SBT_time:
                    frame_pose_time = self.pose_ts[getdata_pose_index]
                    frame_mid_event_index = np.searchsorted(self.events_t, frame_pose_time, side='left')
                    frame_start_event_index = frame_mid_event_index - self.SBN_num // 2
                    frame_end_event_index = frame_mid_event_index + self.SBN_num // 2

                    frame_start_time = self.events_t[frame_start_event_index]
                    frame_end_time = self.events_t[frame_end_event_index]
                    frame_interval = frame_end_time - frame_start_time
                    final_frame_interval = frame_interval

                    # 由于事件相机镜头存在边缘光晕，要滤除
                    flag = False
                    try_count = 0
                    target_area = self.target_DVS_image_height * self.target_DVS_image_width
                    full_area = self.raw_DVS_width * self.raw_DVS_height
                    self.target_events_num = 0.6 * (target_area / full_area) * (
                                frame_end_event_index - frame_start_event_index)
                    while flag == False:
                        DVS_bbox = self.get_box("DVS")  # 获取小图片的位置索引
                        # 这里不管什么模式，都是要取DVS的数据
                        events_flow, events_flow_full = self.get_events(frame_start_event_index, frame_end_event_index,
                                                                        DVS_bbox)
                        if try_count >= 6:
                            # getdata_pose_index = random.randint(0, self.__len__())
                            # print("应该重新选取data_pose_index")
                            break
                        if len(events_flow) < self.target_events_num:
                            try_count += 1
                            # print("重新取DVS_box")
                            continue
                        # 取出对应位置和跨越这几帧的事件(事件坐标进行了偏移，按照截取的范围为基准)
                        if events_flow.shape[0] >= self.target_events_num:
                            flag = True

                    if frame_interval > self.max_skip_frames * self.SBT_time \
                            or frame_start_event_index<0\
                            or flag==False:  # 如果间隔太大，证明这段事件太稀疏
                        getdata_pose_index = random.randint(0, self.__len__())
                        frame_interval = 99999 * self.SBT_time
                # 如果固定的数量覆盖的时间太长就舍弃
            else:
                print("stack_mode 出错")

        """由于事件相机镜头存在边缘光晕，要滤除
        flag=False
        try_count=0
        target_area = self.target_DVS_image_height * self.target_DVS_image_width
        full_area = self.raw_DVS_width * self.raw_DVS_height
        self.target_events_num = 0.6 * (target_area / full_area) * (frame_end_event_index - frame_start_event_index)
        while flag==False:
            DVS_bbox = self.get_box("DVS")  # 获取小图片的位置索引
            #这里不管什么模式，都是要取DVS的数据
            events_flow, events_flow_full = self.get_events(frame_start_event_index, frame_end_event_index,
                                                                DVS_bbox)
            if try_count >= 5:
                # getdata_pose_index = random.randint(0, self.__len__())
                print("应该重新选取data_pose_index")
                event2image_MVSEC(self.events[frame_start_event_index:frame_end_event_index, :], [self.raw_DVS_height, self.raw_DVS_width],
                                  start_index=0, end_index=-1, show_image=True)
                event2image_MVSEC(events_flow, [self.raw_DVS_height, self.raw_DVS_width],
                                  start_index=0, end_index=-1,show_image=True)
                break
            if len(events_flow)<self.target_events_num:
                try_count += 1
                print("重新取DVS_box")
                continue
            # 取出对应位置和跨越这几帧的事件(事件坐标进行了偏移，按照截取的范围为基准)
            if events_flow.shape[0]>=self.target_events_num :
                flag=True
        由于事件相机镜头存在边缘光晕，要滤除"""

        IMG_bbox = self.get_box("IMG")
        frame_mid_event_index = np.searchsorted(self.events_t, frame_pose_time)
        mid_image,mid_full_image = self.get_image_by_event_index(frame_mid_event_index, IMG_bbox)
        #上面的图像有除255

        # start_time = time.time()
        # elapsed = (time.time() - start_time)
        # print("GT用时:", elapsed)


        if self.train:
            # 此模式下不太需要这个增强，SBN可能不应进行这个增强
            events_flow = self.random_dropout_events(events_flow, self.dropout_ratio)


        event_volume = gen_discretized_event_volume(torch.from_numpy(events_flow).cpu(),
                                                    # [self.n_time_bins * 2,  # n_time_bins：体素的片数（正负分开处理）
                                                    [self.n_time_bins * 2,
                                                     self.target_DVS_image_height,  #
                                                     self.target_DVS_image_width])

        event_stacking_images = self.stacking_events(events_flow,
                                                     [self.Stacking_num,  # n_time_bins：体素的片数（正负累加处理）
                                                      self.target_DVS_image_height,
                                                      self.target_DVS_image_width])

        # 上面进行事件投影，获得事件正负独立，体素化离散投影的结果
        # 注意：1~N是正事件，N+1~2N是负事件体素。所以1和N+1是一组，N与2N是一组
        # 计算事件的数量累加图像
        # e_img=event2image(torch.from_numpy(events_RGB_frame).cpu())#可视化
        event_count_images = self.event_counting_images(torch.from_numpy(events_flow).cpu(),
                                                        [2,  # n_time_bins：体素的片数（正负分开处理）
                                                         self.target_DVS_image_height,  # W，H
                                                         self.target_DVS_image_width])

        event_time_images = self.event_timing_images(torch.from_numpy(events_flow).cpu(),
                                                     [2,  # n_time_bins：体素的片数（正负分开处理）
                                                      self.target_DVS_image_height,  # W，H
                                                      self.target_DVS_image_width])

        if self.normalize_events:# 对事件累积图像的张量进行归一化（均值0，方差1）,加紧到0.98，丢弃前后2%的数值

            event_volume = self.normalize_event_volume(event_volume)
            event_count_images=self.normalize_event_volume(event_count_images)#新增，非时间统计特征图都可使用
            event_stacking_images=self.normalize_event_volume(event_stacking_images)#新增，非时间统计特征图都可使用


        """下面是计算mask，有事件的位置为true"""
        # 计算位置掩码（出现事件的位置）
        WIDTH = self.target_DVS_image_width  # 阵面宽
        HEIGHT = self.target_DVS_image_height  # 阵面高
        events_num = events_flow.shape[0]

        # events_RGB_frame = event2image(torch.from_numpy(events_RGB_frame))
        event_mask = torch.zeros(HEIGHT, WIDTH, dtype=torch.int8)
        fill = torch.ones(events_num, dtype=torch.int8)
        index = (torch.from_numpy(events_flow[:, 1]).long(), torch.from_numpy(events_flow[:, 0]).long())
        event_mask = event_mask.index_put(index, fill)


        # show_mask = event_mask.float().numpy()
        # cv2.imshow("event_mask", show_mask)
        # cv2.waitKey(0)

        # 训练则进行数据增强(翻转)

        if self.train:
            # 按照一定的概率翻转图像（事件叠加图，以及对应GT图左右或者上下翻转），数据增强
            # 这里是对图像进行增强，光流是否能翻转
            if np.random.rand() < self.flip_x:
                event_mask = torch.flip(event_mask, dims=[1])
                event_volume = torch.flip(event_volume, dims=[2])
                # prev_image = np.flip(prev_image, axis=2)
                mid_image = np.flip(mid_image, axis=1)
                # next_image = np.flip(next_image, axis=2)
                # flow_frame = np.flip(flow_frame, axis=2)  # 对光流的翻转可能存在问题，要试试看
                # flow_frame[0] = -flow_frame[0]  # 翻转后光流左右相反，数值取反
                event_count_images = torch.flip(event_count_images, dims=[2])
                event_time_images = torch.flip(event_time_images, dims=[2])
                event_stacking_images = torch.flip(event_stacking_images, dims=[2])
                # f_img = flow2image(flow_frame)
            if np.random.rand() < self.flip_y:
                event_mask = torch.flip(event_mask, dims=[0])
                event_volume = torch.flip(event_volume, dims=[1])
                # prev_image = np.flip(prev_image, axis=1)
                mid_image = np.flip(mid_image, axis=0)
                # next_image = np.flip(next_image, axis=1)
                # flow_frame = np.flip(flow_frame, axis=1)
                # flow_frame[1] = -flow_frame[1]  # 翻转后光流上下相反，数值取反
                event_count_images = torch.flip(event_count_images, dims=[1])
                event_time_images = torch.flip(event_time_images, dims=[1])
                event_stacking_images = torch.flip(event_stacking_images, dims=[1])
            # prev_image_gt, next_image_gt = prev_image, next_image
            if self.appearance_augmentation:
                # prev_imag = self.apply_illum_augmentation(prev_image)
                mid_image = self.apply_illum_augmentation(mid_image)
                # next_image = self.apply_illum_augmentation(next_image)


        """开始取对应点云"""
        pcd_index = torch.searchsorted(self.pcd_ts, frame_pose_time)  # pcd_ts[5]替换为目标时间戳
        # curr_scan_pts = o3d.io.read_point_cloud(self.pcd_dir_list[pcd_index])  # 单帧点云读取
        # single_LiDAR_frame = np.asarray(curr_scan_pts.points)  # 单帧点云
        curr_lidar_map=int(self.each_frame_dataset_index[pcd_index])
        """读取对应点云完成"""

        lidar_pose = self.pose[getdata_pose_index]  # 世界坐标系下相机的位姿Pose，也是相机坐标系转为世界坐标系的矩阵
        lidar_pose = seven_to_pose(lidar_pose.numpy())
        CAM_pose = np.matmul(self.VECtor_Projection_model.T_cam0_lidar, np.linalg.inv(lidar_pose))  # T_cam0_lidar*lidar_pose
        DVS_pose = np.matmul(np.linalg.inv(self.VECtor_Projection_model.T_cam0_camDVS), CAM_pose)#将点云转到相机坐标系下

        cam_pose_R = CAM_pose[0:3, 0:3]  # 相机位姿RT
        cam_pose_T = CAM_pose[0:3, 3]

        DVS_pose_R = DVS_pose[0:3, 0:3]  # 事件相机位姿RT
        DVS_pose_T = DVS_pose[0:3, 3]


        events_RGB_frame = event2image_MVSEC(events_flow, [self.target_DVS_image_height, self.target_DVS_image_width])
        events_full_RGB_frame = event2image_MVSEC(events_flow_full, [self.raw_DVS_height, self.raw_DVS_width])


        """测试投影前的点云坐标
        test_R = torch.from_numpy(DVS_pose)
        test_R = test_R.expand(self.full_LiDAR_map.shape[0], -1, -1)
        test_full_lidar = torch.from_numpy(self.full_LiDAR_map)
        test_full_lidar = torch.cat((test_full_lidar, torch.ones(test_full_lidar.shape[0], 1)), dim=1)
        test_full_lidar = torch.unsqueeze(test_full_lidar, dim=2)
        test_full_lidar_after = torch.bmm(test_R, test_full_lidar)
        测试投影前的点云坐标完成"""

        """进行过滤与投影筛选"""
        if self.corresponding_mode== "DVS":
            # start_time = time.time()
            DVS_filted_points=self.VECtor_Projection_model.project_point_filtering( self.full_LiDAR_map[curr_lidar_map], DVS_pose, DVS_bbox ,10,"DVS")

            # elapsed = (time.time() - start_time)
            # print("四棱柱截取用时:", elapsed)
            # start_time = time.time()

            # DVS_filted_points = self.VECtor_Projection_model.Back_point_filtering(DVS_filted_points, DVS_pose,
            #                                                                       15)  # 事件相机投影方向半球+距离过滤

            # elapsed = (time.time() - start_time)
            # print("半球距离过滤用时:", elapsed, " ind=", getdata_pose_index)
            # start_time = time.time()


            DVS_filted_points = remove_hidden_point(DVS_filted_points, DVS_pose)  # 隐藏点去除，效果待测试(0.3s)
            # elapsed = (time.time() - start_time)
            # print("隐藏点移除用时:", elapsed, " ind=", getdata_pose_index)


            enlarged_DVS_bbox = self.enlarge_projection_field(DVS_bbox)
            if self.Enlarge_projection==True:
                projected2DVS_point, enlarged_DVS_flag_points = self.VECtor_Projection_model.Point2Pic("DVS", DVS_filted_points,
                                                                                          DVS_pose_R, DVS_pose_T,
                                                                                          enlarged_DVS_bbox)  # 点云投影到事件相机
                enlarged_valid_points = enlarged_DVS_flag_points[enlarged_DVS_flag_points[:, 3] > 0]
                projected2DVS_point, DVS_flag_points = self.VECtor_Projection_model.Point2Pic("DVS", enlarged_valid_points,
                                                                                              DVS_pose_R, DVS_pose_T,
                                                                                              DVS_bbox)  # 点云投影到事件相机
            else:
                projected2DVS_point, DVS_flag_points = self.VECtor_Projection_model.Point2Pic("DVS", DVS_filted_points,
                                                                                              DVS_pose_R, DVS_pose_T,
                                                                                              DVS_bbox)  # 点云投影到事件相机
                enlarged_valid_points=np.empty(DVS_flag_points[0,0:3])

            # elapsed = (time.time() - start_time)
            # print("点云投影用时:", elapsed, " ind=", getdata_pose_index)
            # start_time = time.time()

            valid_points = DVS_flag_points[DVS_flag_points[:, 3] > 0]
            cam_pose=DVS_pose
            """测试DVS投影
            cv2.imshow("DVS_image_full", events_full_RGB_frame)
            cv2.waitKey(0)
            cv2.imshow("tradition_full_image", mid_full_image)
            cv2.waitKey(0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(DVS_filted_points[:, :3])  # 显示半球过滤后的全部点云
            o3d.visualization.draw_geometries([pcd])
            cv2.imshow("DVS_image_patch", events_RGB_frame)
            cv2.waitKey(0)

            # 校正patch坐标与全图坐标不一致
            compensated_point2d_from_lidar = copy.deepcopy(projected2DVS_point)# projected2IMG_point
            compensated_point2d_from_lidar[:, 0] = compensated_point2d_from_lidar[:, 0] - DVS_bbox[0][1]
            compensated_point2d_from_lidar[:, 1] = compensated_point2d_from_lidar[:, 1] - DVS_bbox[0][0]

            events_full_RGB_frame_with_point = self.VECtor_Projection_model.draw_point(events_full_RGB_frame, projected2DVS_point )
            # mid_image_with_point = self.MVSEC_Projection_model.draw_point(mid_image, point2d_from_lidar)
            events_RGB_frame_with_point = self.VECtor_Projection_model.draw_point(events_RGB_frame,
                                                                                  compensated_point2d_from_lidar)# 事件帧，draw_point会改变原始图
            # event_RGB_image_with_point = self.VECtor_Projection_model.draw_point(events_RGB_frame,projected2DVS_point) #投影点坐标是大图像内的，要校正
            cv2.imshow("events_full_RGB_frame_with_point", events_full_RGB_frame_with_point)
            cv2.waitKey(0)
            cv2.imshow("event_RGB_frame_with_point", events_RGB_frame_with_point)
            cv2.waitKey(0)
            valid_pcd = o3d.geometry.PointCloud()
            valid_pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])  # 显示参与有效投影的点云
            o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释
            测试DVS投影结束"""

        else:#输出传统图像IMG
            # start_time = time.time()
            CAM_filted_points = self.VECtor_Projection_model.project_point_filtering(self.full_LiDAR_map[curr_lidar_map], CAM_pose,
                                                                                     IMG_bbox, 10,"IMG")
            # elapsed = (time.time() - start_time)
            # print("四棱柱截取用时:", elapsed)


            # CAM_filted_points = self.VECtor_Projection_model.Back_point_filtering(CAM_filted_points, CAM_pose,
            #                                                                       15)  # 事件相机投影方向半球+距离过滤

            CAM_filted_points = remove_hidden_point(CAM_filted_points, CAM_pose)#隐藏点去除，效果有问题

            # start_time = time.time()
            enlarged_IMG_bbox = self.enlarge_projection_field(IMG_bbox)
            if self.Enlarge_projection == True:
                projected2IMG_point, enlarged_IMG_flag_points = self.VECtor_Projection_model.Point2Pic("cam0", CAM_filted_points,
                                                                                              cam_pose_R, cam_pose_T,
                                                                                              enlarged_IMG_bbox)  # 传统相机
                enlarged_valid_points = enlarged_IMG_flag_points[enlarged_IMG_flag_points[:, 3] > 0]
                projected2IMG_point, IMG_flag_points = self.VECtor_Projection_model.Point2Pic("cam0",
                                                                                              enlarged_valid_points,
                                                                                              cam_pose_R, cam_pose_T,
                                                                                              IMG_bbox)  # 点云投影到事件相机
            else:
                projected2IMG_point, IMG_flag_points = self.VECtor_Projection_model.Point2Pic("cam0", CAM_filted_points,
                                                                                              cam_pose_R, cam_pose_T,
                                                                                              IMG_bbox)  # 传统相机
                enlarged_valid_points = np.empty(IMG_flag_points[0, 0:3])
            # elapsed = (time.time() - start_time)
            # print("投影用时:", elapsed)

            valid_points = IMG_flag_points[IMG_flag_points[:, 3] > 0]
            cam_pose=CAM_pose



            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])  # 显示半球过滤后的全部点云
            # o3d.visualization.draw_geometries([pcd])

            """测试IMG投影
            cv2.imshow("DVS_image_full", events_full_RGB_frame)
            cv2.waitKey(0)
            cv2.imshow("tradition_full_image", mid_full_image)
            cv2.waitKey(0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(CAM_filted_points[:, :3])  # 显示半球过滤后的全部点云
            o3d.visualization.draw_geometries([pcd])
            cv2.imshow("CAM_image_patch", mid_image)
            cv2.waitKey(0)

            # 校正patch坐标与全图坐标不一致
            compensated_point2d_from_lidar = copy.deepcopy(projected2IMG_point)  # projected2IMG_point
            compensated_point2d_from_lidar[:, 0] = compensated_point2d_from_lidar[:, 0] - IMG_bbox[0][1]
            compensated_point2d_from_lidar[:, 1] = compensated_point2d_from_lidar[:, 1] - IMG_bbox[0][0]

            full_mid_image_with_point = self.VECtor_Projection_model.draw_point(mid_full_image, projected2IMG_point)
            image_with_point = self.VECtor_Projection_model.draw_point(mid_image,compensated_point2d_from_lidar )  # 图像帧
            cv2.imshow("full_mid_image_with_point", full_mid_image_with_point)
            cv2.waitKey(0)
            cv2.imshow("image_with_point", image_with_point)
            cv2.waitKey(0)
            valid_pcd = o3d.geometry.PointCloud()
            valid_pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])  # 显示参与有效投影的点云
            o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释
            测试IMG投影结束"""
        # 投影后2D点是【u，v】，也就是【w,h】

        """增加对取出的对应点云进行随机变换"""
        if self.train:
            random_RT = random_pose(self.random_max_angle, self.random_max_trans)
            random_RT = torch.from_numpy(random_RT)
            # P(2D)=P(C2L)P(3d)=P(C2L)R` RP(3d)
            valid_points = torch.from_numpy(valid_points)
            enhance_points = copy.deepcopy(valid_points)
            repat_pose = random_RT.expand(enhance_points.shape[0], -1, -1)
            enhance_points = torch.unsqueeze(enhance_points, dim=2)
            # R*P(3d)
            enhanced_points = torch.bmm(repat_pose, enhance_points)  # 将点云从雷达坐标系转换到相机坐标系
            enhanced_points = torch.squeeze(enhanced_points)

            if self.corresponding_mode == "DVS":
                enhanced_cam_pose = copy.deepcopy(DVS_pose)
                bbox=DVS_bbox
            else:
                enhanced_cam_pose = copy.deepcopy(CAM_pose)
                bbox = IMG_bbox
            random_RT = random_RT.numpy()
            # P(C2L)R`(R逆)
            enhanced_cam_pose = np.matmul(enhanced_cam_pose, np.linalg.inv(random_RT))

            valid_points = enhanced_points
            cam_pose = enhanced_cam_pose


            # cam_pose_R = enhanced_cam_pose[0:3, 0:3]  # 相机位姿RT
            # cam_pose_T = enhanced_cam_pose[0:3, 3]
            # point2d_from_lidar, flag_points = self.VECtor_Projection_model.Point2Pic("cam0",enhanced_points, cam_pose_R,
            #                                                                         cam_pose_T, bbox)
            # compensated_point2d_from_lidar=copy.deepcopy(point2d_from_lidar)
            # compensated_point2d_from_lidar[:,0]=compensated_point2d_from_lidar[:,0]-bbox[0][1]
            # compensated_point2d_from_lidar[:, 1]=compensated_point2d_from_lidar[:, 1] - bbox[0][0]
            # mid_image_with_point = self.VECtor_Projection_model.draw_point(mid_full_image, point2d_from_lidar)
            # events_RGB_frame_with_point = self.VECtor_Projection_model.draw_point(events_RGB_frame,compensated_point2d_from_lidar)
            # cv2.imshow("tradition_image", mid_image)#调试可视化
            # cv2.waitKey(0)
            # cv2.imshow("mid_image_with_point", mid_image_with_point)#调试可视化
            # cv2.waitKey(0)
            # cv2.imshow("events_RGB_frame_with_point", events_RGB_frame_with_point)#调试可视化
            # cv2.waitKey(0)
            # valid_pcd = o3d.geometry.PointCloud()
            # valid_pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])  # 显示参与有效投影的点云
            # o3d.visualization.draw_geometries([valid_pcd])#调试可视化
        """增加点云随机变换"""


        #潜在死循环隐患
        if valid_points.shape[0]<self.pcd_num:
            return self.get_single_item(random.randint(0, self.__len__()))
        sample_index=np.random.choice(valid_points.shape[0], self.pcd_num, replace=False)
        valid_points=valid_points[sample_index]#点云随机采样为固定数量

        """取对应点云完成"""
        gradient_x = cv2.Sobel(mid_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(mid_image, cv2.CV_64F, 0, 1, ksize=3)
        image_gradient_map = np.stack([gradient_x, gradient_y])
        mid_image = torch.from_numpy(mid_image.copy()).float()
        image_gradient_map = torch.from_numpy(image_gradient_map).float()
        image_gradient_map = torch.squeeze(image_gradient_map)
        events_RGB_frame = torch.from_numpy(events_RGB_frame)

        elapsed = (time.time() - start_time)
        # print("取数据用时:", elapsed," ind=",getdata_pose_index)



        output = {
            "mid_image": mid_image,#中间时刻的灰度图
            "event_volume": event_volume,  # 投影得到的事件体素（T*H*W）
            # "flow_frame": flow_frame,  # 光流GT值
            "event_mask": event_mask,  # 事件出现位置的掩码，后续算loss可能用得到(有事件的为1)
            "event_count_images": event_count_images,  # 根据各个像素位置累积的事件出现次数累积图像，前正后负
            "event_time_images": event_time_images,  # 按照最近（新）事件的时间戳生成的事件时间戳图像，前正后负
            "events_RGB_frame": events_RGB_frame,  # 事件序列可视化图像
            "events_num": events_num,  # 返回的事件序列有效长度(!!!!没有减去mask掉的位置)
            # "depth_frame": depth_frame,  # 增加深度帧
            # "depth_mask": depth_mask,  # 深度图的mask，已经过取反，true即有深度的位置，直接相乘即可
            "image_gradient": image_gradient_map,
            "event_stacking_images": event_stacking_images,  # 堆叠的图
            "valid_points":valid_points, #参与有效投影的点云(pic_mode为DVS/IMG分别为各自对应投影的点云，
            # IMG和DVS的输出分别有patch的大小参数 )
            # "DVS_filted_points":DVS_filted_points #投影半球方向内的所有点云
            "cam_pose": cam_pose,#输出相机的位姿，根据pic_mode的不同分别是DVS与传统相机的位姿
            "enlarged_valid_points": enlarged_valid_points
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
        # print(" __getitem__取了", frame_index)
        return self.get_single_item(frame_index)




if __name__ == '__main__':
    data = Data_VECtor("E:/事件匹配/Event_LiDAR_match/data",
                       "G:\VECtor_Benchmark\Large-scale", [1,2], train=True,
                       stack_mode="SBN",pic_mode="DVS")#SBN/SBT  DVS/IMG
    #注意，数据集序号【】需要先人工检查，确认时间戳由小到大
    test_len=data.__len__()

    t2 = data.__getitem__(233)
    # t3 = data.__getitem__(9607)#3166 1251
    # print(data.__len__())
    # print(data.__getitem__(5))
    # data = MyData(-1)

    from torch.utils.data import DataLoader

    loader = DataLoader(data, batch_size=2, drop_last=True,shuffle=True,num_workers=0)

    for i, batch in enumerate(loader):
        print(i)
        # print(i, batch['event_count_images'])








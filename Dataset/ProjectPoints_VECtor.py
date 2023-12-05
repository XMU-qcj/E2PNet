import cv2
import copy
import numpy as np
import torch
import yaml
import zipfile
import h5py
import hdf5plugin#必须保留
import tqdm

import torchvision
import open3d as o3d
import cv2
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
# from Event_Pretreatment import event_counting_images,event_timing_images
from result_visualization import  event2image_VECtor


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def inverse(self):
        qnorm = self.norm()
        return Quaternion(self.w / qnorm,
                          -self.x / qnorm,
                          -self.y / qnorm,
                          -self.z / qnorm)
    def rot_mat_from_quaternion(self):

        R = np.array([[1-2*self.y**2-2*self.z**2, 2*self.x*self.y+2*self.w*self.z, 2*self.x*self.z-2*self.w*self.y],
                      [2*self.x*self.y-2*self.w*self.z, 1-2*self.x**2-2*self.z**2, 2*self.y*self.z+2*self.w*self.x],
                      [2*self.x*self.z+2*self.w*self.y, 2*self.y*self.z-2*self.w*self.x, 1-2*self.x**2-2*self.y**2]])
        return R

    def quaternion2RotationMatrix(self):
        x, y, z, w = self.x, self.y, self.z, self.w
        rot_matrix00 = 1 - 2 * y * y - 2 * z * z
        rot_matrix01 = 2 * x * y - 2 * w * z
        rot_matrix02 = 2 * x * z + 2 * w * y
        rot_matrix10 = 2 * x * y + 2 * w * z
        rot_matrix11 = 1 - 2 * x * x - 2 * z * z
        rot_matrix12 = 2 * y * z - 2 * w * x
        rot_matrix20 = 2 * x * z - 2 * w * y
        rot_matrix21 = 2 * y * z + 2 * w * x
        rot_matrix22 = 1 - 2 * x * x - 2 * y * y
        return np.asarray([
            [rot_matrix00, rot_matrix01, rot_matrix02],
            [rot_matrix10, rot_matrix11, rot_matrix12],
            [rot_matrix20, rot_matrix21, rot_matrix22]
        ], dtype=np.float64)

    def __mul__(q1, q2):
        r = Quaternion(q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
                       q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
                       q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
                       q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w)
        return r

    def __rmul__(q1, s):
        return Quaternion(q1.w * s, q1.x * s, q1.y * s, q1.z * s)

    def __sub__(q1, q2):
        r = Quaternion(q1.w - q2.w,
                       q1.x - q2.x,
                       q1.y - q2.y,
                       q1.z - q2.z)
        return r

    def __div__(q1, s):
        return Quaternion(q1.w / s, q1.x / s, q1.y / s, q1.z / s)


def quat2mat_fast(quat):
    quat = quat.unsqueeze(0)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def seven_to_pose(pose):
    pose = torch.from_numpy(np.array(list(map(float, pose))))
    translation, rotation_q = pose[:3], pose[3:]
    rotation_mat = quat2mat_fast(rotation_q)
    transform = np.eye(4)
    transform[:3, 3] = translation
    transform[:3, :3] =rotation_mat

    # testQ=Quaternion(rotation_q[3],rotation_q[0],rotation_q[1],rotation_q[2])#

    return transform

def cal_plane_equation(vertex,point1,point2):
    assert vertex.shape[0]==3#点坐标要求为3维
    x1=vertex[0]
    x2=point1[0]
    x3=point2[0]
    y1 = vertex[1]
    y2 = point1[1]
    y3 = point2[1]
    z1 = vertex[2]
    z2 = point1[2]
    z3 = point2[2]

    testA=y1*(z2-z3)+y2*(z3-z1)+y3*(z1-z2)
    A = torch.det(torch.Tensor([ [1, y1, z1], [1, y2, z2], [1, y3, z3] ]))
    B = torch.det(torch.Tensor([[x1, 1, z1], [x2, 1, z2], [x3, 1, z3]]))
    C = torch.det(torch.Tensor([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]]))
    D = -1*torch.det(torch.Tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]))


    return A,B,C,D
class read_VEVtor():
    def __init__(self, Folde_Address):
        pass

def read_events(File_Address):
    VECtor_events = h5py.File(File_Address, 'r')
    # VECtor_events = h5py.File("G:\VECtor Benchmark\Large-scale\corridors-dolly/corridors_dolly1.synced.left_event.hdf5",
    #                           'r')
    events_x = VECtor_events['events/x'][:]
    events_y = VECtor_events['events/y'][:]
    events_t = VECtor_events['events/t'][:]  # 单位微秒，转到秒要/1e6
    events_p = VECtor_events['events/p'][:]
    events_ms_to_idx = VECtor_events['ms_to_idx'][:]  # 毫秒级别索引对应的事件序号
    events_t_offset = VECtor_events['t_offset'][:]  # 事件流的起始绝对时间，上面的t是相对时间，单位微秒10e-6
    # max_t=events_t[-1]
    # test=(events_t_offset+events_t[-1])
    # test2=test/1e6
    return events_x,events_y,events_t,events_p,events_ms_to_idx,events_t_offset


def read_img(IMG_Folde_Address):#从文件夹里读取
    """读取图像"""
    img_folde_dir =IMG_Folde_Address
    img_list = os.listdir(img_folde_dir)
    img_list.sort()

    img_ts = torch.zeros(len(img_list))
    img_ts = img_ts.to(dtype=torch.float64)

    img_list_idx = 0
    torch.set_printoptions(precision=6, sci_mode=False)
    for name in img_list:
        img_ts[img_list_idx] = float(name[:17])
        # print(pic_ts[name_list_idx])
        pic_dir = img_folde_dir + name
        pic_dir2 = img_folde_dir + img_list[img_list_idx]
        if pic_dir != pic_dir2:
            print("pic_dir!=pic_dir2:", pic_dir, pic_dir2)
        img_list_idx += 1

    img = cv2.imread(pic_dir2, 0)
    test_img = cv2.imread(
        "G:\VECtor Benchmark/Large-scale\corridors-dolly\corridors_dolly1.synced.left_camera/1643091527.996090.png", 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)  # numpy数组格式为（H,W,C）
    img_tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
    print(img_tensor.size())
    # input_tensor = img_tensor.clone().detach().to(torch.device('cpu'))  # 到cpu
    # torchvision.utils.save_image(input_tensor, "out_cv.jpg")
    """读取图像"""
    return img_ts



class Projection_model_VECtor:
    def __init__(self,Parameter_path,experiment_name):

        data_path=Parameter_path

        """读取左侧事件相机参数"""
        event_parameter_file=os.path.join(data_path,"left_event_camera_intrinsic_results.yaml")
        event_parameter=self.load_yaml(event_parameter_file)
        self.event_intrinsics=event_parameter['camera_matrix']['data']
        event_intrinsics=self.event_intrinsics
        self.event_CameraMatrix = np.array([[event_intrinsics[0], 0., event_intrinsics[2]],
                           [0., event_intrinsics[4], event_intrinsics[5]],
                           [0., 0., 1.]])
        self.DVS_K = self.event_CameraMatrix
        self.ZERO = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0.,0.,0 ,1.]])

        self.event_distortion_coeffs = np.array(event_parameter['distortion_coefficients']['data'])
        self.image_width = event_parameter['image_width']
        self.image_height = event_parameter['image_height']
        """读取左侧事件相机参数"""

        """读取左侧传统相机参数"""
        camera0_parameter_file = os.path.join(data_path, "left_regular_camera_intrinsic_results.yaml")
        camera0_parameter = self.load_yaml(camera0_parameter_file)

        self.camera0_intrinsics = camera0_parameter['camera_matrix']['data']
        camera0_intrinsics = self.camera0_intrinsics
        self.camera0_CameraMatrix = np.array([[camera0_intrinsics[0], 0., camera0_intrinsics[2]],
                                            [0., camera0_intrinsics[4], camera0_intrinsics[5]],
                                            [0., 0., 1.]])
        self.IMG_K = self.camera0_CameraMatrix
        self.camera0_distortion_coeffs = np.array(camera0_parameter['distortion_coefficients']['data'])
        """读取左侧传统相机参数"""

        """读取相机与激光雷达外参"""
        lidar_camera_file = os.path.join(data_path, "camera_lidar_extrinsic_results.yaml")
        lidar_camera_parameter = self.load_yaml(lidar_camera_file)
        T_cam0_lidar=lidar_camera_parameter['cam0']['T_cam_lidar']
        #transformation from LiDAR to left regular camera
        self.T_cam0_lidar=np.array([[T_cam0_lidar[0], T_cam0_lidar[1], T_cam0_lidar[2],T_cam0_lidar[3]],
                                    [T_cam0_lidar[4], T_cam0_lidar[5], T_cam0_lidar[6],T_cam0_lidar[7]],
                                    [T_cam0_lidar[8], T_cam0_lidar[9], T_cam0_lidar[10], T_cam0_lidar[11]],
                                    [T_cam0_lidar[12],T_cam0_lidar[13],T_cam0_lidar[14],T_cam0_lidar[15]]])
        """读取相机与激光雷达外参"""

        """读取相机与事件相机外参"""
        DVS_camera_file = os.path.join(data_path, "large_scale_joint_camera_extrinsic_results.yaml")
        DVS_camera_parameter = self.load_yaml(DVS_camera_file)
        # transformation from left event camera to left regular camera
        T_cam0_camDVS = DVS_camera_parameter['cam2']['T_cam0_cam2']
        self.T_cam0_camDVS=np.array([[T_cam0_camDVS[0], T_cam0_camDVS[1], T_cam0_camDVS[2],T_cam0_camDVS[3]],
                                [T_cam0_camDVS[4], T_cam0_camDVS[5], T_cam0_camDVS[6],T_cam0_camDVS[7]],
                                [T_cam0_camDVS[8], T_cam0_camDVS[9], T_cam0_camDVS[10], T_cam0_camDVS[11]],
                                [T_cam0_camDVS[12],T_cam0_camDVS[13],T_cam0_camDVS[14] ,T_cam0_camDVS[15]]])
        """读取相机与事件相机外参"""

        if experiment_name == "Large-scale":
            # 大尺度场景下GT是雷达的，DVS<-cam0<-雷达(GT)
            self.T_camDVS_lidar=np.matmul(np.linalg.inv(self.T_cam0_camDVS),self.T_cam0_lidar)
            self.T_camDVS_GT=self.T_camDVS_lidar
            pass

        elif experiment_name=="Small-scale":
            # 室内小尺度场景下GT是body的，DVS<-cam0<-body(GT)
            body_camera_file = os.path.join(data_path, "camera_mocap_extrinsic_results1.yaml")
            body_camera_parameter = self.load_yaml(body_camera_file)
            # transformation from left regular camera to body
            T_body_cam0 = body_camera_parameter['cam0']['T_body_cam']
            self.T_body_cam0 = np.array([[T_body_cam0[0], T_body_cam0[1], T_body_cam0[2], T_body_cam0[3]],
                                         [T_body_cam0[4], T_body_cam0[5], T_body_cam0[6], T_body_cam0[7]],
                                         [T_body_cam0[8], T_body_cam0[9], T_body_cam0[10], T_body_cam0[11]],
                                         [T_body_cam0[12],T_body_cam0[13], T_body_cam0[14], T_body_cam0[15]]])
            self.T_camDVS_body=np.matmul(np.linalg.inv(self.T_cam0_camDVS),np.linalg.inv(self.T_body_cam0))
            self.T_camDVS_GT=self.T_camDVS_body
        else:
            print("初始化时experiment_name出错")

        assert self.T_camDVS_GT[3][3]==1 and self.T_camDVS_GT[3][0]==0 and self.T_camDVS_GT[3][1]==0 and self.T_camDVS_GT[3][2]==0

        """初始化完成"""


    def load_yaml(self,yaml_file_dir):#experiment_name="indoor_flying"

        # yaml_path="E:/事件匹配/Event_LiDAR_match/utility/"+yaml_name
        with open(yaml_file_dir, 'r', encoding='utf-8') as yaml_file:
            intrinsic_extrinsic = yaml.load(yaml_file, Loader=yaml.SafeLoader)
            return intrinsic_extrinsic


    def Point2Pic(self, type,Points, R, T,bbox ):
        if Points.shape[1]==4:
            points = Points[:, 0: 3]  # 舍去最后1维度特征，保留前3个坐标
            flag_points = np.zeros_like(Points)
        elif Points.shape[1]==3:
            points=Points
            flag_points = np.zeros((Points.shape[0],4))
        else:
            print("点云格式有误")

        # flag_points = np.zeros_like(Points)

        # points=points.numpy()
        points=np.float32(points)
        # R=R.numpy()
        # T=T.numpy()
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if type=="cam0":
            imgpoints2, Jacobian = cv2.projectPoints(points, R, T, self.camera0_CameraMatrix, self.camera0_distortion_coeffs,
                                                     jacobian=None)#, self.distortion_coeffs
        elif type=="DVS":
            imgpoints2, Jacobian = cv2.projectPoints(points, R, T, self.event_CameraMatrix, self.event_distortion_coeffs,
                                                     jacobian=None)  # , self.distortion_coeffs
        imgpoints2=np.squeeze(imgpoints2)

        # imgpoints2[:0] = -imgpoints2[:0]
        # imgpoints2[:1] = -imgpoints2[:1]#正负取反
        top_left, bottom_right = bbox  # [h,w]
        flag_h_p = imgpoints2[:, 0] <= 1024#bottom_right[1]  # imgpoints2是[u,v],即[w,h]
        flag_h_n = imgpoints2[:, 0] >= 0#top_left[1]
        flag_h = np.logical_and(flag_h_p, flag_h_n)
        flag_w_p = imgpoints2[:, 1] <= 1224#bottom_right[0]
        flag_w_n = imgpoints2[:, 1] >= 0#top_left[0]
        flag_w = np.logical_and(flag_w_p, flag_w_n)
        flag = flag_h * flag_w
        count = np.sum(flag)

        imgpoints2 = imgpoints2[flag==True]
        # print(imgpoints2[flag==True])

        # if (imgpoints2.shape[0] >  1):
        #     # rect = cv2.minAreaRect(imgpoints2)
        #     # box = np.int0(cv2.boxPoints(rect))
        #     box = minimum_bounding_quadrilateral(imgpoints2)
        #     print(bbox)
        #     print(box)
        #     box = [box[0], box[1], box[2], box[3]]
        #     print(box)
        # else:
        #     box = []
        # res_bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
        
        flag_points[:,0:3]=points
        flag_points[flag==True,3]=1
        return imgpoints2,flag_points#, res_bbox, box
        # return imgpoints2[flag==True],flag_points


    def project2d_simple(self,Points, Pose):#参考
        points = Points[:, 0: 3]#舍去最后1维度特征，保留前3个坐标
        points=torch.from_numpy(points)
        one=torch.ones(points.shape[0])
        one=torch.unsqueeze(one,dim=1)
        points=torch.cat((points,one),dim=1)#构造齐次坐标


        pose=torch.from_numpy(Pose)#保留
        # pose=pose.repeat(points.shape[0],1,1,1)
        # points=torch.from_numpy(points)
        camera_3d_cord=torch.zeros((points.shape[0],4),dtype=torch.float64)
        for i in range(points.shape[0]):
            camera_3d_cord[i]=torch.matmul(pose,points[i])
        camera_3d_cord=camera_3d_cord[:,0:3]
        camera_2d_cord=torch.zeros((points.shape[0],3),dtype=torch.float64)
        for i in range(points.shape[0]):
            camera_2d_cord[i]=torch.matmul(torch.from_numpy(self.camera0_CameraMatrix), camera_3d_cord[i])


        camera_2d_cord=camera_2d_cord/torch.unsqueeze(camera_2d_cord[...,2],dim=1).expand(-1,3)
        imgpoints2=camera_2d_cord[:,0:2]
        flag_h_p = imgpoints2[:, 0] < 346#这里hw有可能相反
        flag_h_n = imgpoints2[:, 0] >= 0
        flag_h = torch.logical_and(flag_h_p, flag_h_n)
        flag_w_p = imgpoints2[:, 1] < 260
        flag_w_n = imgpoints2[:, 1] >= 0
        flag_w = torch.logical_and(flag_w_p, flag_w_n)
        flag = flag_h * flag_w
        count = torch.sum(flag)
        print(imgpoints2[flag == True])

        flag_points = np.zeros_like(Points)
        flag_points[:, 0:3] = points[:, 0:3]
        flag_points[flag == True, 3] = 1

        return imgpoints2[flag == True],flag_points


    def draw_point(self,img, points):
        alpha = 0.4
        overlay = img.copy()
        original = img.copy() 
        if points.shape[0]==0:
            print("投影点不足")
        for point in points:
            point = point.astype(int)
            cv2.circle(overlay, tuple(point.ravel()), 1, (0, 0, 0), -1)
            cv2.circle(original, tuple(point.ravel()), 1, (0, 0, 0), -1)
        pointed_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return pointed_img


    def Back_point_filtering(self,Points,RT,max_distance):#只保留相机正半球的点云,RT位姿左乘点云
        if Points.shape[1]==4:
            points = Points[:, 0: 3]  # 舍去最后1维度特征，保留前3个坐标
        elif Points.shape[1]==3:
            points=Points
        else:
            print("点云格式有误")


        points=torch.from_numpy(points)#
        pose = torch.from_numpy(RT)

        # one = torch.ones(points.shape[0])
        # one = torch.unsqueeze(one, dim=1)
        # points = torch.cat((points, one), dim=1)  # 构造齐次坐标

        # repat_pose=pose.expand(points.shape[0],-1,-1)
        # points=torch.unsqueeze(points,dim=2)
        # points_at_camera4v = torch.bmm(repat_pose, points)#将点云从雷达坐标系转换到相机坐标系
        # points_at_camera4v=torch.squeeze(points_at_camera4v)
        # camera_zr = torch.Tensor([0, 0, 1])#相机坐标系下的光轴
        # camera_zr=camera_zr.to(torch.float64)

        # camera_zr=camera_zr.expand(points_at_camera4v.shape[0],-1)
        # camera_zr=camera_zr.unsqueeze(dim=1)
        # points_at_camera3v=points_at_camera4v[:,0:3]

        # point_3d_distan=torch.norm(points_at_camera3v,dim=1)

        # points_at_camera3v=points_at_camera3v.unsqueeze(dim=2)
        # inner_product=torch.bmm(camera_zr,points_at_camera3v)#实际上是利用光轴与各个点内积判断方向是否一致
        # inner_product=inner_product.squeeze()

        # bool_flag=inner_product>=0
        # test=torch.sum(bool_flag)

        # flag=torch.logical_and(inner_product>=0,point_3d_distan<=max_distance)#夹角90以内，距离20以内
        inv_pose = torch.from_numpy(np.linalg.inv(RT))
        camera_zr = torch.Tensor([0, 0, 1, 1])  # 相机坐标系下的光轴(齐次坐标)
        camera_zr = camera_zr.to(torch.float64)
        camera_at_lidar=torch.mv(inv_pose,camera_zr)
        vector_cam_lidar=points-camera_at_lidar[0:3]#齐次坐标取出前三

        point_3d_distan = torch.norm(vector_cam_lidar, dim=1)
        flag =  point_3d_distan <= max_distance  # 夹角90以内，距离20以内
        """只约束距离"""
        filted_points=Points[flag]#内积大于零就是夹角90度以内
        return filted_points
    def project_point_filtering(self, Points, RT, bbox, max_distance,cam_type):
        assert max_distance>0
        # start_time = time.time()
        enlarge_pixel=0
        pose_lidar2camera=np.linalg.inv(RT)#pose是点云转换到相机坐标的，相反操作要取逆
        pose_lidar2camera = torch.from_numpy(pose_lidar2camera)

        points = Points[:, 0: 3]  # 舍去最后1维度特征，保留前3个坐标
        points = torch.from_numpy(points)

        top_left_at_img=bbox[0]
        bottom_right_at_img=bbox[1]
        if cam_type=="DVS":
            cx=self.event_intrinsics[2]
            cy=self.event_intrinsics[5]
            fx=self.event_intrinsics[0]
            fy=self.event_intrinsics[4]
        elif cam_type=="IMG":
            cx = self.camera0_intrinsics[2]
            cy = self.camera0_intrinsics[5]
            fx = self.camera0_intrinsics[0]
            fy = self.camera0_intrinsics[4]
        else:
            print("cam_type must be DVS or IMG")
        top_left_at_cam = torch.Tensor(top_left_at_img)
        top_left_at_cam[0] = top_left_at_cam[0]-enlarge_pixel
        top_left_at_cam[1] = top_left_at_cam[1] - enlarge_pixel
        top_left_at_cam[0] = top_left_at_cam[0] - cy#左上角为原点的图像坐标系转换为相机坐标系
        top_left_at_cam[1] = top_left_at_cam[1] - cx
        bottom_right_at_cam = torch.Tensor(bottom_right_at_img)
        bottom_right_at_cam[0] = bottom_right_at_cam[0] + enlarge_pixel
        bottom_right_at_cam[1] = bottom_right_at_cam[1] + enlarge_pixel
        bottom_right_at_cam[0] = bottom_right_at_cam[0] - cy
        bottom_right_at_cam[1] = bottom_right_at_cam[1] - cx
        top_right_at_cam=torch.Tensor([top_left_at_cam[0],bottom_right_at_cam[1]])
        bottom_left_at_cam=torch.Tensor([bottom_right_at_cam[0],top_left_at_cam[1]])

        Depth = max_distance  # 由于是射线，以最远点为
        camera_origin_4d_at_cam=torch.Tensor([0,0,0,1/ Depth]).to(torch.float64)#构造各个关键点的齐次坐标（4个顶点+4棱锥顶点）
        #[X,Y,Z,1](depth 放到最后齐次坐标)
        top_left_4d_at_cam = torch.Tensor([top_left_at_cam[1]/ fx, top_left_at_cam[0]/ fy, 1, 1/ Depth]).to(torch.float64)#X=D*x/fx
        top_right_4d_at_cam = torch.Tensor([top_right_at_cam[1]/ fx, top_right_at_cam[0]/ fy, 1, 1/ Depth]).to(torch.float64)  # X=D*x/fx
        bottom_left_4d_at_cam = torch.Tensor([bottom_left_at_cam[1]/ fx, bottom_left_at_cam[0]/ fy, 1, 1/ Depth]).to(torch.float64)  # X=D*x/fx
        bottom_right_4d_at_cam = torch.Tensor([bottom_right_at_cam[1]/ fx, bottom_right_at_cam[0]/ fy, 1, 1/ Depth]) .to(torch.float64) # X=D*x/fx

        camera_origin_4d_at_lidar = torch.matmul(pose_lidar2camera,camera_origin_4d_at_cam)#相机模型点转换到点云坐标系下
        top_left_4d_at_lidar = torch.matmul(pose_lidar2camera,top_left_4d_at_cam)
        top_right_4d_at_lidar = torch.matmul(pose_lidar2camera, top_right_4d_at_cam)
        bottom_left_4d_at_lidar = torch.matmul(pose_lidar2camera, bottom_left_4d_at_cam)
        bottom_right_4d_at_lidar = torch.matmul(pose_lidar2camera, bottom_right_4d_at_cam)

        camera_origin_3d_at_lidar=camera_origin_4d_at_lidar/ camera_origin_4d_at_lidar[3]#齐次坐标转为3维坐标
        camera_origin_3d_at_lidar=camera_origin_3d_at_lidar[:3]
        top_left_3d_at_lidar = top_left_4d_at_lidar/ top_left_4d_at_lidar[3]  # 齐次坐标转为3维坐标
        top_left_3d_at_lidar = top_left_3d_at_lidar[:3]
        top_right_3d_at_lidar = top_right_4d_at_lidar/ top_right_4d_at_lidar[3]  # 齐次坐标转为3维坐标
        top_right_3d_at_lidar = top_right_3d_at_lidar[:3]
        bottom_left_3d_at_lidar = bottom_left_4d_at_lidar/ bottom_left_4d_at_lidar[3]  # 齐次坐标转为3维坐标
        bottom_left_3d_at_lidar = bottom_left_3d_at_lidar[:3]
        bottom_right_3d_at_lidar = bottom_right_4d_at_lidar/ bottom_right_4d_at_lidar[3]  # 齐次坐标转为3维坐标
        bottom_right_3d_at_lidar = bottom_right_3d_at_lidar[:3]



        ##上平面下方：Cz<=-1*(Ax+By+D)
        TP_A,TP_B,TP_C,TP_D=cal_plane_equation(camera_origin_3d_at_lidar,top_left_3d_at_lidar,
                                               top_right_3d_at_lidar)#top_plane
        #平面方程Ax+By+Cz+D=0，x(w) y(h) z(depth)
        #上平面下方：By<=-(Ax+Cz+D)
        TP_mask = TP_C * points[:,2] >= -1*(TP_A*points[:,0] + TP_B*points[:,1]+TP_D)
        TP_num=torch.sum(TP_mask)
        TP_reduce=points.shape[0]-TP_num
        points=points[TP_mask]

        # valid_pcd = o3d.geometry.PointCloud()
        # valid_pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 显示参与有效投影的点云
        # o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释



        ##下平面上方：Cz>=-(Ax+By+D)
        BP_A, BP_B, BP_C, BP_D = cal_plane_equation(camera_origin_3d_at_lidar, bottom_left_3d_at_lidar,
                                                    bottom_right_3d_at_lidar)  # top_plane
        BP_mask=BP_C*points[:,2]<=-1*(BP_A*points[:,0]+BP_B*points[:,1]+BP_D)
        BP_num = torch.sum(BP_mask)
        BP_reduce = points.shape[0] - BP_num
        points = points[BP_mask]

        # valid_pcd = o3d.geometry.PointCloud()
        # valid_pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 显示参与有效投影的点云
        # o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释



        #左平面右方：Ax>=-(By+Cz+D)
        LP_A, LP_B, LP_C, LP_D = cal_plane_equation(camera_origin_3d_at_lidar, top_left_3d_at_lidar,
                                                    bottom_left_3d_at_lidar)  # top_plane
        LP_mask = LP_A * points[:, 0] <= -1 * (LP_B * points[:, 1] + LP_C * points[:, 2] + LP_D)
        LP_num = torch.sum(LP_mask)
        LP_reduce = points.shape[0] - LP_num
        points = points[LP_mask]

        # valid_pcd = o3d.geometry.PointCloud()
        # valid_pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 显示参与有效投影的点云
        # o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释
        #
        #
        #右平面左方：Ax<=-(By+Cz+D)
        RP_A, RP_B, RP_C, RP_D = cal_plane_equation(camera_origin_3d_at_lidar, top_right_3d_at_lidar,
                                                    bottom_right_3d_at_lidar)  # top_plane
        RP_mask = RP_A * points[:, 0] >= -1 * (RP_B * points[:, 1] + RP_C * points[:, 2] + RP_D)
        RP_num = torch.sum(RP_mask)
        RP_reduce = points.shape[0] - RP_num
        points = points[RP_mask]


        # valid_pcd = o3d.geometry.PointCloud()
        # valid_pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 显示参与有效投影的点云
        # o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释

        # elapsed = (time.time() - start_time)
        # print("四棱柱截取用时:", elapsed)
        filted_points=points.numpy()
        return filted_points


if __name__ == '__main__':#需要读取事件、图像、点云、内外参、位姿
    Folde_Address="G:\VECtor_Benchmark"

    """读取事件流数据"""
    VECtor_events = h5py.File(Folde_Address+"\Large-scale\corridors_dolly\corridors_dolly1.synced.left_event.hdf5", 'r')
    # VECtor_events = h5py.File("G:\VECtor Benchmark\Large-scale\corridors-dolly/corridors_dolly1.synced.left_event.hdf5",
    #                           'r')
    events_x = VECtor_events['events/x'][:]
    events_y = VECtor_events['events/y'][:]
    events_t = VECtor_events['events/t'][:]  # 单位微秒，转到秒要/1e6
    events_p = VECtor_events['events/p'][:]
    events_ms_to_idx = VECtor_events['ms_to_idx'][:]  # 毫秒级别索引对应的事件序号
    #即毫秒t为offset序号，得到的值为事件序号
    events_t_offset = VECtor_events['t_offset'][:]  # 事件流的起始绝对时间，上面的t是相对时间，单位微秒10e-6

    # max_t=events_t[-1]
    # test=(events_t_offset+events_t[-1])
    # test2=test/1e6
    """读取事件流数据"""

    """读取图像列表"""
    img_folde_dir = Folde_Address + '/Large-scale/corridors_dolly/corridors_dolly1.synced.left_camera/'
    image_file_list = os.listdir(img_folde_dir)
    image_file_list.sort()

    image_ts = torch.zeros(len(image_file_list))
    image_ts = image_ts.to(dtype=torch.float64)

    img_list_idx = 0
    torch.set_printoptions(precision=6, sci_mode=False)
    for name in image_file_list:
        image_ts[img_list_idx] = float(name[:17])
        # print(pic_ts[name_list_idx])
        pic_dir = img_folde_dir + name
        pic_dir2 = img_folde_dir + image_file_list[img_list_idx]
        if pic_dir!=pic_dir2:
            print("pic_dir!=pic_dir2:",pic_dir,pic_dir2)
        img_list_idx += 1

    img = cv2.imread(pic_dir2,0)
    # test_img=cv2.imread("G:\VECtor Benchmark/Large-scale\corridors-dolly\corridors_dolly1.synced.left_camera/1643091527.996090.png",0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)  # numpy数组格式为（H,W,C）
    img_tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
    print(img_tensor.size())
    # cv2.imshow("test_image", img)
    # cv2.waitKey(0)
    # input_tensor = img_tensor.clone().detach().to(torch.device('cpu'))  # 到cpu
    # torchvision.utils.save_image(input_tensor, "out_cv.jpg")
    """读取图像"""

    """读取点云"""
    pcl_folde_dir = Folde_Address + '\Large-scale\corridors_dolly\corridors_dolly1.synced.lidar'
    pcl_list = os.listdir(pcl_folde_dir)
    pcl_list.sort()

    pcl_ts = torch.zeros(len(pcl_list))
    pcl_ts = pcl_ts.to(dtype=torch.float64)

    pcl_list_idx = 0
    torch.set_printoptions(precision=6, sci_mode=False)
    for name in pcl_list:
        pcl_ts[pcl_list_idx] = float(name[:17])
        # print(pic_ts[name_list_idx])
        pcl_dir = pcl_folde_dir +"/"+ name
        pcl_dir2 = pcl_folde_dir +"/"+ pcl_list[pcl_list_idx]
        if pcl_dir != pcl_dir2:
            print("pcl_dir!=pcl_dir2:", pcl_dir, pcl_dir2)
        pcl_list_idx += 1

    curr_scan_pts = o3d.io.read_point_cloud(pcl_dir)
    LiDAR_frame = np.asarray(curr_scan_pts.points)

    # curr_scan_pts = o3d.io.read_point_cloud("E:\事件匹配\Event_LiDAR_match\data/1643091436.096091.pcd")
    # test_LiDAR_frame = np.asarray(curr_scan_pts.points)

    curr_scan_pts = o3d.io.read_point_cloud("E:\事件匹配\Event_LiDAR_match\data\map\Vector/corridors_dolly.pcd")
    full_LiDAR_map = np.asarray(curr_scan_pts.points)
    # full_LiDAR_map=np.matmul(np.linalg.inv(model.TEST_RO),full_LiDAR_map)
    # full_LiDAR_map = torch.from_numpy(full_LiDAR_map)
    # RO = torch.from_numpy(model.TEST_RO)  # 纠正数据集多乘的旋转矩阵
    # R0 = torch.inverse(RO)
    # R0 = R0.expand(full_LiDAR_map.shape[0], -1, -1)
    # full_LiDAR_map = torch.unsqueeze(full_LiDAR_map, dim=2)
    # full_LiDAR_map = torch.bmm(R0, full_LiDAR_map)
    # full_LiDAR_map = full_LiDAR_map.squeeze()
    # full_LiDAR_map = full_LiDAR_map.numpy()
    """读取点云"""

    """读取位姿GT"""
    pose_dir =Folde_Address+ '/Large-scale/corridors_dolly/' + "corridors_dolly1.synced" + '.gt.txt'
    with open(pose_dir, "r") as pose_file:
        pose_num = len(pose_file.readlines())
        pose_num = pose_num - 2
        pose_ts = torch.zeros(pose_num)
        pose_dict = torch.zeros((pose_num,7))
        pose_ts = pose_ts.to(dtype=torch.float64)
        pose_dict=pose_dict.to(dtype=torch.float64)
        pose_index=0

    with open(pose_dir, "r") as pose_file:
        for line in pose_file.readlines(pose_index):
        # for flag in range(pose_num):
            # line=pose_file.readlines(pose_index+2)
            # print(line)
            if pose_index<2:
                pose_index+=1
                continue#前两行是数据格式备注，跳过
            tokens = line.split(" ")
            pose_ts[pose_index-2] = float(tokens[0])#前两条跳过，这里要扣减
            pose = tokens[1:]
            pose_dict[pose_index-2,0] = float(pose[0])
            pose_dict[pose_index - 2, 1] = float(pose[1])
            pose_dict[pose_index - 2, 2] = float(pose[2])
            pose_dict[pose_index - 2, 3] = float(pose[3])
            pose_dict[pose_index - 2, 4] = float(pose[4])
            pose_dict[pose_index - 2, 5] = float(pose[5])
            pose_dict[pose_index - 2, 6] = float(pose[6])
            # transform = seven_to_pose(pose)
            pose_index += 1
    print("位姿读取完成")
    # corridors_dolly1.synced.gt.txt

    """读取位姿GT"""
    pointcloud_ts=pcl_ts#(events_t_offset+events_t)#单位是微秒，需要乘1e-6
    pose = pose_dict
    # pose_ts = H5file['davis']['left']['pose_ts']  # pose是全局对准的位姿
    image_file_list=image_file_list
    image_ts=image_ts
    """
    img = cv2.imread(img_folde_dir+image_file_list[-1], 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)  # numpy数组格式为（H,W,C）
    img_tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
    # print(img_tensor.size())"""


    test_point_flag = 888
    stack_mode="SBN"
    SBN_num=50000
    SBT_time=0.05
    test_point_time = pointcloud_ts[test_point_flag]
    test_pose_index=np.searchsorted(pose_ts[...],test_point_time)
    test_pose_ts=pose_ts[test_pose_index]
    test_image_index=np.searchsorted(image_ts[...],test_point_time)

    """截取对应的事件流片段"""
    event_mid_index=np.searchsorted((events_t_offset+events_t)/1e6,test_point_time)

    if stack_mode == "SBT":
        event_begin_index=np.searchsorted((events_t_offset+events_t)/1e6,test_point_time-0.5*SBT_time)
        event_end_index = np.searchsorted((events_t_offset + events_t) / 1e6, test_point_time + 0.5 * SBT_time)
    elif stack_mode == "SBN":
        event_begin_index=event_mid_index-SBN_num//2
        event_end_index=event_mid_index+SBN_num//2
    else:
        print("stack_mode出错")
    event_flow_x=events_x[event_begin_index:event_end_index]
    event_flow_y = events_y[event_begin_index:event_end_index]
    event_flow_t = events_t[event_begin_index:event_end_index]
    event_flow_p = events_p[event_begin_index:event_end_index]
    events_frame = event2image_VECtor(event_flow_x,event_flow_y,event_flow_p, [480,640])
    """截取对应的事件流片段"""

    curr_scan_pts = o3d.io.read_point_cloud(pcl_folde_dir +"/"+ pcl_list[test_point_flag])
    LiDAR_frame = np.asarray(curr_scan_pts.points)
    # pcl_ts,pcl_list是点云时间戳和点云帧的文件利比饿哦
    Full_LiDAR_map = full_LiDAR_map


    work_path = os.path.abspath('..')
    Parameter_path = os.path.join(work_path, "data")
    model=Projection_model_VECtor(Parameter_path,"Large-scale")
    lidar_pose=pose[test_pose_index]#世界坐标系下相机的位姿Pose，也是相机坐标系转为世界坐标系的矩阵
    lidar_pose=seven_to_pose(lidar_pose.numpy())
    cam_pose=np.matmul(model.T_cam0_lidar,np.linalg.inv(lidar_pose))  # T_cam0_lidar*lidar_pose
    DVS_pose=np.matmul(np.linalg.inv(model.T_cam0_camDVS),cam_pose)
    # inv_cam_pose=cam_pose
    # inv_cam_pose=np.linalg.inv(cam_pose)#世界坐标系转换为相机坐标系，需要取一个逆


    # r,jacobian = cv2.Rodrigues(cam_pose_R)#旋转矩阵转为旋转向量

    cam_pose_R = cam_pose[0:3, 0:3]#相机位姿RT
    cam_pose_T = cam_pose[0:3, 3]

    DVS_pose_R = DVS_pose[0:3, 0:3]  # 相机位姿RT
    DVS_pose_T = DVS_pose[0:3, 3]

    """测试投影前的点云坐标"""
    test_R=torch.from_numpy(DVS_pose)
    test_R=test_R.expand(full_LiDAR_map.shape[0],-1,-1)
    test_full_lidar=torch.from_numpy(full_LiDAR_map)
    test_full_lidar=torch.cat((test_full_lidar,torch.ones(test_full_lidar.shape[0],1)),dim=1)
    test_full_lidar=torch.unsqueeze(test_full_lidar,dim=2)
    test_full_lidar_after=torch.bmm(test_R,test_full_lidar)
    """测试投影前的点云坐标"""

    """测试单帧投影
    external_parameters=model.T_cam0_lidar#model.T_camDVS_GT#相机到雷达的变换，等价于雷达坐标系的点相机坐标系的变换
    ext_R= external_parameters[0:3, 0:3]#标定的相机外参RT
    ext_T= external_parameters[0:3, 3]
    
    filted_points = model.Back_point_filtering(LiDAR_frame, inv_cam_pose)

    # KU,flag_points=model.Point2Pic(filted_points, ext_R, ext_T, )
    KU, flag_points = model.Point2Pic("cam0", filted_points, ext_R, ext_T, )
    # DIY,flag_points=model.project2d_simple(filted_points,inv_cam_pose)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filted_points[:, :3])  # 显示全部点云
    o3d.visualization.draw_geometries([pcd])

    test_image = cv2.imread(img_folde_dir + image_file_list[test_image_index], 0)
    # test_image=image[test_image_index]

    cv2.imshow("test_image", test_image)
    cv2.waitKey(0)
    test_image_with_point = model.draw_point(test_image, KU)
    cv2.imshow("test_image_with_point", test_image_with_point)
    cv2.waitKey(0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flag_points[:, :3])  # 显示全部点云
    o3d.visualization.draw_geometries([pcd])

    valid_pcd = o3d.geometry.PointCloud()
    valid_points = flag_points[flag_points[:, 3] > 0]
    valid_pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])  # 显示参与有效投影的点云
    o3d.visualization.draw_geometries([valid_pcd])
    测试单帧投影"""


    # filted_points = model.Back_point_filtering(full_LiDAR_map, cam_pose,10)#传统相机
    filted_points = model.Back_point_filtering(full_LiDAR_map, DVS_pose,10)  # 事件相机


    # KU, flag_points = model.Point2Pic("cam0",filted_points, cam_pose_R, cam_pose_T, )#传统相机
    bbox=[[20,20],[532,532]]
    KU, flag_points = model.Point2Pic("DVS", filted_points, DVS_pose_R, DVS_pose_T, bbox)  # 事件相机
    # DIY,flag_points=model.project2d_simple(filted_points,inv_cam_pose)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filted_points[:, :3])  # 显示全部点云
    o3d.visualization.draw_geometries([pcd])

    # Full_map, flag_points = model.Point2Pic(full_LiDAR_map, cam_pose_R, cam_pose_T, )

    # filted_points=model.Back_point_filtering(pointcloud_list[test_point_flag],external_parameters)

    test_image = cv2.imread(img_folde_dir + image_file_list[test_image_index], 0)
    # test_image=image[test_image_index]

    cv2.imshow("test_image", test_image)
    cv2.waitKey(0)
    # test_image_with_point=model.draw_point(test_image,KU)#图像帧
    test_image_with_point = model.draw_point(events_frame, KU)#事件帧
    # events_frame[:,:,1]=test_image_with_point

    cv2.imshow("test_image_with_point", test_image_with_point)
    cv2.waitKey(0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(flag_points[:, :3])#显示全部点云
    o3d.visualization.draw_geometries([pcd])

    valid_pcd = o3d.geometry.PointCloud()
    valid_points=flag_points[flag_points[:,3]>0]
    valid_pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])#显示参与有效投影的点云
    o3d.visualization.draw_geometries([valid_pcd])

    # curr_scan_pts = o3d.io.read_point_cloud(scan_path)
    # curr_scan_pts = np.asarray(curr_scan_pts.points)





    print("处理完成")
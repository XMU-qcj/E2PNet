import cv2
import numpy as np
import torch
import yaml
import zipfile
import h5py
import tqdm


import open3d as o3d
import cv2
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
from result_visualization import  event2image_MVSEC


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

class Projection_model_MVSEC:
    def __init__(self,folder_path,experiment_name):
        self.calibration = self.load_yaml(folder_path,experiment_name)



        self.Pfx = self.calibration['cam0']['projection_matrix'][0][0]
        self.Ppx = self.calibration['cam0']['projection_matrix'][0][2]
        self.Pfy = self.calibration['cam0']['projection_matrix'][1][1]
        self.Ppy = self.calibration['cam0']['projection_matrix'][1][2]

        self.intrinsics=self.calibration['cam0']['intrinsics']

        # intrinsics = self.calibration.intrinsic_extrinsic['cam0']['intrinsics']
        intrinsics=self.intrinsics
        self.projection_matrix = self.calibration['cam0']['projection_matrix']

        self.P = np.array([[self.Pfx, 0., self.Ppx],
                           [0., self.Pfy, self.Ppy],
                           [0., 0., 1.]])

        self.K = np.array([[intrinsics[0], 0., intrinsics[2]],
                           [0., intrinsics[1], intrinsics[3]],
                           [0., 0., 1.]])

        self.CameraMatrix = self.K

        self.distortion_coeffs = np.array(self.calibration['cam0']['distortion_coeffs'])
        resolution = self.calibration['cam0']['resolution']
        # self.resolution=self.calibration['cam0']['resolution']

        T_cam0_lidar=self.calibration['T_cam0_lidar']
        assert T_cam0_lidar[3][3]==1 and T_cam0_lidar[3][0]==0 and T_cam0_lidar[3][1]==0 and T_cam0_lidar[3][2]==0
        self.T_cam2_lidar=np.array([[T_cam0_lidar[0][0], T_cam0_lidar[0][1], T_cam0_lidar[0][2],T_cam0_lidar[0][3]],
                                    [T_cam0_lidar[1][0], T_cam0_lidar[1][1], T_cam0_lidar[1][2],T_cam0_lidar[1][3]],
                                    [T_cam0_lidar[2][0], T_cam0_lidar[2][1], T_cam0_lidar[2][2], T_cam0_lidar[2][3]],
                                    [0., 0.,0 ,1.]])


    def load_yaml(self,folder_path,experiment_name):#experiment_name="indoor_flying"

        yaml_name = "camchain-imucam-"+experiment_name+".yaml"
        # yaml_path="E:/事件匹配/Event_LiDAR_match/data/"+yaml_name
        yaml_path = folder_path+ yaml_name
        
        with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
            intrinsic_extrinsic = yaml.load(yaml_file, Loader=yaml.SafeLoader)
            return intrinsic_extrinsic
        # print("data_type:", data[0:50])
        # print(intrinsic_extrinsic["cam0"])
        #
        # # yaml_file = open("path", "r")
        # # data = yaml.load(yaml_file)
        #
        # return intrinsic_extrinsic

    def Point2Pic(self, Points, R, T,bbox):
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


        start_time = time.time()
        # R=R.numpy()
        # T=T.numpy()
        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # self.distortion_coeffs 
        imgpoints2, Jacobian = cv2.projectPoints(points, R, T, self.CameraMatrix, self.distortion_coeffs,jacobian=None)#, self.distortion_coeffs
        imgpoints2=np.squeeze(imgpoints2)#投影后2D点是【u，v】，也就是【w,h】

        # imgpoints2[:0] = -imgpoints2[:0]
        # imgpoints2[:1] = -imgpoints2[:1]#正负取反

        top_left, bottom_right = bbox#[h,w]

        flag_h_p=imgpoints2[:,0]<=bottom_right[1]#imgpoints2是[u,v],即[w,h]
        flag_h_n = imgpoints2[:, 0] >= top_left[1]
        flag_h=np.logical_and(flag_h_p,flag_h_n)
        flag_w_p=imgpoints2[:,1]<=bottom_right[0]
        flag_w_n = imgpoints2[:, 1] >=top_left[0]
        flag_w=np.logical_and(flag_w_p,flag_w_n)
        flag=flag_h*flag_w
        count=np.sum(flag)
        # print(imgpoints2[flag==True])

        flag_points[:,0:3]=points
        flag_points[flag==True,3]=1
        
        
        return imgpoints2[flag==True],flag_points



    def project2d_simple(self,Points, Pose):#参考
        points = Points[:, 0: 3]#舍去最后1维度特征，保留前3个坐标
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
            camera_2d_cord[i]=torch.matmul(torch.from_numpy(self.CameraMatrix), camera_3d_cord[i])


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
        # print(imgpoints2[flag == True])

        flag_points = np.zeros_like(Points)
        flag_points[:, 0:3] = points[:, 0:3]
        flag_points[flag == True, 3] = 1

        return imgpoints2[flag == True],flag_points



    def draw_point(self,img, points):

        for point in points:
            pointed_img=cv2.circle(img, tuple(point.ravel()), 1, (255, 255, 255), 3, cv2.LINE_8)
        return pointed_img


    def Back_point_filtering(self,Points,RT):#只保留相机正半球的点云,RT位姿左乘点云
        if Points.shape[1]==4:
            points = Points[:, 0: 3]  # 舍去最后1维度特征，保留前3个坐标
        elif Points.shape[1]==3:
            points=Points
        else:
            print("点云格式有误")


        points=torch.from_numpy(points)#
        pose = torch.from_numpy(RT)

        one = torch.ones(points.shape[0])
        one = torch.unsqueeze(one, dim=1)
        points = torch.cat((points, one), dim=1)  # 构造齐次坐标
          # 保留

        repat_pose=pose.expand(points.shape[0],-1,-1)
        points=torch.unsqueeze(points,dim=2)
        points_at_camera4v = torch.bmm(repat_pose, points)#将点云从雷达坐标系转换到相机坐标系
        points_at_camera4v=torch.squeeze(points_at_camera4v)
        camera_zr = torch.Tensor([0, 0, 1])#相机坐标系下的光轴
        camera_zr=camera_zr.to(torch.float64)

        camera_zr=camera_zr.expand(points_at_camera4v.shape[0],-1)
        camera_zr=camera_zr.unsqueeze(dim=1)
        points_at_camera3v=points_at_camera4v[:,0:3]
        points_at_camera3v=points_at_camera3v.unsqueeze(dim=2)
        inner_product=torch.bmm(camera_zr,points_at_camera3v)#实际上是利用光轴与各个点内积判断方向是否一致
        inner_product=inner_product.squeeze()

        # bool_flag=inner_product>=0
        # test=torch.sum(bool_flag)

        filted_points=Points[inner_product>=0]
        return filted_points
    def project_point_filtering(self, Points, RT, bbox, max_distance):
        assert max_distance>0
        # start_time = time.time()
        enlarge_pixel=40

        pose_lidar2camera=np.linalg.inv(RT)#pose是点云转换到相机坐标的，相反操作要取逆
        pose_lidar2camera = torch.from_numpy(pose_lidar2camera)

        points = Points[:, 0: 3]  # 舍去最后1维度特征，保留前3个坐标
        points = torch.from_numpy(points)
        # test_P_xmax =torch.max(points[:,0])
        # test_P_ymax = torch.max(points[:, 1])
        # test_P_zmax = torch.max(points[:, 2])
        # test_P_xmin = torch.min(points[:, 0])
        # test_P_ymin = torch.min(points[:, 1])
        # test_P_zmin = torch.min(points[:, 2])

        top_left_at_img=bbox[0]
        bottom_right_at_img=bbox[1]

        cx=self.intrinsics[2]
        cy=self.intrinsics[3]
        fx=self.intrinsics[0]
        fy=self.intrinsics[1]


        top_left_at_cam = torch.Tensor(top_left_at_img)
        top_left_at_cam[0]=top_left_at_cam[0]-enlarge_pixel
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

        # valid_pcd = o3d.geometry.PointCloud()
        # valid_pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 显示参与有效投影的点云
        # o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释

        ##上平面下方：Cz<=-1*(Ax+By+D)
        TP_A,TP_B,TP_C,TP_D=cal_plane_equation(camera_origin_3d_at_lidar,top_left_3d_at_lidar,
                                               top_right_3d_at_lidar)#top_plane
        #平面方程Ax+By+Cz+D=0，x(w) y(h) z(depth)
        #上平面下方：By<=-(Ax+Cz+D)
        TP_mask = TP_C * points[:,2] >= -1*(TP_A*points[:,0] + TP_B*points[:,1]+TP_D)
        # TP_num=torch.sum(TP_mask)
        # TP_reduce=points.shape[0]-TP_num
        points=points[TP_mask]



        ##下平面上方：Cz>=-(Ax+By+D)
        BP_A, BP_B, BP_C, BP_D = cal_plane_equation(camera_origin_3d_at_lidar, bottom_left_3d_at_lidar,
                                                    bottom_right_3d_at_lidar)  # top_plane
        BP_mask=BP_C*points[:,2]<=-1*(BP_A*points[:,0]+BP_B*points[:,1]+BP_D)
        # BP_num = torch.sum(BP_mask)
        # BP_reduce = points.shape[0] - BP_num
        points = points[BP_mask]

        # valid_pcd = o3d.geometry.PointCloud()
        # valid_pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 显示参与有效投影的点云
        # o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释



        #左平面右方：Ax>=-(By+Cz+D)
        LP_A, LP_B, LP_C, LP_D = cal_plane_equation(camera_origin_3d_at_lidar, top_left_3d_at_lidar,
                                                    bottom_left_3d_at_lidar)  # top_plane
        LP_mask = LP_A * points[:, 0] <= -1 * (LP_B * points[:, 1] + LP_C * points[:, 2] + LP_D)
        # LP_num = torch.sum(LP_mask)
        # LP_reduce = points.shape[0] - LP_num
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
        # RP_num = torch.sum(RP_mask)
        # RP_reduce = points.shape[0] - RP_num
        points = points[RP_mask]


        # valid_pcd = o3d.geometry.PointCloud()
        # valid_pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # 显示参与有效投影的点云
        # o3d.visualization.draw_geometries([valid_pcd])  # 可视化测试用,要注释

        # elapsed = (time.time() - start_time)
        # print("四棱柱截取用时:", elapsed)
        filted_points=points.numpy()
        return filted_points





if __name__ == '__main__':
    H5file = h5py.File("./indoor_flying1_matching.hdf5", 'r')
    pointcloud = H5file['velodyne/pointcloud']
    pointcloud_ts=H5file['velodyne']['pointcloud_ts']
    """不等长的点云序列存入list"""
    pc_list_len = 0
    for i in pointcloud:
        pc_list_len += 1

    # test=pointcloud_ts.shape
    assert pc_list_len==pointcloud_ts.shape[0]
    pointcloud_list = []
    # pointcloud_ts_list=[]
    pointcloud_ts_list = torch.zeros(pc_list_len, dtype=torch.float64)
    # test_pointcloud_ts_list=np.zeros(pc_list_len)
    print("读取不等长点云数据")
    for i in tqdm.tqdm(range(pc_list_len)):

        pointcloud_list.append(torch.from_numpy(pointcloud[str(i)][:]))


    pointcloud_event_index = H5file['velodyne']['pointcloud_event_index']

    events = H5file['davis/left/events']
    events_x=events[:,0]
    events_y = events[:, 1]
    events_t = events[:, 2]
    events_p = events[:, 3]

    odometry_ts = H5file['davis']['left']['odometry_ts']
    odometry = H5file['davis/left/odometry']  # 这个是帧间位姿，与激光雷达联合获取深度与光流GT（随时间漂变）

    depth_image_raw = H5file['davis/left/depth_image_raw']
    depth_image_rect = H5file['davis/left/depth_image_rect']

    pose = H5file['davis/left/pose']  # 与上面是两种等效的表达方式，上面的更高效
    pose_ts = H5file['davis']['left']['pose_ts']  # pose是全局对准的位姿

    image=H5file['davis']['left']['img_frame']
    image_ts=H5file['davis']['left']['image_ts']

    curr_scan_pts = o3d.io.read_point_cloud("./1.pcd")
    full_LiDAR_map = np.asarray(curr_scan_pts.points)


    test_point_flag=888
    stack_mode = "SBN"
    SBN_num = 50000
    SBT_time = 0.05
    test_point_time=pointcloud_ts[test_point_flag]
    test_pose_index=np.searchsorted(pose_ts[...],test_point_time)
    # test_pose_ts=pose_ts[test_pose_index-2:test_pose_index+2]
    test_image_index=np.searchsorted(image_ts[...],test_point_time)
    model=Projection_model_MVSEC("E:/事件匹配/Event_LiDAR_match/data/","indoor_flying")
    cam_pose=pose[test_pose_index]#世界坐标系下相机的位姿Pose，也是相机坐标系转为世界坐标系的矩阵

    inv_cam_pose=np.linalg.inv(cam_pose)#世界坐标系转换为相机坐标系，需要取一个逆

    external_parameters=model.T_cam2_lidar#相机到雷达的变换，等价于雷达坐标系的点相机坐标系的变换
    # test_pose = np.matmul(model.T_cam2_lidar, test_pose)  # 还要乘一个激光雷达到相机的外参矩阵

    test_odo_index = np.searchsorted(odometry_ts[...], test_point_time)
    test_odo = odometry[test_odo_index]
    test_odo=np.matmul(model.T_cam2_lidar, test_odo)
    pose_nts=pose_ts[test_pose_index]

    """截取对应的事件流片段"""
    event_mid_index = np.searchsorted(events_t, test_point_time)

    if stack_mode == "SBT":
        event_begin_index = np.searchsorted(events_t, test_point_time - 0.5 * SBT_time)
        event_end_index = np.searchsorted( events_t, test_point_time + 0.5 * SBT_time)
    elif stack_mode == "SBN":
        event_begin_index = event_mid_index - SBN_num // 2
        event_end_index = event_mid_index + SBN_num // 2
    else:
        print("stack_mode出错")

    event_flow_x = events_x[event_begin_index:event_end_index]
    event_flow_y = events_y[event_begin_index:event_end_index]
    event_flow_t = events_t[event_begin_index:event_end_index]
    event_flow_p = events_p[event_begin_index:event_end_index]
    event_flow=np.stack((event_flow_x,event_flow_y,event_flow_t,event_flow_p),axis=1)

    events_frame = event2image_MVSEC(event_flow, [260, 346])
    """截取对应的事件流片段"""

    # T_lidar2_cam = np.linalg.inv(model.T_cam2_lidar)#测试外参是否要取反
    # r,jacobian = cv2.Rodrigues(cam_pose_R)#旋转矩阵转为旋转向量


    ext_R= external_parameters[0:3, 0:3]#标定外参RT
    ext_T= external_parameters[0:3, 3]


    cam_pose_R = inv_cam_pose[0:3, 0:3]#相机位姿RT
    cam_pose_T = inv_cam_pose[0:3, 3]

    # full_LiDAR_map=np.matmul(np.linalg.inv(model.TEST_RO),full_LiDAR_map)
    """
    full_LiDAR_map=torch.from_numpy(full_LiDAR_map)
    RO=torch.from_numpy(model.TEST_RO)#纠正数据集多乘的旋转矩阵
    R0=torch.inverse(RO)
    R0=R0.expand(full_LiDAR_map.shape[0],-1,-1)
    full_LiDAR_map=torch.unsqueeze(full_LiDAR_map,dim=2)
    full_LiDAR_map=torch.bmm(R0,full_LiDAR_map)
    full_LiDAR_map=full_LiDAR_map.squeeze()
    full_LiDAR_map=full_LiDAR_map.numpy()"""




    # filted_points = model.Back_point_filtering(pointcloud_list[test_point_flag], external_parameters)
    filted_points = model.Back_point_filtering(full_LiDAR_map, inv_cam_pose)

    # KU,flag_points=model.Point2Pic(filted_points, ext_R, ext_T, )
    KU, flag_points = model.Point2Pic(filted_points, cam_pose_R, cam_pose_T, )
    # DIY,flag_points=model.project2d_simple(filted_points,external_parameters)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filted_points[:, :3])  # 显示全部点云
    o3d.visualization.draw_geometries([pcd])

    # Full_map, flag_points = model.Point2Pic(full_LiDAR_map, cam_pose_R, cam_pose_T, )

    # filted_points=model.Back_point_filtering(pointcloud_list[test_point_flag],external_parameters)


    test_image=image[test_image_index]

    cv2.imshow("test_image", test_image)
    cv2.waitKey(0)
    test_image_with_point=model.draw_point(test_image,KU)
    event_image_with_point = model.draw_point(events_frame, KU)
    cv2.imshow("test_image_with_point", test_image_with_point)
    cv2.waitKey(0)
    cv2.imshow("event_image_with_point", event_image_with_point)
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





    print("文件写入完成")
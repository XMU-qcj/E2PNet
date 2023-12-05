import os
import h5py
import glob
import bisect
import numpy as np
import torch.utils.data as data
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_single_pcd(points, save_path):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    
    o3d.io.write_point_cloud(save_path.replace('png','pcd'), pcd)

class CrossTripletDataset(data.Dataset):
    def __init__(self, root, split, cache_size=32):
        self.views = 2
        self.split = split
        self.metadata = {}
        self.cache = {}
        self.cache_size = cache_size
        self.flist = os.path.join(root, "*.h5")
        self.flist = sorted(glob.glob(self.flist))
        for fname in self.flist:
            self._add_metadata(fname)

    def __len__(self):
        return sum([s for _, s in self.metadata.items()])

    def __getitem__(self, i):
        for fname, size in self.metadata.items():
            if i < size:
                break
            i -= size
        if fname not in self.cache:
            self._load_data(fname)
        points, images = self.cache[fname]
#         points = points[:,:,:3]
#         print(points[0:10,0,:])
#         print(points[0,:,-3:].shape)
#         for i in range(10):
#             plot_single_pcd(points[i,:,:3],'/home/lxh/qcj/lcd/lcd/'+str(i)+'.png')
#         print(dsad)
#         points = points[:,:,:3]
        
        return points[i], images[i]

    def _add_metadata(self, fname):
        with h5py.File(fname, "r") as h5:
            assert "points" in h5 and "images" in h5
            assert h5["points"].shape[0] == h5["images"].shape[0]
            size = h5["points"].shape[0]
            self.metadata[fname] = size

    def _load_data(self, fname):
        # Remove a random element from cache
        if len(self.cache) == self.cache_size:
            key = list(self.cache.keys())[0]
            self.cache.pop(key)

        h5 = h5py.File(fname, "r")
        data = (h5["points"][:,:,:3], h5["images"])
        self.cache[fname] = data

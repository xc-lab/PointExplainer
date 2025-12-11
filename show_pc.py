#  -*- coding: utf-8 -*-
'''
Display point cloud

author: xuechao.wang@ugent.be
'''

import numpy as np
import open3d as o3d

from utils.utils import data_reading, get_training_patches


def show_point_cloud(data):
    z_arr = np.array(data[:, 2])
    x_arr = np.array(data[:, 0])
    y_arr = np.array(data[:, 1])

    points = np.array([x_arr, y_arr, z_arr]).reshape(3, -1).T
    # points_color = np.array([a_arr, l_arr, p_arr]).reshape(3, -1).T
    m, n = points.shape
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((points[:, 0], points[:, 1], points[:, 2])).transpose())
    # pcd.colors = o3d.utility.Vector3dVector(np.vstack((points_color[:, 0], points_color[:, 1], points_color[:, 2])).transpose())

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.,0.,0.])
    o3d.visualization.draw_geometries([frame, pcd])


if __name__ == '__main__':

    path = './data/ParkinsonHW/raw_data/KT/C_0003.txt'
    temp_data, _ = data_reading(path, 'ParkinsonHW', 1)

    pc = temp_data[:, [0, 1, 15]]  # Control the number of attributes contained in the point cloud
    show_point_cloud(pc)  # Display the overall point cloud

    patches_data = get_training_patches(pc, 2048, 128)
    for p, patch_data in enumerate(patches_data):
        show_point_cloud(patch_data)  # Display point cloud fragments





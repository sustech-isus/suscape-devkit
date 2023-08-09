#-*-coding:utf-8-*-

'''
the point cloud  class of dataset
author:Tim Liu
'''

import numpy as np
from ..utils.parse_pcd import PointCloud

class LidarPointCloud(object):
    '''
    The Point Cloud class of the lidar_top
    the dimensions 0,1,2 represent x,y,z coordinates which in the lidar_top coordinates frame
    '''
    def __init__(self,points):
        '''
        init the lidarpoint class
        :param points: np.array the original points get from the pcd files
        '''
        self.dims = 4  # x,y,z intensity
        self.points = points
        assert points.shape[0] == self.dims, 'lidar points must have format: %d x n' % self.dims

    def get_points_number(self):
        '''
        get the number of the original points
        :return: number_points
        '''
        number_points = self.points.shape[1]
        return number_points

    @classmethod
    def load_pcdfile(cls,file_path):
        '''
        load the pcd file from the disk with the numpy
        :param file_path: the path of the file
        :return: class
        '''
        pc = PointCloud.from_path(file_path).pcd_to_numpy(intensity = True)
        return cls(pc.T)








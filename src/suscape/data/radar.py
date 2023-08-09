#-*-coding:utf-8-*-

'''
the point cloud  class of dataset
author:Tim Liu
date:2021.06
Copyright: South University of Science and Technology
'''

from ..utils.parse_pcd import PointCloud


class RadarPointCloud(object):
    '''
    The Point Cloud class of the radar
    the dimensions 0,1,2 represent x,y,z coordinates which in the radar coordinates frame
    '''
    def __init__(self,points):
        '''
        init the lidarpoint class
        :param points: np.array the original points get from the pcd files
        '''
        self.dims = 3  # x,y,z
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
        pc = PointCloud.from_path(file_path).pcd_to_numpy(intensity=False)
        return cls(pc.T)








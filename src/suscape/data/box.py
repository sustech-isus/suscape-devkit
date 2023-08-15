#-*-coding:utf-8-*-

'''
the 3D box class of  dataset
author:Tim Liu
'''

from typing import List
import numpy as np
from ..utils.common_utils import euler_angle_to_rotate_matrix_3by3

class Box(object):
    '''
    this class is defined for the 3D box of the dataset,including how to define the box, how to use the box
    and so on.
    '''

    def __init__(self,
                 center: List[float], # x,y,z
                 size: List[float], # x,y,z
                 rotation: np.ndarray = np.array([0.00,0.00,0.00]),
                 score: float = 0.00,
                 obj_id: str = "",
                 name: str = None,):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param rotation: the orientation of the box
        :param score: Classification score, optional.
        :param obj_id: the obj_id of the box,which will be used to the object tracking
        :param name: Box name, optional. Can be used e.g. for denote category name.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(rotation) == np.ndarray
        # assert type(score) == float
        # assert type(obj_id) == str
        # print('name', name)
        # assert type(name) == str

        self.center = np.array(center)
        self.scale = np.array(size)
        self.rotation = rotation
        self.score = score
        self.obj_id = obj_id
        self.name = name

    @property
    def get_rotate_matrix(self):
        '''
        get the rotation matrix from the label info rotation
        :return:  shape 4*4
        '''
        R = euler_angle_to_rotate_matrix_3by3(self.rotation)
        trans = self.center.reshape([-1,1])
        R = np.concatenate([R,trans],axis=-1)
        R = np.concatenate([R,np.array([0,0,0,1]).reshape([1,-1])],axis = 0)
        return R

    def __repr__(self):
        '''
        a string representation of the Box object
        :return:
        '''
        box_str = 'box name：{},score:{:.2f},id:{:d},position:[{:.2f},{:.2f},{:.2f}],scale:[{:.2f},{:.2f},{:.2f}]，' \
                  'rotation:[{:.2f},{:.2f},{:.2f}]，'
        return box_str.format(self.name,self.score,self.obj_id,
                              self.center[0],self.center[1],self.center[2],
                              self.scale[0],self.scale[1],self.scale[2],
                              self.rotation[0],self.rotation[1],self.rotation[2])

    @property
    def corners(self) -> np.ndarray:
        """
        Returns the bounding box corners.
        :return: <np.float: 4, 8> the first three lines is the x,y,z of the box's corners
        """
        x, y, z = self.scale
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        # front-left-bottom front-right-bottom front-right-top front-left-top rear-left-bottom rear-right-bottom rear-right-top rear-left-top
        x_corners = x / 2 * np.array([1,  1,  1, 1, -1, -1, -1, -1])
        y_corners = y / 2 * np.array([1, -1, -1, 1,  1, -1, -1,  1])
        z_corners = z / 2 * np.array([-1, -1, 1, 1,  -1,-1, 1, 1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = np.concatenate([corners.T,np.ones(8).reshape(8,-1)],axis=1)
        # translate the corers to the Lidar coordinate frame
        corners = np.matmul(self.get_rotate_matrix,np.transpose(corners))
        return corners



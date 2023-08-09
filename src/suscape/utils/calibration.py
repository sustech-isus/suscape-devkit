#-*-coding:utf-8-*-

'''
Calibration compute file
author:Tim Liu
'''

import numpy as np
from ..utils.common_utils import get_name_list,get_whole_path,loadjson,str_all_index,euler_angle_to_rotate_matrix_3by3

class Calibration(object):
    '''

    '''
    def __init__(self,calib_path):
        '''
        init the class
        :param calib_path: the path of the calibration file
        '''
        self.calib_path = calib_path
        self.sensor_list = get_name_list(calib_path)
        self.sensor_calib_dic = self.get_sensor_calib_dic()

    def get_sensor_calib_dic(self):
        '''
        get the static calibration information
        :return:dic {'sensor1':[],'sensor2':[]}
        '''
        sensor_calib_dic = {}
        for sensor in self.sensor_list:
            sensor_calib_dic[sensor] = {}
            sensor_calib_list = get_name_list(self.calib_path,sensor)
            assert len(sensor_calib_list) >= 1, 'THE CALIB SHOULD AT LEAST HAVE ONE FILE!'
            for sensor_calib in sensor_calib_list:
                if ".json" in sensor_calib:
                    sensor_calib_dic[sensor][sensor_calib.split(".")[0]] = {}
                    sensor_calib_dic[sensor][sensor_calib.split(".")[0]] = loadjson(get_whole_path(self.calib_path,sensor,sensor_calib))
                else:
                    continue
        return sensor_calib_dic

    def box_in_image(self,box_corners,sensor_name,img_shape,prefix_name):
        '''
        judge the box in the camera image or not
        :param box_corners: the corners of the box
        :param sensor_name: the name of the sensor
        :param img_shape: the shape of the image
        :param prefix_name: the prefix of the file
        :return: np.array 8*2 ,bool
        '''
        corners_in_img, corners_in_img_mask = self.proj_points_to_img(sensor_name,box_corners,img_shape,prefix_name)
        if all(corners_in_img_mask):
            return corners_in_img,True
        else:
            return np.array([0]),False


    def proj_points_to_lidar(self,points,calib_info):
        '''
        project the points in other sensor coord system to the lidar coord system
        :param points: points in other sensor coord system
        :param calib_info: the calibration info of the sensors
        :return: np.array 4*n
        '''
        ones = np.ones(points.shape[1]).reshape(-1, points.shape[1])
        points = np.concatenate([points, ones], axis=0)
        rotation = calib_info['rotation']
        R = euler_angle_to_rotate_matrix_3by3(rotation)
        trans = np.array(calib_info['translation']).reshape([-1, 1])
        R = np.concatenate([R, trans], axis=-1)
        R = np.concatenate([R, np.array([0, 0, 0, 1]).reshape([1, -1])], axis=0)
        points_lidar = np.matmul(R, points)
        return points_lidar

    def calib_radar_to_lidar(self,points,sensor_name,points_type="tracks"):
        '''
        project the points of radar to lidar
        :param points: the points in radar
        :param sensor_name: the name of the sensor
        :param points_type: the type of the points in radar
        :return: np.array  4*n
        '''
        if points_type == "tracks":
            calib_file_prefix = "".join(('tracks_',sensor_name))
        else:
            calib_file_prefix = "".join(('points_', sensor_name))
        radar_calib = self.sensor_calib_dic['radar'][calib_file_prefix]
        points_lidar = self.proj_points_to_lidar(points,radar_calib)
        return points_lidar

    def calib_lidar_to_img(self,sensor_name, points):
        '''
        project the points in Lidar to the image
        :param sensor_name: the name of sensor
        :param points: the corner points in lidar,3*8
        :return: np.array 8*2
        '''
        assert points.shape[0] == 3, "The DIMENSION of one point should be X Y Z"
        ones = np.ones(points.shape[1]).reshape(-1, points.shape[1])
        points = np.concatenate([points, ones], axis=0)

        if "aux" in sensor_name:
            sensor = sensor_name[0:str_all_index(sensor_name, "_")[1]]
            sub_sensor = sensor_name[str_all_index(sensor_name, "_")[1] + 1:]
        else:
            sensor = sensor_name[0:str_all_index(sensor_name, "_")[0]]
            sub_sensor = sensor_name[str_all_index(sensor_name, "_")[0] + 1:]
        cam_calib_sta = self.sensor_calib_dic[sensor][sub_sensor]
        cam_extrinsic = np.array(cam_calib_sta["extrinsic"]).reshape(4, 4)
        cam_intrinsic = np.array(cam_calib_sta["intrinsic"]).reshape(3, 3)
        img_pos_2 = np.matmul(cam_extrinsic, points)
        img_pos_3d = img_pos_2[:3, :]
        img_pos_2d = np.matmul(cam_intrinsic, img_pos_3d)
        depth_mask = img_pos_2d[2, :] >= 0
        img_pos_2d[0, :] = img_pos_2d[0, :] / img_pos_2d[2:, :]
        img_pos_2d[1, :] = img_pos_2d[1, :] / img_pos_2d[2:, :]
        return img_pos_2d,depth_mask

    def proj_points_to_img(self, sensor_name, points, img_shape, prefix_name):
        '''
        project the box to image
        :param sensor_name: the name of sensor
        :param points: the corner points in lidar,3*8
        :param img_shape: the shape of image
        :param prefix_name: the prefix of the file
        :return:
        '''
        img_pos_2d, depth_mask = self.calib_lidar_to_img(sensor_name, points)

        if (np.sum(img_pos_2d[0, :] <= 0)) <= 4:
            img_pos_2d[0, :][np.argwhere(img_pos_2d[0, :] <= 0)] = 0
        if (np.sum(img_pos_2d[0, :] >= img_shape[1])) <= 4:
            img_pos_2d[0, :][np.argwhere(img_pos_2d[0, :] >= img_shape[1])] = img_shape[1]
        if (np.sum(img_pos_2d[1, :] <= 0)) <= 4:
            img_pos_2d[1, :][np.argwhere(img_pos_2d[1, :] <= 0)] = 0
        if (np.sum(img_pos_2d[1, :] >= img_shape[0])) <= 4:
            img_pos_2d[1, :][np.argwhere(img_pos_2d[1, :] >= img_shape[0])] = img_shape[0]

        width_mask = np.logical_and(img_pos_2d[0, :] >= 0, img_pos_2d[0, :] <= img_shape[1])  # img width
        height_mask = np.logical_and(img_pos_2d[1, :] >= 0, img_pos_2d[1, :] <= img_shape[0])  # img height
        xy_mask = np.logical_and(width_mask, height_mask)
        points_in_img_mask = np.logical_and(xy_mask, depth_mask)
        points_in_img = img_pos_2d[0:2, :].T[points_in_img_mask]
        return points_in_img.T, points_in_img_mask



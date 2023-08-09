#-*-coding:utf-8-*-

'''
the common utils of the sustech scape dataset
author:Tim Liu

'''

import numpy as np
import json
import os
from math import degrees,radians,sin,cos


def loadjson(json_path):
    '''
    load the json file
    :param json_path: the path of the specified json file of the dataset
    :return: list or dict
    '''
    assert os.path.exists(json_path), 'THE json_path DOES NOT EXIT.PLEASE CHECK!'
    with open(json_path) as f:
        json_info = json.load(f)
    return json_info


def euler_angle_to_rotate_matrix_3by3(eu):
    '''
    get the rotation matrix according to the "rotation" in the calibration file
    :param eu: euler angle in the calibration file
    :return: np.array 3*3
    '''

    # Calculate rotation about x axis
    R_x = np.array([
        [1, 0, 0],
        [0, cos(eu[0]), -sin(eu[0])],
        [0, sin(eu[0]), cos(eu[0])]
    ])
    # Calculate rotation about y axis
    R_y = np.array([
        [cos(eu[1]), 0, sin(eu[1])],
        [0, 1, 0],
        [-sin(eu[1]), 0, cos(eu[1])]
    ])
    # Calculate rotation about z axis
    R_z = np.array([
        [cos(eu[2]), -sin(eu[2]), 0],
        [sin(eu[2]), cos(eu[2]), 0],
        [0, 0, 1]])
    R = np.matmul(R_x, np.matmul(R_y, R_z))
    return R

def get_whole_path(*args):
    '''
    get the whole path of some pathes
    :param args: the sub dir name
    :return: wholepath: str
    '''
    whole_path = ""
    for name in args:
            whole_path = os.path.join(whole_path,name)
    return whole_path

def radian_to_degree(radian):
    '''
    define the function,whcih is used to transfer the radian to the angle
    :param radian:float
    :return float
    '''
    angle = degrees(radian)
    return angle

def degree_to_radian(angle):
    '''
    defien the function,which is used to transfer the degree to the radian
    :param float
    :return radian
    '''
    radian = radians(angle)
    return radian

def get_name_list(rootpath,*args):
    '''
    get the nama list in the given path
    :param rootpath: the root path
    :param args: thr subdir which in the rootpath,and not only one
    :return:namelist:[]
    '''
    whole_path = rootpath
    for name in args:
            if "." in name:
                    continue
            whole_path = get_whole_path(whole_path,name)
    # print(whole_path)
    assert os.path.exists(whole_path), 'THE PATH DOES NOT EXIST.PLEASE CHECK!'
    name_list = os.listdir(whole_path)
    return name_list

def str_all_index(str_,a):
    '''
    get index of the given a
    :param str_: string
    :param a: sub-string in the str_
    :return: index_list : list
    '''
    index_list = []
    start = 0
    while True:
        x = str_.find(a, start)
        if x > -1:
            start = x + 1
            index_list.append(x)
        else:
            break
    return index_list


if __name__ == '__main__':
    # color = classname_to_color["Car"]
    a = "aux_camera_front"
    a1 = "aux_camera_front_right"
    a2 = "camera_front"
    a3 = "camera_front_right"
    b = str_all_index(a,"_")
    b1 = str_all_index(a1,"_")
    b2 = str_all_index(a2, "_")
    b3 = str_all_index(a3, "_")
    c = a[0:b[1]]
    c1 = a1[0:b1[1]]
    c2 = a2[0:b2[0]]
    c3 = a3[0:b3[0]]

    print(c)
    print("aaaa")
# coding:utf-8

'''
Name  : get_iou.py
Author: Tim Liu

Desc:
'''

import numpy as np
from shapely.geometry import Polygon

def check_orthogonal(a,b,c):
    '''
    check that vector (b - a) is orthogonal to the vector (c - a)
    :param a:
    :param b:
    :param c:
    :return:
    '''
    q = b[0] - a[0]
    p = c[0] - a[0]
    r = b[1] - a[1]
    s = c[1] - a[1]
    p1 = q * p
    r1 = r * s
    t = p1 + r1

    res = np.isclose((b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1]), 0)
    # print(res)
    return res
    # return np.isclose((b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1]), 0)

def calculate_ground_bbox_coords(bbox):
    """
    assume that the 3D box has lower plane parallel to the ground.
    :param bbox Box
    """

    length,width = bbox.scale[0],bbox.scale[1]
    center_x,center_y,center_z = bbox.center[0],bbox.center[1],bbox.center[2]
    rotation_matrix = bbox.get_rotate_matrix

    cos_angle = rotation_matrix[0, 0]
    sin_angle = rotation_matrix[0, 1]

    point_0_x = center_x + length / 2 * cos_angle + width / 2 * sin_angle
    point_0_y = center_y + length / 2 * sin_angle - width / 2 * cos_angle

    point_1_x = center_x + length / 2 * cos_angle - width / 2 * sin_angle
    point_1_y = center_y + length / 2 * sin_angle + width / 2 * cos_angle

    point_2_x = center_x - length / 2 * cos_angle - width / 2 * sin_angle
    point_2_y = center_y - length / 2 * sin_angle + width / 2 * cos_angle

    point_3_x = center_x - length / 2 * cos_angle + width / 2 * sin_angle
    point_3_y = center_y - length / 2 * sin_angle - width / 2 * cos_angle

    point_0 = point_0_x, point_0_y
    point_1 = point_1_x, point_1_y
    point_2 = point_2_x, point_2_y
    point_3 = point_3_x, point_3_y

    assert check_orthogonal(point_0, point_1, point_3)
    assert check_orthogonal(point_1, point_0, point_2)
    assert check_orthogonal(point_2, point_1, point_3)
    assert check_orthogonal(point_3, point_0, point_2)

    ground_bbox_coords = Polygon(
        [
            (point_0_x, point_0_y),
            (point_1_x, point_1_y),
            (point_2_x, point_2_y),
            (point_3_x, point_3_y),
            (point_0_x, point_0_y),
        ]
    )

    return ground_bbox_coords

def get_height_intersection(bbox1,bbox2):
    '''
    get the intersection of height between two boxes
    :param bbox1: Box
    :param bbox2: Box
    :return:
    '''
    min_z1 = bbox1.center[2] - bbox1.scale[2]/2
    max_z1 = bbox1.center[2] + bbox1.scale[2]/2
    min_z2 = bbox2.center[2] - bbox2.scale[2] / 2
    max_z2 = bbox2.center[2] + bbox2.scale[2] / 2
    min_z = max(min_z1, min_z2)
    max_z = min(max_z1, max_z2)
    return max(0,max_z - min_z)

def get_area_intersection(bbox1, bbox2):
    '''
    get the intersection of area between two boxes
    :param bbox1:Box
    :param bbox2:Box
    :return:
    '''
    ground_bbox1 = calculate_ground_bbox_coords(bbox1)
    ground_bbox2 = calculate_ground_bbox_coords(bbox2)
    result = ground_bbox1.intersection(ground_bbox2).area
    assert result <= bbox1.scale[0] * bbox1.scale[1] + 0.0001
    assert result <= bbox2.scale[0] * bbox2.scale[1] + 0.0001
    return result

def get_intersection(bbox1,bbox2):
    '''
    get the intersection between two boxes
    :param bbox1:Box
    :param bbox2:Box
    :return:
    '''
    height_intersection = get_height_intersection(bbox1,bbox2)
    area_intersection = get_area_intersection(bbox1, bbox2)
    return height_intersection * area_intersection

def get_volume(bbox):
    '''
    get the volume of box
    :param bbox:Box
    :return:
    '''
    volume = np.prod(bbox.scale)
    return volume

def get_iou(bbox1,bbox2):
    '''
    get iou between two boxes
    :param bbox1:Box
    :param bbox2:Box
    :return:
    '''
    intersection = get_intersection(bbox1,bbox2)
    volume_bbox1 = get_volume(bbox1)
    volume_bbox2 = get_volume(bbox2)
    union = volume_bbox1 + volume_bbox2 - intersection
    iou = np.clip(intersection / union, 0, 1)
    return iou

def get_ious(gt_boxes,predicted_box):
    return [get_iou(x['bbox'],predicted_box) for x in gt_boxes]
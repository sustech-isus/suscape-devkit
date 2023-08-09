#-*-coding:utf-8-*-

'''
2d eval
author:Tim Liu
Ref: Licensed under The MIT License
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np
from suscape.dataset.dataset import Dataset
from utils.common_utils import get_whole_path,get_name_list,loadjson
from eval.detection_2d.det_eval_2d_util import eval


class Box2D(object):
    '''
    the data class of 2D Box
    '''
    def __init__(self,bbox,name,score = 0.0):
        self.bbox = bbox
        self.name = name
        self.score = score

class DetEval2D(object):
    '''
    base class of the 2d detection eval
    '''
    def __init__(self,test_path,class_names):
        '''
        init the DetEval2D class
        :param test_path: the path of the dataset
        :param class_names: the name of classes will be tested
        '''
        assert test_path != None, 'Error,Please init the class with the path of the test dataset'
        self.test_dataset = Dataset(test_path)
        self.class_names = class_names
        self.sub_sensor_list = ['front','front_left','front_right','rear','rear_left','rear_right']

    def get_anno(self,sensor_name):
        '''
        get the annotations of the test dataset
        :param sensor_name: the sensor name (camera or aux_camera)
        :return: []:[{'frame_name':str,bbox:Box2D},{},{}]
        '''
        annos_info = []
        camera_type = sensor_name.split('_')[0]
        camera_name = sensor_name.split('_')[1]
        for scene_name in self.test_dataset.label_fusion_path_dic.keys():
            anno_folder_path = get_whole_path(self.test_dataset.label_fusion_path_dic[scene_name],camera_type,camera_name)
            anno_name_list = get_name_list(anno_folder_path)
            for anno_name in anno_name_list:
                anno_path = get_whole_path(get_whole_path(anno_folder_path,anno_name))
                anno_info = loadjson(anno_path)
                objs = anno_info['objs']
                for obj in objs:
                    frame_info = {}
                    bbox = Box2D([obj['rect']['x1'],obj['rect']['y1'],obj['rect']['x2'],obj['rect']['y2']],
                                 obj['obj_type'])
                    frame_info['name'] = "".join((scene_name, '_', anno_info['cameraType'], '_',
                                          anno_info['cameraName'], '_', anno_info['frame']))
                    frame_info['bbox'] = bbox
                    annos_info.append(frame_info)
        return annos_info

    def get_det_res(self,res_json_path):
        '''
        get the detection result info from the result txt file
        :param res_json_path: the json file path of the detection result
        :return: []:[{'frame_name': str,'bbox':Box2D},{}]
        '''
        assert os.path.exists(res_json_path), 'Error: The result file does not exist!'
        res_info = loadjson(res_json_path)
        res_eval = []
        for frame_info in res_info:
            name = frame_info['name']
            bboxes = frame_info['det_res'] # [{'rect':{'x1':,'y1':,'x2':,'y2':,},'obj_type':...}]
            for info in bboxes:
                frame_res_info = {}
                bbox = Box2D([info['rect']['x1'], info['rect']['y1'], info['rect']['x2'], info['rect']['y2']],
                             info['obj_type'],score = info['score'])
                frame_res_info['name'] = name
                frame_res_info['bbox'] = bbox
                res_eval.append(frame_res_info)
        return res_eval

    def eval(self,res_json_path,sensor_name,iou_thresh,use_07_metric = True):
        '''
        the main function of the evaluation
        :param res_json_path: the json file path of the detection result
        :param sensor_name: the sensor name (camera or aux_camera)
        :param iou_thresh: the thresh of the iou to judge the tp of fp
        :param use_07_metric:
        :return:
        '''
        metrics = {}
        gt_info = self.get_anno(sensor_name)
        det_info = self.get_det_res(res_json_path)
        average_precisions = eval(det_info,gt_info,self.class_names,iou_thresh,use_07_metric)
        mAPs = np.mean(average_precisions)
        metrics['average_precisions'] = average_precisions.tolist()
        metrics['Final mAP'] = float(mAPs)
        metrics['class_names'] = self.class_names
        return metrics


if __name__ == '__main__':

    test_data_path = './example/2d_metric'
    class_names = ["Car","Pedestrian"]
    eval_2d = DetEval2D(test_data_path,class_names)
    res_json_path = './metric_test/2d/camera/res_front.json'
    sensor_name = 'camera_front'
    metrics = eval_2d.eval(res_json_path,sensor_name,0.5)
    for key in metrics.keys():
        print(key,':' ,metrics[key])

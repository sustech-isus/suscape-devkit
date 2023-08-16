# coding:utf-8

'''
Name  : evaluate.py
Author: Tim Liu

Desc:
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import numpy as np
from ...dataset import Dataset
from ...utils.common_utils import get_whole_path,loadjson,get_name_list
from ...data.box import Box
from .accumulate import eval



class DetEval3D(object):
    '''
    base class of the 3d detection eval
    '''

    def __init__(self,test_path,class_names):
        '''
        init the class
        :param test_path: path of test dataset
        :param class_names: the class names need to be test
        '''
        assert test_path != None , 'Error,Please init the class with the path of the test dataset'
        self.test_dataset = Dataset(test_path)
        self.class_names = class_names

    def get_box(self,box):
        '''
        get box according to the json info
        :param box: the box in json file
        :return: box: Box
        '''
        center = [box["psr"]["position"]["x"], box["psr"]["position"]["y"], box["psr"]["position"]["z"]]
        scale = [box["psr"]["scale"]["x"], box["psr"]["scale"]["y"], box["psr"]["scale"]["z"]]
        rotation = np.array([box["psr"]["rotation"]["x"], box["psr"]["rotation"]["y"], box["psr"]["rotation"]["z"]])
        name = box["obj_type"]
        if 'obj_id' not in box.keys():
            obj_id = 0
        else:
            obj_id = int(box["obj_id"])
        if 'score' not in box.keys():
            score = 0.00
        else:
            score = box['score']
        box = Box(center=center,size=scale,rotation = rotation,score=score,obj_id = obj_id,name=name)
        return box

    def get_annos_dataset(self):
        '''
        get the annotations of the test dataset
        :return: List [{'name': str scene_frame,'bbox':Box}]
        '''
        annos_info = []
        for scene_name in self.test_dataset.label_path_dic.keys():
            label_folder_path = self.test_dataset.label_path_dic[scene_name]
            label_list = get_name_list(label_folder_path)
            assert len(label_list) > 0, "The len of the label list should not be 0,Please check the label path!!!!"
            for label_name in label_list:
                anno_path = get_whole_path(label_folder_path, label_name)
                anno_info = loadjson(anno_path)
                if type(anno_info) is list:
                    anno_info = anno_info
                else:
                    anno_info = anno_info['objs']
                for info in anno_info:
                    frame_info = {}
                    box = self.get_box(info)
                    frame_info['name'] = scene_name + '_' + label_name[:-5]
                    frame_info['bbox'] = box
                    annos_info.append(frame_info)
        return annos_info

    def get_det_res(self,res_json_path):
        '''
        get the detect result info from the result json file
        :param res_json_path: the path of the result json file
        :return: [{'name': str scene_frame,'bboxes':Box}]
        '''
        assert os.path.exists(res_json_path), 'Error: The result file does not exist!'
        res_info = loadjson(res_json_path)
        res_eval = []
        for frame in res_info:
            name = frame['scene'] + '_' + frame['frame']
            bboxes = frame['objs'] # [{psr:{'position':,'scale':,'rotation':},'score':,'obj_type':},...]
            for info in bboxes:
                frame_res_info = {}
                box = self.get_box(info)
                frame_res_info['name'] = name
                frame_res_info['bbox'] = box
                res_eval.append(frame_res_info)
        return res_eval


    def eval(self,res_json_path):
        '''
        eval the algorithm to get the mAp
        :param res_json_path: the detection result json file
        :return:
        '''

        print('Begin to accumulating the detection metric')
        annos_info = self.get_annos_dataset()
        res_eval = self.get_det_res(res_json_path)
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        metrics = {}
        average_precisions = eval(annos_info,res_eval,self.class_names,iou_thresholds)
        APs_data = [['IOU', 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]
        mAPs = np.mean(average_precisions, axis=0)
        mAPs_cate = np.mean(average_precisions, axis=1)
        final_mAP = np.mean(mAPs)
        metrics['average_precisions'] = average_precisions.tolist()
        metrics['mAPs'] = mAPs.tolist()
        metrics['Final mAP'] = float(final_mAP)
        metrics['class_names'] = self.class_names
        # metrics['mAPs_cate'] = mAPs_cate.tolist()
        return metrics

if __name__ == "__main__":
    eval_test = DetEval3D('./example/3d_metric',["Car","Pedestrian"])
    metric = eval_test.eval('./metric_test/3d/res.json')
    for key in metric.keys():
        print(key, ':', metric[key])
# coding:utf-8

'''
Name  : evaluate.py
Author: Tim Liu

Desc:
'''

import numpy as np
from ...dataset import SuscapeDataset
from ...data.box import Box
from ...eval.detection_3d.accumulate import eval

CLASSES = ['Car', 'Pedestrian', 'ScooterRider', 'Truck', 'Scooter',
                'Bicycle', 'Van', 'Bus', 'BicycleRider', 
                'Trimotorcycle',
                ]

def format_ann_box(box):
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
            obj_id = ""
        else:
            obj_id = str(box["obj_id"])
        if 'score' not in box.keys():
            score = 0.00
        else:
            score = box['score']
        box = Box(center=center,size=scale,rotation = rotation,score=score,obj_id = obj_id,name=name)
        return box


def flatten_anns(anns):
    '''
    flatten the ann to the format of the eval function
    :param ann: the ann in json file
    :return: the ann in the format of the eval function
    '''
    ann_list = []
    
    for frame in anns:
        name = frame['scene'] + '_' + frame['frame']
        
        for info in frame['objs']:
            box = {
                'name': name,
                'bbox': format_ann_box(info)
            }
    
            ann_list.append(box)
    return ann_list

def evaluate(det, gt, classes=CLASSES):
    '''
    eval the algorithm to get the mAp
    :param res_json_path: the detection result json file
    :return:
    '''

    res_eval = flatten_anns(det)
    gt_eval = flatten_anns(gt)

    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metrics = {}
    average_precisions = eval(gt_eval,res_eval,classes,iou_thresholds)
    mAPs = np.mean(average_precisions, axis=0)
    mAPs_cate = np.mean(average_precisions, axis=1)
    final_mAP = np.mean(mAPs)
    metrics['classes'] = classes
    metrics['ious'] = iou_thresholds
    metrics['AP'] = average_precisions.tolist()
    metrics['mAPs'] = mAPs.tolist()
    metrics['mAPs_cate'] = mAPs_cate.tolist()
    metrics['final_mAP'] = float(final_mAP)

    metrics_string = f"{'IOU':<15}"
    for i, iou in enumerate(iou_thresholds):
        metrics_string += f"{iou:<10} "
    metrics_string += "\n"


    for i, c in enumerate(classes):
        metrics_string += f"{c:<15}"
        for j, ap in enumerate(average_precisions[i]):
            metrics_string += f"{ap:<10.4f} "
        metrics_string += "\n"
    
    metrics_string += f"\n{'mAP':<15}"
    for i, mAP in enumerate(mAPs):
        metrics_string += f"{mAP:<10.4f}"
    metrics_string += "\n"

    metrics_string += f"\n{'':<15}{'mAP_cate':<10}\n"
    for i, mAP in enumerate(mAPs_cate):
        metrics_string += f"{classes[i]:<15}{mAP:<10.4f}\n"
    
    metrics_string += f"\n{'Final mAP':<15}{final_mAP:<10.4f}\n"
    
    return metrics, metrics_string

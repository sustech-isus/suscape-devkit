#-*-coding:utf-8-*-

'''
Name  : evaluate.py
Author: Tim Liu
Desc:
'''

import sys
import os
from utils.common_utils import get_whole_path,get_name_list
from .seg_utils import get_anno_png,intersect_and_union,get_mIoU,labels
from PIL import Image
import numpy as np

class SegEval(object):
    '''
    the class of evaluating the semantic segmentation algorithm
    '''
    def __init__(self,test_data_path):
        assert test_data_path != None, 'Error,Please init the class with the path of the test dataset'
        self.test_img_path = get_whole_path(test_data_path,'image')
        self.test_label_path = get_whole_path(test_data_path,'label')

    def get_anno_png(self,file_prefix,label_list):
        '''
        get the png files,the annotation information included
        :param file_prefix: the prefix name of the file
        :return: png file
        '''
        anno_path = get_whole_path(self.test_label_path,"".join((file_prefix,".json")))
        anno_png = get_anno_png(anno_path,label_list)
        return anno_png

    def eval(self,seg_res_path,num_classes = 44, ignore_pixel = 255,label_list = []):
        '''
        eval the segmentation results according to the resulted png
        :param seg_res_path: the root path of the resulted png
        :return:
        '''
        res_png_name_list = get_name_list(seg_res_path)
        test_img_name_list = get_name_list(self.test_img_path)
        assert len(res_png_name_list) == len(test_img_name_list),'ERROR: List of images for test and res are not equal.'
        total_area_intersect = np.zeros((num_classes,), dtype=np.float64)
        total_area_union = np.zeros((num_classes,), dtype=np.float64)
        total_area_res = np.zeros((num_classes,), dtype=np.float64)
        total_area_anno = np.zeros((num_classes,), dtype=np.float64)
        for name in res_png_name_list:
            # print(name)
            res_png_path = get_whole_path(seg_res_path,name)
            res_png = np.array(Image.open(res_png_path))
            # print(np.where(res_png == 10))
            file_prefix = name[:-4]
            anno_png = np.array(self.get_anno_png(file_prefix,label_list))
            area_intersect, area_union, area_res, area_anno = \
                intersect_and_union(res_png, anno_png, num_classes, ignore_pixel)
            total_area_intersect += area_intersect
            total_area_union += area_union
            total_area_res += area_res
            total_area_anno += area_anno
        ret_metrics = get_mIoU(total_area_intersect,total_area_union,total_area_anno)
        return ret_metrics



if __name__ == "__main__":
    eval_test = SegEval('./example/seg')
    metric = eval_test.eval('./metric_test/seg',label_list=labels)
    for key in metric.keys():
        print(key, ':', metric[key])




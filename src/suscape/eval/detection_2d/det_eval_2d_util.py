#-*-coding:utf-8-*-
'''
2d eval
author:Tim Liu
Ref: Licensed under The MIT License
'''
import numpy as np
from collections import defaultdict

def get_ap(rec, prec, use_07_metric=False):
    """
    Compute AP given precision and recall.If use_07_metric is true, uses the 11 point method (default:False).
    :param rec: recall
    :param prec: precision
    :param use_07_metric: use the 07-metric or not
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            #print("the mpre[i - 1] is:" ,mpre[i - 1])
            #print("the mpre[i - 1] is:", mpre[i])
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def group_by_key(data_info,key):
    '''
    group the data by givec key
    :param data_info: data_info: dic
    :param key: given key
    :return: defaultdict(list)
    '''
    groups = defaultdict(list)
    for box in data_info:
        if key == 'bbox':
            groups[box[key].name].append(box)
        else:
            groups[box[key]].append(box)
    return groups

def get_iou(bbox1,bbox2):
    '''
    get the iou between the two boxes
    :param bbox1: [x1,y1,x2,y2]
    :param bbox2: [x1,y1,x2,y2]
    :return:
    '''
    # inters
    ixmin = np.maximum(bbox1.bbox[0], bbox2.bbox[0])
    iymin = np.maximum(bbox1.bbox[1], bbox2.bbox[1])
    ixmax = np.minimum(bbox1.bbox[2], bbox2.bbox[2])
    iymax = np.minimum(bbox1.bbox[3], bbox2.bbox[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((bbox2.bbox[2] - bbox2.bbox[0] + 1.) * (bbox2.bbox[3] - bbox2.bbox[1] + 1.) +
           (bbox1.bbox[2] - bbox1.bbox[0] + 1.) * (bbox1.bbox[3] - bbox1.bbox[1] + 1.) - inters)
    overlaps = inters / uni
    return overlaps

def get_ious(gt_boxes,predicted_box):
    return [get_iou(x['bbox'],predicted_box) for x in gt_boxes]

def eval_single_class(gt_info,predictions,iou_threshold,use_07_metric =True):
    '''
    evaluate the single class
    :param gt_info: the ground truth
    :param predictions: the predicted result
    :param iou_threshold:the threshold of iou
    :param use_07_metric: use the 07-metric or not
    :return:
    '''
    num_gts = len(gt_info)
    gt_name_info = group_by_key(gt_info, 'name')
    all_gt_checked = {
        name: np.zeros(len(boxes))
        for name, boxes in gt_name_info.items()
    }
    predictions = sorted(predictions, key=lambda x: x['bbox'].score, reverse=True)
    num_predictions = len(predictions)
    tps = np.zeros(num_predictions)
    fps = np.zeros(num_predictions)
    for prediction_index, prediction in enumerate(predictions):
        predicted_box = prediction['bbox']
        frame_name = prediction['name']
        max_overlap = -np.inf
        jmax = -1
        if frame_name in gt_name_info:
            frame_gt_boxes = gt_name_info[frame_name]
            # gt_boxes per sample
            frame_gt_checked = all_gt_checked[frame_name]
        else:
            frame_gt_boxes = []
            frame_gt_checked = None
        if len(frame_gt_boxes) > 0:
            overlaps = get_ious(frame_gt_boxes, predicted_box)
            max_overlap = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if max_overlap > iou_threshold:
            if frame_gt_checked[jmax] == 0:
                tps[prediction_index] = 1.0
                frame_gt_checked[jmax] = 1
            else:
                fps[prediction_index] = 1.0
        else:
            fps[prediction_index, ] = 1.0
        # compute precision recall
    fps = np.cumsum(fps)
    tps = np.cumsum(tps)
    recalls = tps / float(num_gts)
    precisions = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
    ap = get_ap(recalls, precisions, use_07_metric)
    return recalls, precisions, ap



def eval(det_info,gt_info,class_names,iou_threshold=0.5,use_07_metric=False):
    '''
    the function to get ap
    :param det_info: the information of the detection result
    :param gt_info: the information of the label
    :param class_names: the class name of the dataset
    :param iou_threshold: the threshold of the iou
    :param use_07_metric: the metric type
    :return:
    '''
    assert 0 <= iou_threshold <= 1,'ERROR: The threshold of iou should be larger than 1 !!!'
    average_precisions = np.zeros(len(class_names))
    gt_by_class_name = group_by_key(gt_info, 'bbox')
    pred_by_class_name = group_by_key(det_info, 'bbox')
    for class_id,class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            recalls, precisions, average_precision = eval_single_class(gt_by_class_name[class_name], pred_by_class_name[class_name],
                iou_threshold,use_07_metric)
            average_precisions[class_id] = average_precision
    return average_precisions
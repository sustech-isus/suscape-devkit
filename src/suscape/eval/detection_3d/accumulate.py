# coding:utf-8

'''
Name  : accumulate.py
Author: Tim Liu
'''

from .utils import group_by_key
from .get_iou import *

def get_envelope(precisions):
    """
    Compute the precision envelope.
    :param precisions
    :return
    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions

def get_ap(recalls, precisions):
    """
    Calculate average precision.
    :param  recalls:
    :param  precisions:
    :return: average precision. float
    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    precisions = get_envelope(precisions)
    # to calculate area under PR curve, look for points where X axis (recall) changes value
    # print(recalls[1:])
    # print(recalls[:-1])
    i = np.where(recalls[1:] != recalls[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap

def eval_single_class(gt_info, predictions, iou_thresholds):
    '''
    evaluate the single class
    :param gt_info: the ground truth
    :param predictions: the predicted result
    :param iou_thresholds: the threshold of iou
    :return:recalls, precisions, aps
    '''
    num_gts = len(gt_info)
    gt_name_info = group_by_key(gt_info,'name')
    all_gt_checked = {
        name: np.zeros((len(boxes), len(iou_thresholds)))
        for name, boxes in gt_name_info.items()
    }
    predictions = sorted(predictions, key=lambda x: x['bbox'].score, reverse=True)
    num_predictions = len(predictions)
    tps = np.zeros((num_predictions, len(iou_thresholds)))
    fps = np.zeros((num_predictions, len(iou_thresholds)))
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
        for i, iou_threshold in enumerate(iou_thresholds):
            if max_overlap > iou_threshold:
                if frame_gt_checked[jmax, i] == 0:
                    tps[prediction_index, i] = 1.0
                    frame_gt_checked[jmax, i] = 1
                else:
                    fps[prediction_index, i] = 1.0
            else:
                fps[prediction_index, i] = 1.0
    # compute precision recall
    fps = np.cumsum(fps, axis=0)
    tps = np.cumsum(tps, axis=0)
    recalls = tps / float(num_gts)
    # avoid divide by zero in case the first detection
    # matches a difficult ground truth
    # sum = tps + fps
    # precisions = tps / np.maximum(sum, np.finfo(np.float64).eps)
    precisions = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)
    aps = []
    for i in range(len(iou_thresholds)):
        recall = recalls[:, i]
        precision = precisions[:, i]
        assert np.all(0 <= recall) & np.all(recall <= 1)
        assert np.all(0 <= precision) & np.all(precision <= 1)
        ap = get_ap(recall, precision)
        aps.append(ap)
    aps = np.array(aps)
    return recalls, precisions, aps

def eval(gt_info,res_info,class_names,iou_thresholds):
    '''
    accumulate the map
    :param gt_info:  the ground truth
    :param res_info: the predicted result
    :param class_names: the class names need to be tested
    :param iou_thresholds: the thresholds of iou
    :return: average_precisions average precision
    '''
    assert all([0 <= iou_th <= 1 for iou_th in iou_thresholds])
    average_precisions = np.zeros((len(class_names), len(iou_thresholds)))
    gt_by_class_name = group_by_key(gt_info, 'bbox')
    pred_by_class_name = group_by_key(res_info, 'bbox')
    for class_id,class_name in enumerate(class_names):
        if class_name in pred_by_class_name:

            if len(gt_by_class_name[class_name]) > 0:
                recalls, precisions, average_precision = eval_single_class(
                    gt_by_class_name[class_name], pred_by_class_name[class_name],
                    iou_thresholds)
                average_precisions[class_id, :] = average_precision
    return average_precisions

if __name__ == "__main__":

    min_overlaps = eval({},{},['Car','Pedestrian'])
    min_overlap = min_overlaps[:,:,0]
    print(min_overlap)

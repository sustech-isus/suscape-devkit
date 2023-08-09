#-*-coding:utf-8-*-
'''
the utils python files,which is used for the eval_seg.py
'''

from PIL import Image, ImageDraw
from utils.common_utils import loadjson
import numpy as np
from collections import OrderedDict



WIDTH = 2048
HEIGHT = 1536

labelName={
    "ego_car": "自车",
    "sky": "天空",
    "concrete_road": "水泥路面",
    "asphalt_road": "柏油路面",
    "asphalt_road": "沥青路面",
    "sand_stone_road": "砂石路面",
    "earth_road": "泥土路面",
    "farmland": "农田",
    "footpath": "人行小路",
    "side_walk": "人行区域",
    "side_walk": "路边人行区域",
    "road_shoulder": "路肩",
    "road_curb": "路沿",
    "water": "水面",
    "grass": "草地",
    "earth": "土地",
    "earth": "泥土坡",
    "stone": "石头",
    "trees": "树木",
    "building": "建筑物",
    "low_wall": "矮墙",
    "fence": "栅栏",
    "electric_light_pole": "路灯/电线杆",
    "electric_tower": "电线塔",
    "electric_wire": "电线",
    "pole": "柱子",
    "road_board": "指示牌",
    "traffic_light": "交通灯",
    "car": "汽车",
    "pedestrian": "行人",
    "truck": "卡车",
    "bus": "大巴",
    "train": "火车",
    "bicycle_rider": "自行车骑行人",
    "scooter_rider": "电动车骑行人",
    "motorcycle_rider": "摩托车骑行人",
    "bicycle": "自行车",
    "scooter": "电动车",
    "motorcycle": "摩托车",
    "trimotocycle": "三轮车",
    "animal": "动物",
    "trash_can": "垃圾桶",
    "cone": "椎形筒",
    "clutter": "杂物(某些场景路边堆积的杂物)",
    "umbrella": "伞或棚",
    "void": "其他(可能无法确定的类别)",
    "error": "缺ID",
}

labels = [
        "void",
        "sky",
        "water",
        "grass",
        "earth",
        "stone",
        "trees",
        "farmland",
        "building",
        "low_wall",
        "fence",
        "concrete_road",
        "asphalt_road",
        "sand_stone_road",
        "earth_road",
        "footpath",
        "side_walk",
        "road_shoulder",
        "road_curb",
        'ego_car',
        "car",
        "pedestrian",
        "truck",
        "bus",
        "train",
        "bicycle_rider",
        "scooter_rider",
        "motorcycle_rider",
        "bicycle",
        "scooter",
        "motorcycle",
        "trimotocycle",
        "animal",
        "electric_light_pole",
        "electric_tower",
        "electric_wire",
        "pole",
        "road_board",
        "traffic_light",
        "trash_can",
        "cone",
        "clutter",
        "umbrella",
        "error",
]

def get_anno_png(anno_path,label_list):
    '''
    get the png with annotation
    @param anno_path: the path of the label
    @return:
    '''
    anno_info = loadjson(anno_path)
    img = Image.new("L", [WIDTH, HEIGHT])
    img1 = ImageDraw.Draw(img)
    for obj in anno_info['objects']:
        pts = list(map(lambda p: (p['x'], p['y']), obj['polygon']))
        img1.polygon(pts, fill=label_list.index(obj['label']))
    return img

def intersect_and_union(res_png,anno_png,num_classes = 42, ignore_pixel = 255):
    '''
    Calculate intersection and Union between the two pngs
    @param res_png: the segmentation result png
    @param anno_png: the annotated png
    @param num_classes: the number of classes of the dataset
    @param ignore_pixel: the pixel should be ignored
    @return:
    '''
    assert not np.any(np.isnan(res_png))
    assert not np.any(np.isnan(anno_png))
    mask = (anno_png != ignore_pixel)
    res = res_png[mask]
    anno = anno_png[mask]
    intersect = res[res == anno]
    area_intersect,_ = np.histogram(intersect.astype(np.float32), bins=num_classes,range=(0,(num_classes-1)))
    area_res,_ = np.histogram(res.astype(np.float64), bins=num_classes,range=(0,(num_classes-1)))
    area_anno,_ = np.histogram(anno.astype(np.float64), bins=num_classes,range=(0,(num_classes-1)))
    area_union = area_res + area_anno - area_intersect
    return area_intersect,area_union,area_res,area_anno

def get_mIoU(total_area_intersect,total_area_union,total_area_anno):
    '''
    get the mIoU
    @param total_area_intersect: the intersect
    @param total_area_union: the union
    @param total_area_res: the area of result
    @param total_area_anno: the area of annotation
    @return:
    '''

    ignore_label = np.where(total_area_anno == 0)[0]
    if len(ignore_label) != 0:
        total_area_anno[ignore_label] = 1
        total_area_union[ignore_label] = 1
        total_area_intersect[ignore_label] = 1
    all_acc = total_area_intersect.sum() / total_area_anno.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_anno
    miou = np.mean(iou)
    ret_metrics['IoU'] = iou
    ret_metrics['Acc'] = acc
    ret_metrics['mIoU'] = miou
    return ret_metrics

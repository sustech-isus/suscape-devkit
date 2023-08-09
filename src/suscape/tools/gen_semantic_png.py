


import argparse
from PIL import Image, ImageDraw
import json
import os

import re


parser = argparse.ArgumentParser(description='start web server for sem-seg')        

parser.add_argument('--data', type=str, default='./data', help="")
parser.add_argument('--scenes', type=str, default='.*', help="")

args = parser.parse_args()


root_dir = args.data

WIDTH=2048
HEIGHT=1536

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

def draw_one_frame(scene_path, frame):
    print(frame)
    label_file = os.path.join(scene_path, 'label', frame+".json")

    with open(label_file)  as f:
        label = json.load(f)

        img = Image.new("L", [WIDTH, HEIGHT]) 
        img1 = ImageDraw.Draw(img)  

        for obj in label['objects']:
            pts = list(map(lambda p: (p['x'], p['y']), obj['polygon']))
            img1.polygon(pts, fill =labels.index(obj['label']))

        img.save(os.path.join(scene_path, 'semantic', frame+'.png'))


def process_scene(scene):
    scene_path = os.path.join(root_dir, scene)

    if not os.path.exists(os.path.join(scene_path, 'semantic')):
        os.mkdir(os.path.join(scene_path, 'semantic'))

    files = os.listdir(os.path.join(scene_path, 'image'))
    files.sort()

    for f in files:
        frame = os.path.splitext(f)[0]
        draw_one_frame(scene_path, frame)



if __name__ == "__main__":
    
    scenes = os.listdir(args.data)
    for scene in scenes:
        if re.fullmatch(args.scenes, scene):
            print(scene)
            process_scene(scene)
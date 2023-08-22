
import os
import json
import re
import numpy as np
from .bin_pcd_reader import BinPcdReader

class SuscapeDataset:
    def __init__(self, root_dir, dir_org="") -> None:
        
        cfg = self._build_dataset_cfgs(root_dir, dir_org)
        self.camera_dir = cfg['camera']
        self.lidar_dir = cfg['lidar']
        self.aux_camera_dir = cfg['aux_camera']
        self.label_dir = cfg['label']
        self.calib_dir = cfg['calib']
        self.desc_dir = cfg['desc']
        self.meta_dir = cfg['meta']
        self.radar_dir = cfg['radar']
        self.aux_lidar_dir = cfg['aux_lidar']
        self.label_fusion_dir = cfg['label_fusion']
        self.lidar_pose_dir = cfg['lidar_pose']

        pass

    def _build_dataset_cfgs(self, root, dir_org):

        if dir_org == '':
            if os.path.exists(root + '/lidar'):
                dir_org = 'by_data_folder'
            else:
                dir_org = 'by_scene'

        dataset_cfg={}
        if dir_org == 'by_scene':
            for d in ['lidar',  'label', 'camera', 'calib', 'aux_lidar', 'aux_camera', 'radar', 'desc', 'meta','label_fusion', 'lidar_pose']:
                dataset_cfg[d] = root
            dataset_cfg['root'] = root
            
        elif dir_org == 'by_data_folder':
            for d in ['lidar',  'label', 'camera', 'calib', 'aux_lidar', 'aux_camera', 'radar', 'desc', 'meta','label_fusion', 'lidar_pose']:
                dataset_cfg[d] =(root + '/' + d)
            dataset_cfg['root'] = root
        
        return dataset_cfg

    def get_all_scene_desc(self, scene_pattern='.*'):
        
        scenes = self.get_scene_names()
        
        descs = {}

        for n in scenes:
            if re.fullmatch(scene_pattern, n):
                try:
                    descs[n] = self.get_scene_desc(n)
                except:
                    print('failed reading scene:', n)
                    raise
        return descs

    def get_scene_names(self):
        scenes = os.listdir(self.lidar_dir)
        scenes.sort()
        return scenes

    

    def get_all_objs(self, s):
      label_folder = os.path.join(self.label_dir, s, "label")
      if not os.path.isdir(label_folder):
        return []
        
      files = os.listdir(label_folder)

      files = filter(lambda x: x.split(".")[-1]=="json", files)


      def file_2_objs(f):
          if  not os.path.exists(f):
            return []
          with open(f) as fd:
              ann = json.load(fd)
              if 'objs' in ann:
                boxes = ann['objs']
              else:
                boxes = ann
              objs = [x for x in map(lambda b: {"category":b["obj_type"], "id": b["obj_id"]}, boxes)]
              return objs

      boxes = map(lambda f: file_2_objs(os.path.join(self.label_dir, s, "label", f)), files)

      # the following map makes the category-id pairs unique in scene
      all_objs={}
      for x in boxes:
          for o in x:
              
              k = str(o["category"])+"-"+str(o["id"])

              if all_objs.get(k):
                all_objs[k]['count']= all_objs[k]['count']+1
              else:
                all_objs[k]= {
                  "category": o["category"],
                  "id": o["id"],
                  "count": 1
                }

      return [x for x in  all_objs.values()]



    
    def get_calib_lidar2cam(self, scene_info, frame, camera_type, camera_name):
        s = scene_info["scene"]
        static_calib = np.array(scene_info["calib"][camera_type][camera_name]["lidar_to_camera"]).reshape([4,4])

        frame_calib_file = os.path.join(self.calib_dir, s, "calib", camera_type, camera_name, frame+".json")
        if not os.path.exists(frame_calib_file):
            return static_calib
        with open(frame_calib_file) as f:
            local_calib = json.load(f)
            trans = np.array(local_calib['lidar_transform']).reshape((4,4))
            lidar2cam = np.matmul(static_calib, trans)
            return lidar2cam
    
    def read_desc(self, s):
        scene_dir = os.path.join(self.desc_dir, s)
        desc = {}
        if os.path.exists(os.path.join(scene_dir, "desc.json")):
            with open(os.path.join(scene_dir, "desc.json")) as f:
                desc = json.load(f)
        
        return desc
        

    def get_scene_info(self, s):
        scene = {
            "scene": s,
            "frames": []
        }

        frames = os.listdir(os.path.join(self.lidar_dir, s, "lidar"))

        frames.sort()

        scene["lidar_ext"]="pcd"
        for f in frames:
            #if os.path.isfile("./data/"+s+"/lidar/"+f):
            filename, fileext = os.path.splitext(f)
            scene["frames"].append(filename)
            scene["lidar_ext"] = fileext

        # point_transform_matrix=[]

        # if os.path.isfile(os.path.join(scene_dir, "point_transform.txt")):
        #     with open(os.path.join(scene_dir, "point_transform.txt"))  as f:
        #         point_transform_matrix=f.read()
        #         point_transform_matrix = point_transform_matrix.split(",")

        
        if os.path.exists(os.path.join(self.desc_dir, s, "desc.json")):
            with open(os.path.join(self.desc_dir, s, "desc.json")) as f:
                desc = json.load(f)
                scene["desc"] = desc

        # calib will be read when frame is loaded. since each frame may have different calib.
        # read default calib for whole scene.
        calib = {}
        if os.path.exists(os.path.join(self.calib_dir, s, "calib")):
            sensor_types = os.listdir(os.path.join(self.calib_dir, s, 'calib'))        
            for sensor_type in sensor_types:
                calib[sensor_type] = {}
                if os.path.exists(os.path.join(self.calib_dir, s, "calib",sensor_type)):
                    calibs = os.listdir(os.path.join(self.calib_dir, s, "calib", sensor_type))
                    for c in calibs:
                        calib_file = os.path.join(self.calib_dir, s, "calib", sensor_type, c)
                        calib_name, ext = os.path.splitext(c)
                        if os.path.isfile(calib_file) and ext==".json": #ignore directories.
                            #print(calib_file)
                            try:
                                with open(calib_file)  as f:
                                    cal = json.load(f)
                                    calib[sensor_type][calib_name] = cal
                            except: 
                                print('reading calib failed: ', f)
                                assert False, f            


        scene["calib"] = calib


        # camera names
        camera = []
        camera_ext = ""
        cam_path = os.path.join(self.camera_dir, s, "camera")
        if os.path.exists(cam_path):
            cams = os.listdir(cam_path)
            for c in cams:
                cam_file = os.path.join(self.camera_dir, s, "camera", c)
                if os.path.isdir(cam_file):
                    camera.append(c)

                    if camera_ext == "":
                        #detect camera file ext
                        files = os.listdir(cam_file)
                        if len(files)>=2:
                            _,camera_ext = os.path.splitext(files[0])

        camera.sort()
        if camera_ext == "":
            camera_ext = ".jpg"
        scene["camera_ext"] = camera_ext
        scene["camera"] = camera


        aux_camera = []
        aux_camera_ext = ""
        aux_cam_path = os.path.join(self.aux_camera_dir, s, "aux_camera")
        if os.path.exists(aux_cam_path):
            cams = os.listdir(aux_cam_path)
            for c in cams:
                cam_file = os.path.join(aux_cam_path, c)
                if os.path.isdir(cam_file):
                    aux_camera.append(c)

                    if aux_camera_ext == "":
                        #detect camera file ext
                        files = os.listdir(cam_file)
                        if len(files)>=2:
                            _,aux_camera_ext = os.path.splitext(files[0])

        aux_camera.sort()
        if aux_camera_ext == "":
            aux_camera_ext = ".jpg"
        scene["aux_camera_ext"] = aux_camera_ext
        scene["aux_camera"] = aux_camera


        # radar names
        radar = []
        radar_ext = ""
        radar_path = os.path.join(self.radar_dir, s, "radar")
        if os.path.exists(radar_path):
            radars = os.listdir(radar_path)
            for r in radars:
                radar_file = os.path.join(self.radar_dir, s, "radar", r)
                if os.path.isdir(radar_file):
                    radar.append(r)
                    if radar_ext == "":
                        #detect camera file ext
                        files = os.listdir(radar_file)
                        if len(files)>=2:
                            _,radar_ext = os.path.splitext(files[0])

        if radar_ext == "":
            radar_ext = ".pcd"
        scene["radar_ext"] = radar_ext
        scene["radar"] = radar

        # aux lidar names
        aux_lidar = []
        aux_lidar_ext = ""
        aux_lidar_path = os.path.join(self.aux_lidar_dir, s, "aux_lidar")
        if os.path.exists(aux_lidar_path):
            lidars = os.listdir(aux_lidar_path)
            for r in lidars:
                lidar_file = os.path.join(self.aux_lidar_dir, s, "aux_lidar", r)
                if os.path.isdir(lidar_file):
                    aux_lidar.append(r)
                    if radar_ext == "":
                        #detect camera file ext
                        files = os.listdir(radar_file)
                        if len(files)>=2:
                            _,aux_lidar_ext = os.path.splitext(files[0])

        if aux_lidar_ext == "":
            aux_lidar_ext = ".pcd"
        scene["aux_lidar_ext"] = aux_lidar_ext
        scene["aux_lidar"] = aux_lidar


        scene["boxtype"] = "psr"

        # lidar_pose
        lidar_pose= {}
        lidar_pose_path = os.path.join(self.lidar_pose_dir, s, "lidar_pose")
        if os.path.exists(lidar_pose_path):
            poses = os.listdir(lidar_pose_path)
            for p in poses:
                p_file = os.path.join(lidar_pose_path, p)
                with open(p_file)  as f:
                        pose = json.load(f)
                        lidar_pose[os.path.splitext(p)[0]] = pose
        
        scene['lidar_pose'] = lidar_pose
        return scene


    def get_frames(self, scene):
        
        frames = os.listdir(os.path.join(self.lidar_dir, scene, "lidar"))
        frames = [os.path.splitext(f)[0] for f in frames]
        frames.sort()
        return frames
    
    def read_label(self, scene, frame):
        "read 3d boxes"
        if not os.path.exists(os.path.join(self.label_dir, scene, 'label')):
            print('label path does not exist', self.label_dir, scene, 'label')
            return {'objs': []}
        
        filename = os.path.join(self.label_dir, scene, "label", frame+".json")   # backward compatible
        
        if os.path.exists(filename):
            if (os.path.isfile(filename)):
                with open(filename,"r") as f:
                    ann=json.load(f)
                    return ann
        else:
            print('label file does not exist', filename)
        return {'objs': []}

    def read_all_labels(self):
        all = []
        for s in self.get_scene_names():
            for f in self.get_frames(s):
                l = self.read_label(s,f)
                l['scene'] = s
                l['frame'] = f
                all.append(l)
        return all
    
    def read_image_annotations(self, scene, frame, camera_type, camera_name):
        filename = os.path.join(self.label_fusion_dir, scene, "label_fusion", camera_type, camera_name, frame+".json")   # backward compatible
        if os.path.exists(filename):
            if (os.path.isfile(filename)):
                with open(filename,"r") as f:
                    ann=json.load(f)
                    #print(ann)          
                    return ann
        return {'objs': []}


    def read_all_image_annotations(self, scene, frame, cameras, aux_cameras):
        ann = {
            "camera": {},
            "aux_camera": {}
        }
        for c in cameras.split(','):
            filename = os.path.join(self.label_fusion_dir, scene, "label_fusion", 'camera', c, frame+".json")   # backward compatible
            if os.path.exists(filename):
                if (os.path.isfile(filename)):
                    with open(filename,"r") as f:
                        ann['camera'][c] = json.load(f)


        for c in aux_cameras.split(','):
            filename = os.path.join(self.label_fusion_dir, scene, "label_fusion", 'aux_camera', c, frame+".json")   # backward compatible
            if os.path.exists(filename):
                if (os.path.isfile(filename)):
                    with open(filename,"r") as f:
                        ann['aux_camera'][c] = json.load(f)

        return ann

    def read_lidar_pose(self, scene, frame):
        filename = os.path.join(self.lidar_pose_dir, scene, "lidar_pose", frame+".json")
        if (os.path.isfile(filename)):
            with open(filename,"r") as f:
                p=json.load(f)
                return p
        else:
            return None
    
    def read_ego_pose(self, scene, frame):
        filename = os.path.join(self.lidar_pose_dir, scene, "ego_pose", frame+".json")
        if (os.path.isfile(filename)):
            with open(filename,"r") as f:
                p=json.load(f)
                return p
        else:
            return None
    def read_calib(self, scene, frame):
        'read static calibration, all extrinsics are sensor to lidar_top'
        calib = {}

        if not os.path.exists(os.path.join(self.calib_dir, scene, "calib")):
            return calib

        calib_folder = os.path.join(self.calib_dir, scene, "calib")
        sensor_types = os.listdir(calib_folder)
        
        for sensor_type in sensor_types:
            this_type_calib = {}
            sensors = os.listdir(os.path.join(calib_folder, sensor_type))
            for sensor in sensors:
                sensor_file = os.path.join(calib_folder, sensor_type, sensor, frame+".json")
                if os.path.exists(sensor_file) and os.path.isfile(sensor_file):
                    with open(sensor_file, "r") as f:
                        p=json.load(f)
                        this_type_calib[sensor] = p
            if this_type_calib:
                calib[sensor_type] = this_type_calib

        return calib
    def read_lidar(self, scene, frame):
        "returns a numpy array of shape [N, 4], x,y,z,intensity"
        lidar_file = os.path.join(self.lidar_dir, scene, "lidar", frame+".pcd")
        if os.path.exists(lidar_file):
            pcd_reader = BinPcdReader(lidar_file)
            
            bin_data = np.stack([pcd_reader.pc_data[x] for x in ['x', 'y', 'z', 'intensity']], axis=-1)
            bin_data = bin_data[(bin_data[:,0]!=0) & (bin_data[:,1]!=0) & (bin_data[:,2]!=0)]
            
            return bin_data
        else:
            raise FileNotFoundError(lidar_file)
            return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='suscape dataset test')        
    parser.add_argument('data', type=str, help='data folder')
    args = parser.parse_args()

    dataset = SuscapeDataset(args.data)
    print(len(dataset.get_scene_names()), 'scenes')
    print(dataset.get_scene_info("scene-000000"))
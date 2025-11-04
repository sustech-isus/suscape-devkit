
import os
import json
import re
import numpy as np
from .bin_pcd_reader import BinPcdReader
import math
from PIL import Image

def euler_angle_to_rotate_matrix(eu, t):  # ZYX order.
    theta = eu
    #Calculate rotation about x axis
    R_x = np.array([
        [1,       0,              0],
        [0,       math.cos(theta[0]),   -math.sin(theta[0])],
        [0,       math.sin(theta[0]),   math.cos(theta[0])]
    ])

    #Calculate rotation about y axis
    R_y = np.array([
        [math.cos(theta[1]),      0,      math.sin(theta[1])],
        [0,                       1,      0],
        [-math.sin(theta[1]),     0,      math.cos(theta[1])]
    ])

    #Calculate rotation about z axis
    R_z = np.array([
        [math.cos(theta[2]),    -math.sin(theta[2]),      0],
        [math.sin(theta[2]),    math.cos(theta[2]),       0],
        [0,               0,                  1]])

    R = np.matmul(R_x, np.matmul(R_y, R_z))

    t = t.reshape([-1,1])
    R = np.concatenate([R,t], axis=-1)
    R = np.concatenate([R, np.array([0,0,0,1]).reshape([1,-1])], axis=0)
    return R

# default rotation order: xyz
def euler_angle_to_rotate_matrix_3x3(eu):
    theta = eu
    #Calculate rotation about x axis
    R_x = np.array([
        [1,       0,              0],
        [0,       math.cos(theta[0]),   -math.sin(theta[0])],
        [0,       math.sin(theta[0]),   math.cos(theta[0])]
    ])

    #Calculate rotation about y axis
    R_y = np.array([
        [math.cos(theta[1]),      0,      math.sin(theta[1])],
        [0,                       1,      0],
        [-math.sin(theta[1]),     0,      math.cos(theta[1])]
    ])

    #Calculate rotation about z axis
    R_z = np.array([
        [math.cos(theta[2]),    -math.sin(theta[2]),      0],
        [math.sin(theta[2]),    math.cos(theta[2]),       0],
        [0,               0,                  1]])

    R = np.matmul(R_x, np.matmul(R_y, R_z))
    return R


#  euler_angle_to_rotate_matrix(np.array([0, np.pi/3, np.pi/2]), np.array([1,2,3]))
def psr_to_xyz(p,s,r):
    trans_matrix = euler_angle_to_rotate_matrix(r, p)

    x=s[0]/2
    y=s[1]/2
    z=s[2]/2
    

    local_coord = np.array([
        x, y, -z, 1,   x, -y, -z, 1,  #front-left-bottom, front-right-bottom
        x, -y, z, 1,   x, y, z, 1,    #front-right-top,   front-left-top

        -x, y, -z, 1,   -x, -y, -z, 1,#rear-left-bottom, rear-right-bottom
        -x, -y, z, 1,   -x, y, z, 1,  #rear-right-top,   rear-left-top
        
        #middle plane
        #0, y, -z, 1,   0, -y, -z, 1,  #rear-left-bottom, rear-right-bottom
        #0, -y, z, 1,   0, y, z, 1,    #rear-right-top,   rear-left-top
        ]).reshape((-1,4))

    world_coord = np.matmul(trans_matrix, np.transpose(local_coord))
    
    return np.transpose(world_coord[0:3,:])

def box_to_nparray(box):
    return np.array([
        [box["position"]["x"], box["position"]["y"], box["position"]["z"]],
        [box["scale"]["x"], box["scale"]["y"], box["scale"]["z"]],
        [box["rotation"]["x"], box["rotation"]["y"], box["rotation"]["z"]],
    ])



def box_position(box):
    return np.array(
        [box["position"]["x"], box["position"]["y"], box["position"]["z"]])

def box3d_to_corners(b):
    box = box_to_nparray(b["psr"])    
    box3d = psr_to_xyz(box[0], box[1], box[2])
    return box3d


def proj_pts3d_to_img(pts, extrinsic_matrix, intrinsic_matrix, width=None, height=None):

    #print(pts.shape, extrinsic_matrix.shape)
    
    pts = np.concatenate([pts, np.ones([pts.shape[0],1])], axis=1)
    imgpos = np.matmul(pts, np.transpose(extrinsic_matrix))

    # rect matrix shall be applied here, for kitti

    imgpos3 = imgpos[:, :3]
    
    
    #imgpos3 = imgpos3[filter_in_front]        

    # if imgpos3.shape[0] < 1:
    #     return None

    imgpos2 = np.matmul(imgpos3, np.transpose(intrinsic_matrix))

    imgfinal = imgpos2/imgpos2[:,2:3]

    filter_in_frontview = imgpos3[:,2] > 0
    

    if width and height:
        filter_in_image = (imgfinal[:,0] >= 0) & (imgfinal[:,0] < width) & (imgfinal[:,1] >= 0) & (imgfinal[:,1] < height)
        return imgfinal[filter_in_frontview & filter_in_image]
    else:
        return imgfinal, filter_in_frontview


def combine_calib(static_calib, local_calib):
    trans = np.array(local_calib['lidar_transform']).reshape((4,4))
    extrinsic = np.matmul(static_calib['extrinsic'], trans)
    return (extrinsic,static_calib['intrinsic'])


def get_calib_for_frame(scene_path, meta, camera_type, camera, frame):
    local_calib_file = os.path.join(scene_path, 'calib', camera_type, camera, frame+".json")
    if os.path.exists(local_calib_file):
        with open (local_calib_file) as f:
            local_calib = json.load(f)
            (extrinsic, intrinsic) = combine_calib(meta['calib'][camera_type][camera], local_calib)
            return (extrinsic, intrinsic)
    else:
        (extrinsic,intrinsic) =  (meta['calib'][camera_type][camera]['extrinsic'], meta['calib'][camera_type][camera]['intrinsic'])
        return (extrinsic, intrinsic)
        
def choose_best_camera_for_obj(obj, scene_path, meta, camera_type, cameras, frame):
    obj_pos = np.array([box_position(obj['psr'])])

    dists =[]
    filters = []
    for camera in cameras:
        extrinsic, intrinsic = get_calib_for_frame(scene_path, meta, camera_type, camera, frame)
        p, filter_in_frontview = proj_pts3d_to_img(obj_pos, extrinsic, intrinsic)
        w = meta[camera_type][camera]['width']
        h = meta[camera_type][camera]['height']
        
        if np.any(filter_in_frontview):
            dis_squared = (p[0][0]-w/2) * (p[0][0]-w/2) + (p[0][1]-h/2)*(p[0][1]-h/2)
        else:
            dis_squared = (w*w+h*h) * 10

        dists.append(dis_squared)        
        filters.append(filter_in_frontview[0])
    
    best_idx = np.argmin(dists)

    if filters[best_idx]:
        return cameras[best_idx]
    else:
        return None
    



def crop_box_pts(pts, box, ground_level=0.3):
    """ return points in box coordinate system"""
    eu = [box['psr']['rotation']['x'], box['psr']['rotation']['y'], box['psr']['rotation']['z']]
    trans_matrix = euler_angle_to_rotate_matrix_3x3(eu)

    center = np.array([box['psr']['position']['x'], box['psr']['position']['y'], box['psr']['position']['z']])
    box_pts = np.matmul((pts - center), (trans_matrix))   # M^-1 * x = M^T * x = (x^T * M)^T

   
    if box['psr']['scale']['z'] < 2:
        ground_level = box['psr']['scale']['z'] * 0.15

    filter =  (box_pts[:, 0] < box['psr']['scale']['x']/2) & (box_pts[:, 0] > - box['psr']['scale']['x']/2) & \
             (box_pts[:, 1] < box['psr']['scale']['y']/2) & (box_pts[:, 1] > - box['psr']['scale']['y']/2)

    topfilter =    filter & (box_pts[:, 2] < box['psr']['scale']['z']/2) & (box_pts[:, 2] >= - box['psr']['scale']['z']/2 + ground_level)

    groundfilter = filter & (box_pts[:, 2] < -box['psr']['scale']['z']/2+ground_level) & (box_pts[:, 2] > - box['psr']['scale']['z']/2)
    
    
    return [box_pts[topfilter], box_pts[groundfilter], ground_level]



def crop_pts(pts, box):
    """ return points in lidar coordinate system"""
    eu = [box['psr']['rotation']['x'], box['psr']['rotation']['y'], box['psr']['rotation']['z']]
    trans_matrix = euler_angle_to_rotate_matrix_3x3(eu)

    center = np.array([box['psr']['position']['x'], box['psr']['position']['y'], box['psr']['position']['z']])
    box_pts = np.matmul((pts - center), (trans_matrix))

    ground_level = 0.3
    if box['psr']['scale']['z'] < 2:
        ground_level = box['psr']['scale']['z'] * 0.15


    filter =  (box_pts[:, 0] < box['psr']['scale']['x']/2) & (box_pts[:, 0] > - box['psr']['scale']['x']/2) & \
             (box_pts[:, 1] < box['psr']['scale']['y']/2) & (box_pts[:, 1] > - box['psr']['scale']['y']/2)

    topfilter =    filter & (box_pts[:, 2] < box['psr']['scale']['z']/2) & (box_pts[:, 2] >= - box['psr']['scale']['z']/2 + ground_level)

    groundfilter = filter & (box_pts[:, 2] < -box['psr']['scale']['z']/2+ground_level) & (box_pts[:, 2] > - box['psr']['scale']['z']/2)
    
    
    return [pts[topfilter], pts[groundfilter], topfilter, groundfilter]

def remove_box(pts, box, ground_level=0.0, scaling = 1.0):
    eu = [box['psr']['rotation']['x'], box['psr']['rotation']['y'], box['psr']['rotation']['z']]
    trans_matrix = euler_angle_to_rotate_matrix_3x3(eu)

    center = np.array([box['psr']['position']['x'], box['psr']['position']['y'], box['psr']['position']['z']])
    box_pts = np.matmul((pts - center), (trans_matrix))
    filter_3d =  (box_pts[:, 0] < box['psr']['scale']['x']* scaling /2) & (box_pts[:, 0] > - box['psr']['scale']['x']* scaling /2) & \
             (box_pts[:, 1] < box['psr']['scale']['y']* scaling /2) & (box_pts[:, 1] > - box['psr']['scale']['y']* scaling /2) & \
             (box_pts[:, 2] < box['psr']['scale']['z'] * scaling /2) & (box_pts[:, 2] > - box['psr']['scale']['z']/2 + ground_level)
    return ~filter_3d

def color_obj_by_image(pts, box, image, extrinsic, intrinsic, ground_level=0):
    eu = [box['psr']['rotation']['x'], box['psr']['rotation']['y'], box['psr']['rotation']['z']]
    trans_matrix = euler_angle_to_rotate_matrix_3x3(eu)

    center = np.array([box['psr']['position']['x'], box['psr']['position']['y'], box['psr']['position']['z']])
    box_pts = np.matmul((pts - center), (trans_matrix))
    filter_3d =  (box_pts[:, 0] < box['psr']['scale']['x']/2) & (box_pts[:, 0] > - box['psr']['scale']['x']/2) & \
             (box_pts[:, 1] < box['psr']['scale']['y']/2) & (box_pts[:, 1] > - box['psr']['scale']['y']/2) & \
             (box_pts[:, 2] < box['psr']['scale']['z']/2) & (box_pts[:, 2] > - box['psr']['scale']['z']/2 + ground_level)
    
    target_pts = pts[filter_3d]

    imgpts, filter_in_frontview = proj_pts3d_to_img(target_pts, extrinsic, intrinsic)
    imgpts = imgpts.astype(np.int32)[:,0:2]

    height, width,_ = image.shape
    filter_inside_img = (imgpts[:,0] >= 0) & (imgpts[:,0] < width) & (imgpts[:,1] >= 0) & (imgpts[:,1] < height)
    filter_img = filter_in_frontview & filter_inside_img
    imgpts = imgpts[filter_img]
    pts_color = image[imgpts[:,1],imgpts[:,0],:]

    return box_pts[filter_3d][filter_img], pts_color, np.all(filter_img)

def gen_2dbox_for_obj_pts(box3d_pts, extrinsic, intrinsic, width, height):
    img_pts_top = proj_pts3d_to_img(box3d_pts[0], extrinsic, intrinsic, width, height)
    img_pts_ground = proj_pts3d_to_img(box3d_pts[1], extrinsic, intrinsic, width, height)

    if img_pts_top.shape[0]>3:
        p1 = np.min(img_pts_top, axis=0)
        p2 = np.max(img_pts_top, axis=0)

        if img_pts_ground.shape[0] > 1:
            q1 = np.min(img_pts_ground, axis=0)
            q2 = np.max(img_pts_ground, axis=0)

        return {
                "x1": p1[0],
                "y1": p1[1],
                "x2": p2[0],
                "y2": q2[1] if img_pts_ground.shape[0]>1 else p2[1] #p2[1]
            }
    return None


def gen_2dbox_for_obj_corners(box3d, extrinsic, intrinsic, width, height):

    corners = box3d_to_corners(box3d)
    corners_img = proj_pts3d_to_img(corners, extrinsic, intrinsic,width, height) 
    if corners_img.shape[0] == 0:
        print("rect points all out of image", o['obj_id'])
        return None
        
    corners_img = corners_img[:, 0:2]
    p1 = np.min(corners_img, axis=0)
    p2 = np.max(corners_img, axis=0)

    rect = {
            "x1": p1[0],
            "y1": p1[1],
            "x2": p2[0],
            "y2": p2[1]
        }

    return rect

def box_distance(box):
    p = box['psr']['position']
    return math.sqrt((p['x']*p['x'] + p['y']*p['y']))




class SuscapeScene:
    def __init__(self, data_root, name, cfg=None):
        self.name = name
        self.data_root = data_root
        
        # self.scene_path = os.path.join(data_root, name)

        self.cfg = cfg
        self.path = {}

        if cfg is None:
            self.path['camera'] = os.path.join(data_root, name, "camera")
            self.path['lidar'] = os.path.join(data_root, name, "lidar")
            self.path['aux_camera'] = os.path.join(data_root, name, "aux_camera")
            self.path['label'] = os.path.join(data_root, name, "label")
            self.path['calib'] = os.path.join(data_root, name, "calib")
            self.path['desc'] = os.path.join(data_root, name, "desc")
            self.path['meta'] = os.path.join(data_root, name, "meta")
            self.path['radar'] = os.path.join(data_root, name, "radar")
            self.path['aux_lidar'] = os.path.join(data_root, name, "aux_lidar")
            self.path['label_fusion'] = os.path.join(data_root, name, "label_fusion")
            self.path['lidar_pose'] = os.path.join(data_root, name, "lidar_pose")
            self.path['ego_pose'] = os.path.join(data_root, name, "ego_pose")

        else:
            self.path['camera'] = os.path.join(cfg['camera'], name, "camera")
            self.path['lidar'] = os.path.join(cfg['lidar'], name, "lidar")
            self.path['aux_camera'] = os.path.join(cfg['aux_camera'], name, "aux_camera")
            self.path['label'] = os.path.join(cfg['label'], name, "label")
            self.path['calib'] = os.path.join(cfg['calib'], name, "calib")
            self.path['desc'] = os.path.join(cfg['desc'], name, "desc")
            self.path['meta'] = os.path.join(cfg['meta'], name, "meta")
            self.path['radar'] = os.path.join(cfg['radar'], name, "radar")
            self.path['aux_lidar'] = os.path.join(cfg['aux_lidar'], name, "aux_lidar")
            self.path['label_fusion'] = os.path.join(cfg['label_fusion'], name, "label_fusion")
            self.path['lidar_pose'] = os.path.join(cfg['lidar_pose'], name, "lidar_pose")
            self.path['ego_pose'] = os.path.join(cfg['ego_pose'], name, "ego_pose")

        self.meta = self.read_scene_meta()


    def get_image_path(self, camera_type, camera, frame):
        ext = self.meta[camera_type][camera]['ext']
        img_file = os.path.join(self.path[camera_type], camera, frame + ext)
        return img_file

    def read_scene_meta(self):
        'read scene metadata give scene path'
        meta = {}

        frames = os.listdir(self.path['lidar'])
        frames = [*map(lambda f: os.path.splitext(f)[0], frames)]
        frames.sort()
        meta['frames'] = frames

        for camera_type in ['camera', 'aux_camera']:
            if os.path.exists(self.path[camera_type]):
                meta[camera_type] = {}
                for c in os.listdir(self.path[camera_type]):
                    files = os.listdir(os.path.join(self.path[camera_type], c))
                    any_file = files[0]

                    meta[camera_type][c] = {
                        'ext': os.path.splitext(any_file)[1]
                    }

                    for any_file in files:
                        if os.path.exists(os.path.join(self.path[camera_type], c, any_file)):
                            img = Image.open(os.path.join(self.path[camera_type], c, any_file))
                            meta[camera_type][c]['width'] = img.width
                            meta[camera_type][c]['height'] = img.height
                            break
                    
                    

        meta['calib'] = {}
        for camera_type in ['camera',  'aux_camera']:   
            if camera_type in meta:
                for camera in meta[camera_type]:
                    meta['calib'][camera_type] = {}
                    for camera in meta[camera_type]:
                        with open(os.path.join(self.path['calib'], camera_type, camera+".json")) as f:
                            meta['calib'][camera_type][camera] = json.load(f)
                        meta['calib'][camera_type][camera]['extrinsic'] = np.reshape(np.array(meta['calib'][camera_type][camera]['lidar_to_camera']), [4,4])
                        meta['calib'][camera_type][camera]['intrinsic'] = np.reshape(np.array(meta['calib'][camera_type][camera]['intrinsic']), [3,3])

        return meta


    def load_labels(self):
        label_folder = self.path['label']
        
        self.labels = {}
        for frame in self.meta['frames']:
            
            file = frame + '.json'
            
            if not os.path.exists(os.path.join(label_folder, file)):
                self.labels[frame] = []
                continue

            with open(os.path.join(label_folder, file)) as f:
                
                try:
                    labels = json.load(f)
                except:
                    print("error in reading file", file)
                    continue

                if "objs" in labels:
                    objs = labels['objs']
                else:
                    objs = labels
                
                self.labels[frame] = objs
    


    def get_boxes_by_frame(self, frame):
        if not self.labels:
            self.load_labels()
        
        return self.labels[frame]
    
    def _find_obj_by_id(self, boxes, id):
            for b in boxes:
                if b['obj_id']==id:
                    return b
            return None
    
    def get_boxes_of_obj(self, id):
        

        ret = {}
        for frame, boxes in self.labels.items():
            b = self._find_obj_by_id(boxes, id)
            if b:
                ret[frame] = b
        return ret
    
    def find_box_in_frame(self, frame, id):
        if frame not in self.labels:
            return None
        return self._find_obj_by_id(self.labels[frame], id)
    

    def get_calib_for_frame(self, camera_type, camera, frame):
        
        def combine_calib(static_calib, local_calib):
            trans = np.array(local_calib['lidar_transform']).reshape((4,4))
            extrinsic = np.matmul(static_calib['extrinsic'], trans)
            return (extrinsic,static_calib['intrinsic'])
        

        static_calib = self.meta['calib'][camera_type][camera]

        local_calib_file = os.path.join(self.path['calib'], camera_type, camera, frame+".json")
        if os.path.exists(local_calib_file):
            with open (local_calib_file) as f:
                local_calib = json.load(f)
                (extrinsic, intrinsic) = combine_calib(static_calib, local_calib)
                return (extrinsic, intrinsic)
        else:
            (extrinsic,intrinsic) =  (static_calib['extrinsic'], static_calib['intrinsic'])
            return (extrinsic, intrinsic)
        

    def list_objs(self):
        ret = {}

        if not self.labels:
            self.load_labels()

        for _, (frame, objs) in enumerate(self.labels.items()):
            for l in objs:
                #color = get_color(l["obj_id"])
                obj_id = l["obj_id"]
                
                if not obj_id in objs:
                    ret[obj_id] = l['obj_type']
        return [(i, ret[i]) for i in ret.keys()]


    def read_lidar(self, frame):
        # load lidar points
        
        lidar_file = os.path.join(self.path['lidar'], frame+".pcd")
        if not os.path.exists(lidar_file):
            return None
    
        pc = BinPcdReader(lidar_file)
            
        data = [pc.pc_data['x'], 
                        pc.pc_data['y'], 
                        pc.pc_data['z'],
                        pc.pc_data['intensity']]
        # if pc.pc_data['rgb']:
        #     data.append(pc.pc_data['rgb'] // 65536 / 256.0)
        #     data.append(pc.pc_data['rgb'] // 256 % 256 /256.0)
        #     data.append(pc.pc_data['rgb'] % 256 /256.0)
        # elif pc.pc_data['r']:
        #     data.append(pc.pc_data['r'])
        #     data.append(pc.pc_data['g'])
        #     data.append(pc.pc_data['b'])'

        # if has field 'rgb', append it to data
        if 'rgb' in pc.pc_data.dtype.names:
            data.append(pc.pc_data['rgb'] // 65536 / 256.0)
            data.append(pc.pc_data['rgb'] // 256 % 256 /256.0)
            data.append(pc.pc_data['rgb'] % 256 /256.0)

        pts =  np.stack(data, axis=-1)
        pts = pts[(pts[:,0]!=0) | (pts[:,1]!=0) | (pts[:,2]!=0)]
        return pts

    def read_ego_pose(self, frame):
        pose_file = os.path.join(self.path['ego_pose'], frame+'.json')
        with open(pose_file) as f:
            pose = json.load(f)
            return pose

    def load_ego_pose(self):
        ego_pose_folder = self.path["ego_pose"]
        self.ego_pose = {}
        for frame in self.meta['frames']:
            file = frame + '.json'
            with open(os.path.join(ego_pose_folder, file)) as f:
                self.ego_pose[frame] = json.load(f)
        
        return self.ego_pose
    
    def load_lidar_pose(self):
        lidar_pose_folder = self.path["lidar_pose"]
        self.lidar_pose = {}
        for frame in self.meta['frames']:
            file = frame + '.json'
            with open(os.path.join(lidar_pose_folder, file)) as f:
                self.lidar_pose[frame] = json.load(f)
        
        return self.lidar_pose
    
    def read_lidar_pose(self, frame):
        pose_file = os.path.join(self.path['lidar_pose'], frame+'.json')
        with open(pose_file) as f:
            pose = json.load(f)
            return pose
        
    
    def ego_pose_to_lidar_pose_rotate_matrix(self, p):
        eu = [float(p['roll']), float(p['pitch']), float(p['yaw'])]
        t = [float(p['x']), float(p['y']), float(p['z'])]

        return euler_angle_to_rotate_matrix(eu, t)



class SuscapeDataset:
    def __init__(self, root_dir, dir_org="") -> None:
        self.root_dir = root_dir
        cfg = self._build_dataset_cfgs(root_dir, dir_org)
        self.cfg = cfg
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
        self.ego_pose_dir = cfg['ego_pose']

        pass

    def _build_dataset_cfgs(self, root, dir_org):

        if dir_org == '':
            if os.path.exists(root + '/lidar'):
                dir_org = 'by_data_folder'
            else:
                dir_org = 'by_scene'

        dataset_cfg={}
        data_types = ['lidar',  'label', 'camera', 'calib', 'aux_lidar', 'aux_camera', 'radar', 'desc', 'meta','label_fusion', 'lidar_pose', 'ego_pose']
        if dir_org == 'by_scene':
            for d in data_types:
                dataset_cfg[d] = root
            dataset_cfg['root'] = root
            
        elif dir_org == 'by_data_folder':
            for d in data_types:
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

    def get_scene(self, scene_name):
        return SuscapeScene(self.root_dir, 
                              scene_name, 
                              self.cfg)
        

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
                # static part
                static_calib_file = os.path.join(calib_folder, sensor_type, sensor+".json")
                if os.path.exists(static_calib_file) and os.path.isfile(static_calib_file):
                    with open(static_calib_file, 'r') as f:
                        p = json.load(f)
                        this_t
                # dynamic part
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
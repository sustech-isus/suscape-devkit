#-*-coding:utf-8-*-
'''
transfer the dataset to kitti format
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import numpy as np
import cv2
from kitti_package.kitti_util import project_to_image,roty,compute_box_3d
from kitti_package.kitti_calibration import Calibration as Calibration_kitti
from utils.common_utils import get_whole_path,get_name_list,loadjson
from utils.parse_pcd import PointCloud

PC_AREA_SCOPE = [[-40, 40], [-1,   3], [0, 70.4]]

def check_pc_range(xyz):
    """
    check the point cloud range
    :param xyz: [x, y, z]
    :return: bool
    """
    x_range, y_range, z_range = PC_AREA_SCOPE
    if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
            (z_range[0] <= xyz[2] <= z_range[1]):
        return True
    return False

def convert_calib(sustech_root,kitti_root,rotate_matrix=None):
    '''
    convert the calibration to kitti format
    :param sustech_root: the root dir of the original dataset
    :param kitti_root: the kitti dir of the kitti
    :param rotate_matrix: rotate matrix of the dataset
    :return: save_path save_path of the calibration file
    '''
    calib_path = get_whole_path(sustech_root, 'calib', 'camera')
    calib_camera_front_path = get_whole_path(calib_path, 'front.json')
    assert os.path.exists(calib_camera_front_path), 'THE calib_camera_front_path DOES NOT EXIST.PLEASE CHECK!'
    c = loadjson(calib_camera_front_path)
    P2 = np.asarray(c['intrinsic']).reshape(3, 3)
    P2 = np.concatenate([P2, np.zeros((3, 1))], axis=1).reshape(-1).tolist()
    V2C = np.asarray(c['extrinsic']).reshape(4, 4)
    V2C = V2C[:3, :]
    if rotate_matrix is not None:
        rotate_matrix = np.concatenate([rotate_matrix,np.zeros((3,1))],axis=1)
        rotate_matrix = np.concatenate([rotate_matrix,np.zeros((1,4))],axis=0)
        rotate_matrix[3,3] = 1
        V2C = V2C @ np.linalg.inv(rotate_matrix.T)
    V2C = V2C.reshape(-1).tolist()
    R0 = np.identity(3).reshape(-1).tolist()
    list2str = lambda x: ' '.join(map(str, x))
    kitti_format = [
        'P0: ' + list2str(np.zeros_like(P2).tolist()),
        'P1: ' + list2str(np.zeros_like(P2).tolist()),
        'P2: ' + list2str(P2),
        'P3: ' + list2str(np.zeros_like(P2).tolist()),
        'R0_rect: ' + list2str(R0),
        'Tr_velo_to_cam: ' + list2str(V2C),
        'Tr_imu_to_velo: ' + list2str(np.zeros_like(V2C).tolist())
    ]
    save_path = get_whole_path(kitti_root, 'kitti_format.txt')
    with open(save_path, 'w') as f:
        f.write('\n'.join(kitti_format))
    return save_path

def check_box_in_image(l,w,h,ry,pos, P,xmin,ymin,xmax,ymax):
    '''
    check the box in the image or not
    :param l: box scale
    :param w: box scale
    :param h: box scale
    :param ry: rotate along y-axis
    :param pos: center position of the box
    :param P: calibration information
    :param xmin: pixel range
    :param ymin: pixel range
    :param xmax: pixel range
    :param ymax: pixel range
    :return: bool
    '''
    # compute rotational matrix around yaw axis
    R = roty(ry)
    # 3d bounding box dimensions
    l = l
    w = w
    h = h
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + pos[0]
    corners_3d[1, :] = corners_3d[1, :] + pos[1]
    corners_3d[2, :] = corners_3d[2, :] + pos[2]
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return False
    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    fov_inds = (
            (corners_2d[:, 0] < xmax)
            & (corners_2d[:, 0] >= xmin)
            & (corners_2d[:, 1] < ymax)
            & (corners_2d[:, 1] >= ymin)
    )
    points_in_img_mask = fov_inds & True
    if all(points_in_img_mask):
        return True
    else:
        return False

def rotate_along_z_matrix(rot_angle):
    '''
    rotation of the z axis
    :param rot_angle: rotation angle
    :return:
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([
        [cosval, -sinval, 0.0],
        [sinval, cosval, 0.0],
        [0.0,0.0,1.0]
    ])
    return rotmat.T

global_kitti_frame_id = 0
def allocate_kitti_frame_id():
    global global_kitti_frame_id 
    id = "{:06d}".format(global_kitti_frame_id)
    global_kitti_frame_id += 1
    return id


def SUSTS2KITTI(scene_root, kitti_root):
    '''
    main function of the transfer tool
    :param scene_root: the dir of the scene
    :param kitti_root: the save dir of the transferred kitti dataset
    :return: None
    '''
    sus_label_folder_path = get_whole_path(scene_root, 'label')
    sus_img_path = get_whole_path(scene_root, 'camera', 'front')
    sus_lidar_bin_path = get_whole_path(scene_root,'lidar_bin')
    if not os.path.exists(sus_lidar_bin_path):
        os.makedirs(sus_lidar_bin_path,exist_ok=True)
    
    if not "trans_pcd_file":
        pcd_path = get_whole_path(scene_root, 'lidar')
        for pcd_name in get_name_list(pcd_path):
            pcd_file_path = get_whole_path(pcd_path, pcd_name)
            assert os.path.exists(pcd_file_path), 'THE pcd_file_path DOES NOT EXIST.PLEASE CHECK!'
            pts = PointCloud.from_path(pcd_file_path).pcd_to_numpy(intensity = True)
            bin_file_path = get_whole_path(sus_lidar_bin_path,pcd_name[:-3] + 'bin')
            pts = pts.reshape(-1).astype(np.float32)
            pts.tofile(bin_file_path)

    angle = np.pi/2
    rotate_matrix = rotate_along_z_matrix(angle)
    inverse = rotate_along_z_matrix(-angle)
    calib_path = convert_calib(scene_root,kitti_root,rotate_matrix=rotate_matrix)
    calib = Calibration_kitti(calib_path)
    for img_name in get_name_list(sus_img_path):

        if True: #args.rename_frames_after_kitti:
            kitti_frame_id = allocate_kitti_frame_id()
        else:
            kitti_frame_id = img_name[:-4]



        sus_label_name = img_name[:-3] + 'json'
        sus_lidar_name = img_name[:-3] + 'bin'


        kitti_label_name = kitti_frame_id + '.txt'

        sus_label_path = get_whole_path(sus_label_folder_path,sus_label_name)
        # print(sus_label_path)
        susts = loadjson(sus_label_path)
        assert susts != [], "The label file is empty,Please Check!!"
        if type(susts) is list:
            susts = susts
        else:
            susts = susts['objs']
        kitti = []
        for obj in susts:
            psr = obj['psr']
            pos = np.asarray([psr['position']['x'], psr['position']['y'], psr['position']['z']])[np.newaxis, :]
            pos = pos @ rotate_matrix
            l, w, h = psr['scale']['x'], psr['scale']['y'], psr['scale']['z']
            pos[0,2] -= h/2
            pos = calib.project_velo_to_rect(pos).reshape(-1)
            ry = -psr['rotation']['z'] - np.pi / 2 - angle
            #if not check_pc_range(pos):
            #    continue
            if not check_box_in_image(l, w, h, ry, pos, calib.P, 0, 0, 2048, 1536):
                continue
            if "BicycleRider" in obj['obj_type']:
                cls_type = "Cyclist"
            elif "Child" in obj['obj_type'] or "RoadWorker" in obj['obj_type']:
                cls_type = "Pedestrian"
            else:
                cls_type = obj['obj_type']
            trucation = -1.0
            occlusion = -1.0
            beta = np.arctan2(pos[2], pos[0])
            alpha = ry + beta - np.sign(beta) * np.pi / 2
            # box2d = [-1] * 4
            corners_2d,_ = compute_box_3d(l,w,h,pos,ry, calib.P)
            box2d = calib.project_8p_to_4p(corners_2d)
            kitti_format = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                           % (cls_type, trucation, int(occlusion), alpha, box2d[0], box2d[1],
                              box2d[2], box2d[3], h, w, l, pos[0], pos[1], pos[2],
                              ry)
            kitti.append(kitti_format)
        if kitti != []:

            

            kitti_label_path = get_whole_path(kitti_root,'label_2')
            kitti_calib_path = get_whole_path(kitti_root,'calib')
            kitti_img_path = get_whole_path(kitti_root,'image_2')
            kitti_lidar_path = get_whole_path(kitti_root,'velodyne')

            os.makedirs(kitti_label_path, exist_ok=True)
            os.makedirs(kitti_calib_path, exist_ok=True)
            os.makedirs(kitti_img_path, exist_ok=True)
            os.makedirs(kitti_lidar_path, exist_ok=True)
            
            # lable
            with open(get_whole_path(kitti_label_path,kitti_label_name),'w') as f:
                f.write('\n'.join(kitti))

            # points
            points = np.fromfile(get_whole_path(sus_lidar_bin_path,sus_lidar_name),dtype=np.float32).reshape((-1,4))
            points[:,:3] = points[:,:3] @ rotate_matrix
            points.tofile(get_whole_path(kitti_lidar_path, kitti_frame_id + ".bin"))

            # calib
            os.system('cp %s %s' % (calib_path, get_whole_path(kitti_calib_path, kitti_frame_id + ".txt")))

            # image
            img_path = get_whole_path(sus_img_path, img_name)
            assert os.path.exists(img_path), 'THE img_path DOES NOT EXIST.PLEASE CHECK!'
            img = cv2.imread(img_path)
            cv2.imwrite(get_whole_path(kitti_img_path, kitti_frame_id+'.png'),img)
        else:
            # print("the path is: ", get_whole_path(sus_label_path, sus_label_name))
            continue



if __name__ == '__main__':
    # for susts_img_fn in tqdm(os.listdir(susts_img_dir)):
    susts_root_dir = '/home/lie/nas2/proj01_3d2d_dataset_submit1121/scene0'
    kitti_root = '/home/lie/fast/proj01_kitti_0'
    scene_list = get_name_list(susts_root_dir)
    # for scene in tqdm(scene_list):
    for scene in scene_list:
        scene_dir = os.path.join(susts_root_dir,scene)
        save_kitti = os.path.join(kitti_root,scene,"training")
        if not os.path.exists(save_kitti):
            os.makedirs(save_kitti)
        SUSTS2KITTI(scene_dir, save_kitti)

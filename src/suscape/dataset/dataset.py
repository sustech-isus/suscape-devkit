#-*-coding:utf-8-*-

'''
the devkit of the dataset,which is a dataset for the Autonomous Driving(AD)
author:Tim Liu
'''

import cv2
import numpy as np
import os
import mayavi.mlab as mlab
from ..utils.common_utils import get_name_list,get_whole_path,loadjson,str_all_index
from ..utils.obj_color import classname_to_color,hex_to_rgb
from ..data.box import Box
from ..data.lidar_pointcloud import LidarPointCloud
from ..data.radar import RadarPointCloud
from ..data.ego_pose import egoPose
from ..utils.calibration import Calibration


class Dataset(object):
    '''
    base class for dataset devkit to use
    '''
    def __init__(self,data_root):
        '''
        init the dataset class, all the member variable is the path of the sensors
        :param data_root: str the dataset root path
        '''
        self.data_root = data_root
        self.scene_list = get_name_list(self.data_root)
        self.lidar_path_dic = self.get_folder_path("lidar")
        self.label_path_dic = self.get_folder_path("label")
        self.label_fusion_path_dic = self.get_folder_path("label_fusion")
        self.camera_path_dic = self.get_folder_path("camera")
        self.auxcamera_path_dic = self.get_folder_path("aux_camera")
        self.radar_path_dic = self.get_folder_path("radar")
        self.calib_path_dic = self.get_folder_path("calib")
        self.ego_pose_path_dic = self.get_folder_path("ego_pose")

    def get_folder_path(self,folder_name = "lidar"):
        '''
        get the whole path of sensors of the datasets
        :param folder_name: str the sensor name of the ego
        :return: dic{"scene_name1":sensor_path,"scene_name2":sensor_path}
        '''
        folder_dic = {}
        scene_list = self.scene_list
        data_root = self.data_root
        for scene in scene_list:
            folder_dic[scene] = get_whole_path(data_root, scene, folder_name)
        return folder_dic

    def get_boxes(self,anno_info):
        '''
        get the box and init the Box class, the box in the Lidar coordinate system
        :param anno_info: str of list, the path of the annotation file or the list of the annotation information
        :return: boxes: list:[Box1,Box2,...]
        '''
        boxes = []
        box_info = loadjson(anno_info)
        if type(box_info) is list:
            box_list = box_info
        else:
            box_list = box_info['objs']
        for box in box_list:
            center = [box["psr"]["position"]["x"],box["psr"]["position"]["y"],box["psr"]["position"]["z"]]
            scale = [box["psr"]["scale"]["x"],box["psr"]["scale"]["y"],box["psr"]["scale"]["z"]]
            rotation = np.array([box["psr"]["rotation"]["x"],box["psr"]["rotation"]["y"],box["psr"]["rotation"]["z"]])
            name = box["obj_type"]
            if 'obj_id' not in box.keys():
                obj_id = 0
            else:
                obj_id = int(box["obj_id"])
            if 'score' not in box.keys():
                score = 0.00
            else:
                score = box['score']
            boxes.append(Box(center=center,size=scale,
                             rotation = rotation,
                             score=score,
                             obj_id = obj_id,
                             name=name))
        return boxes

    def draw_box_in_image(self,scene_name,file_prefixname,sensor_name):
        '''
        draw the label box in image
        :param scene_name: str the scene name in the dataset
        :param file_prefixname: str the prefix name of the sensor file
        :param sensor_name: str the name of the sensor
        :return: img np.ndarray return the image with boxes
        '''
        from ..utils.draw_gt import draw_box_in_img
        if 'aux' in sensor_name:
            sub_sensor = sensor_name[str_all_index(sensor_name, "_")[1] + 1:]
            img_path = get_whole_path(self.auxcamera_path_dic[scene_name],sub_sensor,file_prefixname + ".jpg")
            assert os.path.exists(img_path), "".join(("Error: ",img_path," of aux-camera Does not exists"))
        else:
            sub_sensor = sensor_name[str_all_index(sensor_name, "_")[0] + 1:]
            img_path = get_whole_path(self.camera_path_dic[scene_name], sub_sensor, file_prefixname + ".jpg")
            assert os.path.exists(img_path), "".join(("Error: ",img_path," of camera Does not exists"))
        img = cv2.imread(img_path)  # 2048 * 1536
        img_shape = img.shape[0:2]  # img_height,img_width
        # need to transfer the box(in the lidar_top coordinate) to the camera
        sensor_calib_path = get_whole_path(self.calib_path_dic[scene_name])
        calib = Calibration(sensor_calib_path)
        anno_path = get_whole_path(self.label_path_dic[scene_name], file_prefixname + ".json")
        boxes = self.get_boxes(anno_path)
        assert len(boxes) > 0, 'Error: The 3D box list is NONE,PLEASE CHECK!'
        for box in boxes:
            box_corners = box.corners[0:3, :]
            corners_in_img, mask = calib.box_in_image(box_corners, sensor_name, img_shape,file_prefixname)
            if mask is True:
                color = hex_to_rgb(classname_to_color[box.name])
                img = draw_box_in_img(img, corners_in_img.T, color=(color[2],color[1],color[0]), thickness=2)
        return img

    def load_lidar_pcd(self,lidar_path,file_prefixname):
        '''
        load the lidar pcd file
        :param lidar_path: the path of lidar sensor
        :param file_prefixname: the prefix name of the file
        :return: points: np.array 3*n
        '''
        lidar_pcd_path = get_whole_path(lidar_path, file_prefixname + ".pcd")
        assert os.path.exists(lidar_pcd_path), "".join(("Error: ",lidar_pcd_path," of lidar Does not exists"))
        pointcloud = LidarPointCloud.load_pcdfile(lidar_pcd_path)
        return pointcloud.points[0:3]

    def load_radar_pcd(self,scene_name,sensor_name,file_prefixname,type = 'tracks'):
        '''
        load the radar pcd file
        :param scene_name: the name of the scene
        :param sensor_name: the name of the sensor
        :param file_prefixname: the prefix name of the file
        :param type: the type of the radar points
        :return: points: np.array 3*n
        '''

        if type == 'tracks':
            radar_path = get_whole_path(self.radar_path_dic[scene_name],''.join(('tracks_',sensor_name)))
            radar_pcd_path = get_whole_path(radar_path, file_prefixname + ".pcd")
        else:
            radar_path = get_whole_path(self.radar_path_dic[scene_name],''.join(('points_',sensor_name)))
            radar_pcd_path = get_whole_path(radar_path, file_prefixname + ".pcd")
        assert os.path.exists(radar_pcd_path), "".join(("Error: ",radar_pcd_path," of radar Does not exists"))
        point_cloud = RadarPointCloud.load_pcdfile(radar_pcd_path)
        return point_cloud.points[0:3]

    def render_lidar_withbox_plt(self,axes,scene_name,file_prefixname,axes_limit= 100):
        '''
        draw the label box in lidar coord
        :param axes: Axis onto which the box should be drawn
        :param scene_name: the name of the scene
        :param file_prefixname: the prefix name of the file
        :param boxes: the boxes in the lidar scene
        :param axes_limit: float the limit of the axes
        :return:
        '''
        from ..utils.draw_gt import draw_lidar_scenes_plt, draw_anno_plt
        points = self.load_lidar_pcd(self.lidar_path_dic[scene_name],file_prefixname)
        draw_lidar_scenes_plt(axis=axes, points=points[:3, :])
        # render the annotation on the scenes
        anno_path = get_whole_path(self.label_path_dic[scene_name], file_prefixname + ".json")
        assert os.path.exists(anno_path), 'THE anno_path of lidar DOES NOT EXIT.PLEASE CHECK!'
        boxes = self.get_boxes(anno_path)
        assert len(boxes) > 0, 'Error: The 3D box list is NONE,PLEASE CHECK!'
        for box in boxes:
            color = np.array(hex_to_rgb(classname_to_color[box.name]))/255.0
            draw_anno_plt(axis=axes, box=box, colors=(color,color,color))
        axes.set_xlim(-axes_limit, axes_limit)
        axes.set_ylim(-axes_limit, axes_limit)


    def render_radar_in_lidar(self,scene_name,sensor_name,file_prefixname,calib,fig,points_type = 'tracks'):
        '''
        draw points of radar in lidar
        :param scene_name: the name of the scene
        :param sensor_name: the name of the sensor
        :param file_prefixname: prefix name of the file
        :param calib: calibration information
        :param fig: the figure will rendered
        :param points_type: the type of the radar points
        :return:
        '''
        from ..utils.draw_gt import draw_radar_points
        radar_points = self.load_radar_pcd(scene_name,sensor_name,file_prefixname,points_type)
        radar_in_lidar = calib.calib_radar_to_lidar(radar_points,sensor_name,points_type)
        draw_radar_points(radar_in_lidar[0:3,:].T,fig)

    def render_lidar_withbox_mayavi(self,fig,scene_name,file_prefixname,with_radar = False):
        '''
        draw the label box in lidar coord using mayavi
        :param fig: mlab.figure
        :param points: np.array points cloud
        :param boxes:Box the label in lidar coord
        :return:
        '''
        from ..utils.draw_gt import draw_lidar_scenes_mayavi,draw_gt_boxes3d
        points = self.load_lidar_pcd(self.lidar_path_dic[scene_name],file_prefixname)
        draw_lidar_scenes_mayavi(points.T,fig=fig,pts_color=(1,1,1))
        if with_radar:
            calib = Calibration(get_whole_path(self.calib_path_dic[scene_name]))
            self.render_radar_in_lidar(scene_name,'front',file_prefixname,calib,fig,points_type='points')
            self.render_radar_in_lidar(scene_name, 'front_left', file_prefixname, calib, fig, points_type='points')
            self.render_radar_in_lidar(scene_name, 'front_right', file_prefixname, calib, fig, points_type='points')
            self.render_radar_in_lidar(scene_name, 'rear', file_prefixname, calib, fig, points_type='points')
            self.render_radar_in_lidar(scene_name, 'rear_left', file_prefixname, calib, fig, points_type='points')
            self.render_radar_in_lidar(scene_name, 'rear_right', file_prefixname, calib, fig, points_type='points')
        anno_path = get_whole_path(self.label_path_dic[scene_name], file_prefixname + ".json")
        assert os.path.exists(anno_path), "".join(("Error: ",anno_path," Does not exists"))
        boxes = self.get_boxes(anno_path)
        assert len(boxes) > 0, 'Error: The 3D box list is NONE,PLEASE CHECK!'
        for box in boxes:
            color = np.array(hex_to_rgb(classname_to_color[box.name]))/255.0
            draw_gt_boxes3d(box.corners.T[:,0:3],fig=fig,color = (color[0],color[1],color[2]))


    def render_lidar(self,scene_name,file_prefixname,render_type = "plt",save = False,with_radar = False):
        '''
        render lidar scene and gt box use two types
        :param scene_name: str,the name of the scene
        :param file_prefixname: str,the name of the sensor file
        :param render_type: str,the type of the rendering lidar,plt and mayavi
        :param save: str,save the result or not
        :param with_radar: Optional.If show the point cloud with radar or not
        :return:
        '''
        if render_type == 'plt':
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 1, figsize=(18, 18))
            self.render_lidar_withbox_plt(axes, scene_name,file_prefixname)
            if save is not None:
                plt.axis('off')
                plt.savefig('./test.jpg', bbox_inches='tight', pad_inches=0, dpi=200)
        else:
            fig = mlab.figure(
                figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
            )
            self.render_lidar_withbox_mayavi(fig, scene_name,file_prefixname,with_radar)
            mlab.show()

    def read_ego_pose(self,ego_pose_path):
        '''
        read the ego pose according to the path
        @param ego_pose_path: the path of the ego pose
        @return: class egoPose
        '''
        ego_pose = egoPose(ego_pose_path)
        return ego_pose

    def render(self,scene_name,file_prefixname,sensor_name='lidar',render_type= "plt",save= False,with_radar = False):
        '''
        draw the annotation box on the image or the pointcloud,
        :param scene_name: str,the scene name in the dataset
        :param file_prefixname: str,the prefix name of the sensor file
        :param sensor_name: str, the name of the sensor
        :param render_type: str, the type of render the point cloud
        :param outputpath: Optional. If you want to save the result,which is rendered
        :param with_aux: Optional.If show the point cloud with aux-lidar or not
        :param with_radar: Optional.If show the point cloud with radar or not
        :return:
        '''
        if "lidar" in sensor_name:
            self.render_lidar(scene_name,file_prefixname,render_type,save,with_radar)
        elif "camera" in sensor_name:
            img = self.draw_box_in_image(scene_name,file_prefixname,sensor_name)
            if save :
                cv2.imwrite("".join(('./',scene_name,'_',sensor_name,'_',file_prefixname,'.jpg')),img)

if __name__ == "__main__":
    dataset = Dataset("./example/dataset_mini")
    dataset.render(scene_name="scene-000000",
                   file_prefixname="1656932600.000",
                   render_type="",
                   # sensor_name="camera_front",
                   sensor_name="lidar",
                   save=True,
                   with_radar=True
                   )
    ego_pose = dataset.read_ego_pose('./example/dataset_mini/scene-000000/ego_pose/1656932600.000.json')
    print(ego_pose.x)
















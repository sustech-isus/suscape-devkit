#-*-coding:utf-8-*-

'''
the point cloud  class of dataset
author:Tim Liu
'''

from ..utils.common_utils import loadjson

class egoPose(object):
    '''
    the egoPose class
    '''
    def __init__(self,egopose_path):
        '''
        init the egoPose class
        :param egopose_path: the path of ego pose ,which is a json file
        '''
        self.egopose_dic = loadjson(egopose_path)
        self.lat = float(self.egopose_dic['lat'])
        self.lng = float(self.egopose_dic['lng'])
        self.height = float(self.egopose_dic['height'])
        self.north_vel = float(self.egopose_dic['north_vel'])
        self.east_vel = float(self.egopose_dic['east_vel'])
        self.up_vel = float(self.egopose_dic['up_vel'])
        self.roll = float(self.egopose_dic['roll'])
        self.pitch = float(self.egopose_dic['pitch'])
        self.azimuth = float(self.egopose_dic['azimuth'])
        self.x = float(self.egopose_dic['x'])
        self.y = float(self.egopose_dic['y'])
        self.z = float(self.egopose_dic['z'])



if __name__ == "__main__":
    egopose_path = "./example/scene-000000/dataset_mini/ego_pose/1656932600.000.json"
    ego_pose = egoPose(egopose_path)
    a = 1



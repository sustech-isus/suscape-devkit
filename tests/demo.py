from suscape.dataset import SuscapeDataset, SuscapeScene, box3d_to_corners


dataset = SuscapeDataset('../suscape-test')

print(dataset.get_scene_names())

scene = dataset.get_scene("scene-000040")

print(scene.meta['frames'])
print(scene.meta['calib']['camera']['front']['intrinsic'])
print(scene.meta['calib']['camera']['front']['lidar_to_camera'])


scene.load_labels()
print(scene.labels[scene.meta['frames'][0]])

boxes = scene.get_boxes_by_frame(scene.meta['frames'][0])
print(boxes)

print(scene.get_boxes_of_obj(id="1"))

print(scene.find_box_in_frame(frame=scene.meta['frames'][0], id="1"))


calib = scene.get_calib_for_frame("camera", "front", scene.meta['frames'][0])
lidar2cam, intrinsic = calib[0], calib[1]
print("lidar2cam:", lidar2cam)
print("intrinsic:", intrinsic)

print(scene.list_objs())


print(scene.read_lidar(scene.meta['frames'][0]))

print(scene.read_lidar_pose(scene.meta['frames'][0]))
scene.load_lidar_pose()
print(scene.lidar_pose[scene.meta['frames'][1]])


print(box3d_to_corners(boxes[1]))



# read one front image, show it
# pip install opencv-python matplotlib
import matplotlib.pyplot as plt
import cv2

imgpath = scene.get_image_path("camera", "front", scene.meta['frames'][0])
img = cv2.imread(imgpath)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()



# show 3d lidar pts
# pip install open3d
import open3d as o3d
pts = scene.read_lidar(scene.meta['frames'][0])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts[:,:3])
o3d.visualization.draw_geometries([pcd])


# project 3d points onto image
import numpy as np
frame = scene.meta['frames'][0]
pts = scene.read_lidar(frame)
image = cv2.imread(scene.get_image_path("camera", "front", frame))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

calib = scene.get_calib_for_frame("camera", "front", frame)
lidar2cam, intrinsic = calib[0], calib[1]
# filter points in front of camera
pts_hom = np.hstack((pts[:,:3], np.ones((pts.shape[0],1))))
pts_cam = (lidar2cam @ pts_hom.T).T
pts_cam = pts_cam[pts_cam[:,2]>0]
# project
pts_2d = (intrinsic @ pts_cam[:,:3].T).T
pts_2d[:,0] /= pts_2d[:,2]
pts_2d[:,1] /= pts_2d[:,2]  

# filter those out of image
h, w, _ = image.shape
pts_2d = pts_2d[(pts_2d[:,0]>=0) & (pts_2d[:,0]<w) & (pts_2d[:,1]>=0) & (pts_2d[:,1]<h)]

for p in pts_2d:
    cv2.circle(image, (int(p[0]), int(p[1])), 1, (0,255,0), -1)
plt.imshow(image)
plt.show()
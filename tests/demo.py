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
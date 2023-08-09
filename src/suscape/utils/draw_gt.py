#-*-coding:utf-8-*-

'''
render the 3D box render on the scenes
author:Tim Liu

'''



import cv2
import numpy as np
from matplotlib.axes import Axes
from ..data.box import Box
import mayavi.mlab as mlab

def draw_rect_plt(axis,selected_corners, color,linewidth):
    '''
    draw rectangle of the 3D box
    :param axis: Axis onto which the box should be drawn
    :param selected_corners: the corners,of which is selected
    :param color: the color of the lines
    :param linewidth: Width in pixel of the box sides.
    :return:
    '''
    prev = selected_corners[-1]
    for corner in selected_corners:
        axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
        prev = corner

def draw_lidar_scenes_plt(axis,points,axes_limit= 100):
    '''
    render the whole point cloud scenes using the plt
    :param axis: Axis onto which the box should be drawn
    :param points: the points of the whole scenes
    :param axes_limit: the limit of the axes
    :return: scatter: scatter the whole scenes
    '''
    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
    point_scale = 0.2
    scatter = axis.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
    # return scatter

def draw_anno_plt(axis: Axes,box:Box,colors = ('b', 'r', 'k'),linewidth = 2):
    '''
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param box: the Box class,which is defined in the data package
    :param colors: Valid Matplotlib colors (<str> or normalized RGB tuple) for front,back and sides.
    :param linewidth: Width in pixel of the box sides.
    :return:
    '''
    corners = box.corners[:2, :]
    # Draw the sides
    for i in range(4):
        axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                  [corners.T[i][1], corners.T[i + 4][1]],
                  color=colors[2], linewidth=linewidth)
    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect_plt(axis, corners.T[:4], colors[0], linewidth)
    draw_rect_plt(axis, corners.T[4:], colors[1], linewidth)
    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    axis.plot([center_bottom[0], center_bottom_forward[0]],
              [center_bottom[1], center_bottom_forward[1]],
              color=colors[0], linewidth=linewidth)

def draw_box_in_img(image,qs,color=(0,255,0),thickness=2):
    '''
    Draw 3d bounding box in image
    qs: (8,3) array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    :param image: the image should be drawn
    :param qs: the corners
    :param color:
    :param thickness:
    :return:
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def draw_radar_points(points,fig = None):
    '''
    draw radar point in the
    :param points the points in radar
    :return:
    '''
    for point in points:
        mlab.points3d(point[0], point[1], point[2], color=(1, 0, 0), mode="sphere", scale_factor=1.2,figure=fig)
    return fig

def draw_all_points_mayavi(points,fig = None,pts_scale=0.3,pts_color=None):
    '''
    draw the lidar scenes
    :param points:the points
    :param fig:
    :param pts_scale:
    :return:
    '''

    pts_mode = "point"
    color = points[:, 2]
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color,
                  color=pts_color,
                  mode=pts_mode,
                  colormap="gnuplot",
                  scale_factor=pts_scale,
                  figure=fig)
    # return fig

def draw_lidar_scenes_mayavi(points,fig = None,pts_color=None):
    '''
    draw the lidar scenes use mayavi
    :param pc: the pc data of the scene
    :param color: numpy array (n) of intensity or whatever
    :return: fig: created or used fig
    '''
    fig = draw_all_points_mayavi(points,fig=fig,pts_color=pts_color)
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2,figure=fig)
    # draw axis
    axes = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]],[0, axes[0, 1]],[0, axes[0, 2]],
                color=(1, 0, 0),
                tube_radius=None,
                figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0),
                tube_radius=None,
                figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1),
                tube_radius=None,
                figure=fig)

def draw_gt_boxes3d(box,fig,color=(1, 1, 1),line_width=1):
    '''
    draw the 3D bounding box
    :param box: the corners of the box,numpy array(8,3) for x y z
    :param fig: mayavi figure handler
    :param color: RGB value tuple in range (0,1), box line color
    :param line_width: box line width
    :return: fig: updated fig
    '''
    for k in range(0, 4):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        mlab.plot3d([box[i, 0], box[j, 0]],[box[i, 1], box[j, 1]],[box[i, 2], box[j, 2]],
                    color=color,
                    tube_radius=None,
                    line_width=line_width,
                    figure=fig)
        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([box[i, 0], box[j, 0]],[box[i, 1], box[j, 1]],[box[i, 2], box[j, 2]],
                    color=color,
                    tube_radius=None,
                    line_width=line_width,
                    figure=fig)
        i, j = k, k + 4
        mlab.plot3d([box[i, 0], box[j, 0]],[box[i, 1], box[j, 1]],[box[i, 2], box[j, 2]],
                    color=color,
                    tube_radius=None,
                    line_width=line_width,
                    figure=fig)






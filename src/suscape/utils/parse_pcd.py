#-*-coding:utf-8-*-
"""
parse the pcd file for python3
Ref: https://github.com/dimatura/pypcd
"""

import re
import copy
import numpy as np
import warnings

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#'.encode()) or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln.decode())
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = [int(i) for i in value.split()]
            # metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata

def _metadata_is_consistent(metadata):
    """ Sanity check for metadata. Just some basic checks.
    """
    checks = []
    required = ('version', 'fields', 'size', 'width', 'height', 'points',
                'viewpoint', 'data')
    for f in required:
        if f not in metadata:
            print('%s required' % f)
    checks.append((lambda m: all([k in m for k in required]),
                   'missing field'))
    # print("te len of the list(m['count']) is: ",list(m['count']))
    checks.append((lambda m: len(m['type']) == len(list(m['count'])) ==
                   len(list(m['fields'])),
                   'length of type, count and fields must be equal'))
    checks.append((lambda m: m['height'] > 0,
                   'height must be greater than 0'))
    checks.append((lambda m: m['width'] > 0,
                   'width must be greater than 0'))
    checks.append((lambda m: m['points'] > 0,
                   'points must be greater than 0'))
    checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                   'binary_compressed'),
                   'unknown data type:'
                   'should be ascii/binary/binary_compressed'))
    ok = True
    for check, msg in checks:
        if not check(metadata):
            print('error:', msg)
            ok = False
    return ok

def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.

    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.

    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    other = 1
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%s_%04d' % (f,str(other), i) for i in range(c)])
            typenames.extend([np_type]*c)
            other = other + 1
    a = list(zip(fieldnames, typenames))
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_binary_pc_data(f, dtype, metadata):
    # print("the dtype.itemsize is: ",dtype.itemsize)
    # print("the dtype['x'] is: ",dtype['x'])
    rowstep = metadata['points']*dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)

def point_cloud_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip()
        header.append(ln)
        if ln.startswith('DATA'.encode()):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break

    pc_data = parse_binary_pc_data(f, dtype, metadata)

    return PointCloud(metadata, pc_data)


def point_cloud_from_path(fname):
    """ load point cloud in binary format
    """
    with open(fname, 'rb') as f:
        pc = point_cloud_from_fileobj(f)
    return pc


class PointCloud(object):
    """ Wrapper for point cloud data.

    The variable members of this class parallel the ones used by
    the PCD metadata (and similar to PCL and ROS PointCloud2 messages),

    ``pc_data`` holds the actual data as a structured numpy array.

    The other relevant metadata variables are:

    - ``version``: Version, usually .7
    - ``fields``: Field names, e.g. ``['x', 'y' 'z']``.
    - ``size.`: Field sizes in bytes, e.g. ``[4, 4, 4]``.
    - ``count``: Counts per field e.g. ``[1, 1, 1]``. NB: Multi-count field
      support is sketchy.
    - ``width``: Number of points, for unstructured point clouds (assumed by
      most operations).
    - ``height``: 1 for unstructured point clouds (again, what we assume most
      of the time.
    - ``viewpoint``: A pose for the viewpoint of the cloud, as
      x y z qw qx qy qz, e.g. ``[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]``.
    - ``points``: Number of points.
    - ``type``: Data type of each field, e.g. ``[F, F, F]``.
    - ``data``: Data storage format. One of ``ascii``, ``binary`` or ``binary_compressed``.

    See `PCL docs <http://pointclouds.org/documentation/tutorials/pcd_file_format.php>`__
    for more information.
    """

    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """ returns copy of metadata """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def check_sanity(self):
        # pdb.set_trace()
        md = self.get_metadata()
        assert(_metadata_is_consistent(md))
        assert(len(self.pc_data) == self.points)
        assert(self.width*self.height == self.points)
        assert(len(self.fields) == len(self.count))
        assert(len(self.fields) == len(self.type))


    def copy(self):
        new_pc_data = np.copy(self.pc_data)
        new_metadata = self.get_metadata()
        return PointCloud(new_metadata, new_pc_data)

    @staticmethod
    def from_path(fname):
        return point_cloud_from_path(fname)

    def pcd_to_numpy(self,intensity = True):
        '''

        :return:
        '''
        pcd_x = self.pc_data['x']
        x_nan_bool = np.isnan(pcd_x)
        pcd_y = self.pc_data['y']
        y_nan_bool = np.isnan(pcd_y)
        pcd_z = self.pc_data['z']
        z_nan_bool = np.isnan(pcd_z)
        valid_point_bool = np.logical_or(np.logical_or(x_nan_bool,y_nan_bool),z_nan_bool)
        valid_index = np.argwhere(valid_point_bool == False)
        if intensity:
            pcd_intensity = self.pc_data['intensity'][valid_index]
            pc_numpy = np.stack([pcd_x[valid_index],pcd_y[valid_index],pcd_z[valid_index],pcd_intensity],axis=1)[:,:,0]
            return pc_numpy
        else:
            pc_numpy = np.stack([pcd_x[valid_index], pcd_y[valid_index], pcd_z[valid_index]], axis=1)[:,:, 0]
            return pc_numpy





if __name__ == '__main__':

    # test for lidar
    file_path = "../example/scene-000000/lidar/1656932600.000.pcd"
    pc = PointCloud.from_path(file_path)
    lidar_pc_numpy = pc.pcd_to_numpy(intensity = True)
    # test for radar
    file_path = "../example/scene-000000/radar/points_front/1656932600.000.pcd"
    pc = PointCloud.from_path(file_path)
    radar_pc_numpy = pc.pcd_to_numpy(intensity = False)
    # radar_pc_numpy = pc.radar_pcd_to_numpy
    print("aaaaaa")
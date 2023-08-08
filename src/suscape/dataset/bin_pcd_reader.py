
import numpy as np
import re
import warnings

class BinPcdReader:
    """ Read binary PCD files.
    """
    def __init__(self, filename):
        self.filename = filename
        self.metadata = None
        self.points = None


        numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
        #numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
        self.pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


        self._read()

    def _read(self):
        with open(self.filename, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip()
                header.append(ln)
                if ln.startswith('DATA'.encode()):
                    break

            metadata = self.parse_header(header)
            dtype = self._build_dtype(metadata)
            self.pc_data = self.parse_binary_pc_data(f, dtype, metadata)

  
    def parse_binary_pc_data(self, f, dtype, metadata):
        # print("the dtype.itemsize is: ",dtype.itemsize)
        # print("the dtype['x'] is: ",dtype['x'])
        rowstep = metadata['points']*dtype.itemsize
        # for some reason pcl adds empty space at the end of files
        buf = f.read(rowstep)
        return np.fromstring(buf, dtype=dtype)
    
    def parse_header(self, lines):
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

    def _build_dtype(self, metadata):
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
            np_type = self.pcd_type_to_numpy_type[(t, s)]
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
        
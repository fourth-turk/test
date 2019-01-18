# routines for reading, writing mrc files

import os
import struct
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import io

MODE = []
HEADER = ('nx', 'ny', 'nz', 'mode', 'nxstart', 'nystart', 'nzstart', 
            'mx', 'my', 'mz', 'cella', 'cellb', 'mapc', 'mapr', 'maps', 'dmin', 'dmax', 'mean', 
            'ispg', 'nsymbt', 'exttyp', 'nversion', 'origin', 'mapstr', 'machst', 'rms', 'nlabl')


HEADER_INFO = (
# (variable, data typestart byte , end byte)
    ('nx', 'i', 0, 4), 
    ('ny', 'i', 4, 8),
    ('nz', 'i', 8, 12),
    ('mode', 'i', 12, 16),
    ('nxstart', 'i', 16, 20),
    ('nystart', 'i', 20, 24),
    ('nzstart', 'i', 24, 28),
    ('mx', 'i', 28, 32),
    ('my', 'i', 32, 36),
    ('mz', 'i', 36, 40),
    ('cella', '3f', 40, 52),
    ('cellb', '3f', 52, 64),
    ('mapc', 'i', 64, 68),
    ('mapr', 'i', 68, 72),
    ('maps', 'i', 72, 76),
    ('dmin', 'i', 76, 80),
    ('dmax', 'i', 80, 84),
    ('mean', 'i', 84, 88),
    ('ispg', 'i', 88, 92),
    ('nsymbt', 'i', 92, 96),
    ('exttyp', '4s', 104, 108),
    ('nversion', 'i', 108, 112),
    ('origin', '3f', 196, 208),
    ('map', '4s', 208, 212),
    ('machst', '4s', 212, 216),
    ('rms', 'f', 216, 220),
    ('nlabl', 'i', 220, 224))

SPACEGROUP = {
# header ISPG space group number
    'image_or_imagestack':0,
    'volumeEM_ET':1,
    'volume_stack':401
    }

class mrc:
    """
    useful:
    http://msg.ucsf.edu/IVE/IVE4_HTML/EditHeader2.html

    main mrc class
    mrc 2014 standard
    from: https://doi.org/10.1016/j.jsb.2015.04.002

    1 1–4 Int32 NXNumber of columns
    2 4–8 Int32 NYNumber of rows
    3 9–12 Int32 NZNumber of sections
    4 13–16 Int32 MODEData type
    …
    8 29–32 Int32 MXNumber of intervals along X of the “unit cell”
    9 33–36 Int32 MYNumber of intervals along Y of the “unit cell”
    10 37–40 Int32 MZNumber of intervals along Z of the “unit cell”
    11 –1341–52 Float32 CELLACell dimension in angstroms
    …
    20 77–80 Float32 DMINMinimum density value
    21 81–84 Float32 DMAXMaximum density value
    22 85–88 Float32 DMEANMean density value
    23 89–92 Int32 ISPGSpace group number 0, 1, or 401
    24 93–96 Int32 NSYMBTNumber of bytes in extended header
    …
    27 105–108 Char EXTTYPEExtended header type
    28 109–112 Int32 NVERSIONFormat version identification number
    …
    50–52 197–208 Float32 ORIGINOrigin in X, Y, Z used in transform
    53 209–212 Char MAPCharacter string ‘MAP’ to identify file type
    54 213–216 Char MACHSTMachine stamp
    55 217–220 Float32 RMSRMS deviation of map from mean density
    """
    def __init__(self):
        self.header = None
        self.header_extended = None
        self.img = None

    def _unpack(self, fmt, data, start, end):
        return struct.unpack(fmt, data[start:end])

    def read_mrc(self, path):
        """
        store image part into numpy array
        """
        self.path = path
        with open(self.path, "rb") as f:
            mrc_file = f.read()

        # read heder, determine where header, extender header end
        # return header, image
        self.mrc_file = mrc_file

        # header
        self._header_new(mrc_file)
        # self._header(mrc_file)
        self._pxsize
        self._get_data_type
        self._check_size

        # read image
        image = np.frombuffer(mrc_file, dtype=self.dtype,
                              count=-1, offset=self.header_end)

        # spacegroup for EM: image or volume
        # nx, ny, nz is in numpy array nz, ny, nx
        spacegroup = self.header['ispg']
        if spacegroup == 0 and self.header['nz'] == 1:
            # single image
            self.img = np.reshape(image, (self.header['ny'], self.header['nx']))

        elif spacegroup == 0 and self.header['nz'] != 1:
            # image stack
            self.img = np.reshape(image, (self.header['ny'], self.header['nx']))
        elif spacegroup == 401:
            # EM volume
            self.img = np.reshape(image, (self.header['nz'], self.header['ny'], self.header['nx']))
        print('read img data', self.img.shape)

        # when does it make sense to return attribute
        # return self.img

    def _header_new(self, mrc_file):
        """
        stores header: .header {key:value}
        extended header: .header_extended
        end of header: .header_end
        """
        # header
        # could have made my life easier with multiples of 4 and struct.unpack_from

        header = dict.fromkeys([i[0] for i in HEADER_INFO])

        for i in HEADER_INFO:
            variable, data_type, start_byte, end_byte = i
            if len(data_type) == 1: # to not have tuples of length 1: (value,) -> value
                header[variable] = struct.unpack(data_type, mrc_file[start_byte:end_byte])[0]
            else:
                header[variable] = struct.unpack(data_type, mrc_file[start_byte:end_byte])

        self.header = header

        # extended header
        if header['nsymbt'] != 0:
            self.header_extended = mrc_file[1024:(1024 + header['nsymbt'])]
        elif header['nsymbt'] == 0:
            self.header_extended = None
        else:
            raise ValueError('Extended header not defined in mrc! nsymbt: {}'.format(header['nsymbt']))

        # image data begins at
        self.header_end = 1024 + header['nsymbt']


    def _header_obsolete(self, mrc_file):
        # convention from http://www.ccpem.ac.uk/mrc_format/mrc2014.php#note8
        self.nx = struct.unpack('@i', mrc_file[0:4])
        self.ny = struct.unpack('@i', mrc_file[4:8])
        self.nz = struct.unpack('@i', mrc_file[8:12])
        self.mode = struct.unpack('@i', mrc_file[12:16])
        self.nxstart = struct.unpack('@i', mrc_file[16:20])
        self.nystart = struct.unpack('@i', mrc_file[20:24])
        self.nzstart = struct.unpack('@i', mrc_file[24:28])
        self.mx = struct.unpack('@i', mrc_file[28:32])
        self.my = struct.unpack('@i', mrc_file[32:36])
        self.mz = struct.unpack('@i', mrc_file[36:40])
        self.cella = struct.unpack('@3f', mrc_file[40:52])
        self.cellb = struct.unpack('@3f', mrc_file[52:64])
        self.mapc = struct.unpack('@i', mrc_file[64:68])
        self.mapr = struct.unpack('@i', mrc_file[68:72])
        self.maps = struct.unpack('@i', mrc_file[72:76])
        self.dmin = struct.unpack('@f', mrc_file[76:80])
        self.dmax = struct.unpack('@f', mrc_file[80:84])
        self.mean = struct.unpack('@f', mrc_file[84:88])
        self.ispg = struct.unpack('@i', mrc_file[88:92])
        self.nsymbt = struct.unpack('@i', mrc_file[92:96])
        # self.extra = struct.unpack('@220s', mrc_file[0:220]) # ...
        self.exttyp = struct.unpack('@4s', mrc_file[104:108])
        self.nversion = struct.unpack('@i', mrc_file[108:112])  # ...
        self.origin = struct.unpack('@3f', mrc_file[196:208])
        self.map = struct.unpack('@4s', mrc_file[208:212])[0].decode()
        # self.machst = struct.unpack('@4x', mrc_file[212:216])[0] # endiannes
        self.machst = mrc_file[212:216]  # endiannes
        self.rms = struct.unpack('@f', mrc_file[216:220])
        self.nlabl = struct.unpack('@i', mrc_file[220:224])
        # self.labels = struct.unpack('@800s', mrc_file[224:1024])[0].decode()

        # extended header
        """
        Word 24 contains the item NSYMBT in the MRC/CCP4 format (see Table 1), which gives the number
        of bytes used for symmetry data inserted between the main header (1024 bytes in length) and the data block
        from: https://doi.org/10.1016/j.jsb.2015.04.002
        """
        self.header_end = 1024 + self.nsymbt[0]

        # self.header = {'nx':self.nx, 'ny':self.ny, 'nz':self.nz, 'mode':self.mode,
        #                'nxstart':self.nxstart, 'nystart':self.nystart, 'nzstart':self.nzstart,
        #                'mx':self.mx, 'my':self.my, 'mz':self.mz, 'cella':self.cella, 'cellb':self.cellb,
        #                'mapc':self.mapc, 'mapr':self.mapr, 'maps':self.maps, 'dmin':self.dmin,
        #                'dmax':self.dmax, 'mean':self.mean, 'ispg':self.ispg, 'nsymbt':self.nsymbt,
        #                'exttyp':self.exttyp, 'nversion':self.nversion, 'origin':self.origin,
        #                'map':self.map, 'machst':self.machst, 'rms':self.rms, 'nlabl':self.nlabl}

    @property
    def _pxsize(self):
        self.pxsize = self.header['cella'][0] / self.header['nx']
        return self.pxsize

    @property
    def _get_data_type(self):
        """
        header bytes 13-16 = mode:
        0 8-bit signed integer (range -128 to 127)
        1 16-bit signed integer
        2 32-bit signed real
        3 transform : complex 16-bit integers
        4 transform : complex 32-bit reals
        6 16-bit unsigned integer
        """
        dtype = None
        if self.header['mode'] == 0:
            dtype = np.int8

        elif self.header['mode'] == 1:
            dtype = np.int16

        elif self.header['mode'] == 2:
            dtype = np.float32

        elif self.header['mode'] == 3:
            # no native complex int16 dtype, needs np.dtype([('re', np.int16), ('im', np.int16)])
            dtype = None

        elif self.header['mode'] == 4:
            dtype = np.complex32

        elif self.header['mode'] == 6:
            dtype = np.uint16
        else:
            raise ValueError('np.dtype: {}, mode: {} (mode 3 complex int16 is not supported)'.format(
                dtype, self.header['mode']))

        self.dtype = dtype
        # return self.dtype

    def _get_mode(self, data_type):
        """
        data type of the mrc data
        returns mode [0,..,6] for writing a new mrc header
        """
        mode = None
        if data_type == np.int8:
            mode = 0
        if data_type == np.int16:
            mode = 1
        if data_type == np.float32:
            mode = 2
        if data_type == np.complex64:
            mode = 4
        if data_type == np.uint16:
            mode = 6
        if data_type == None:
            raise ValueError('data type is: {}'.format(data_type))
        if mode != None:
            return mode
        else:
            raise ValueError('mode not found for data type: {}'.format(data_type))


    @property
    def _check_size(self):
        """
        check whether or not the mrc file has size expected from header
        """
        # fails for 2d images, mult by 0

        image_size = self.header['nx'] * self.header['ny'] * \
            self.header['nz'] * np.dtype(self.dtype).itemsize
        expected_size = self.header_end + image_size
        file_size = os.path.getsize(self.path)

        print('check file size:')
        print('header: {}'.format(self.header_end))
        print('header extended: {}'.format(self.header_extended))
        print('image: {} ({},{},{},{} bytes)'.format(
            image_size, self.header['nx'], self.header['ny'], self.header['nz'], np.dtype(self.dtype).itemsize))
        print('expected size = {}, actual size = {}'.format(
            expected_size, file_size))
        if expected_size != file_size:
            raise ValueError(
                'file size not matching header info!!! file: {} image_size: {}'.format(self.path, image_size))

    def header_print(self):
        for l in HEADER_INFO:
            key = l[0]
            print(key, ':', self.header[key])


    def write_mrc(self, img, path):
        # construct header
        frame = bytearray(1024)

        # header_args
        ha = dict.fromkeys(['nx', 'ny', 'nz', 'mode', 'nxstart', 'nystart', 'nzstart', 
            'mx', 'my', 'mz', 'cella', 'cellb', 'mapc', 'mapr', 'maps', 'dmin', 'dmax', 'mean', 
            'ispg', 'nsymbt', 'exttyp', 'nversion', 'origin', 'map', 'machst', 'rms', 'nlabl'])

        # # update header from attributes
        for key, value in self.header.items():
            if key in ha:
                ha[key] = value
        print(ha)

        # header should it be **args, *kwargs, list, dict
        # want to avoid typing in several time all the attributes
        # for mode a list outside the class for more readable code?
        # when reading a file put header attributes in self.header.*
        # posittion and whole int and float types could be in a dictionary or list outside


        # def _unpack(self, fmt, data):
        #     return struct.unpack(self._endian + fmt, data)

        # def _pack(self, fmt, *values):
        #     return struct.pack(self._endian + fmt, *values)


        # things calculated for new image
        # dimension .shape
        if len(img.shape) == 2:
            ny, nx = img.shape
            nz = 1
        elif len(img.shape) == 3: # volume or img stack
            ny, nx, nz = img.shape # not tested

        mode = self._get_mode(img.dtype)
        dmin = np.amin(img)
        dmax = np.amax(img)
        mean = np.mean(img)

        ha['nx'] = nx
        ha['ny'] = ny
        ha['nz'] = nz
        ha['mode'] = self._get_mode(img.dtype)
        ha['nxstart'] = self.header['nxstart']
        ha['nystart'] = self.header['nystart']
        ha['nzstart'] = self.header['nzstart']
        ha['mx'] = nx
        ha['my'] = ny
        ha['mz'] = self.header['mz']
        ha['cella'] = (self.pxsize * nx, self.pxsize * ny, self.pxsize * ha['mz'])
        ha['cellb'] = (90.0, 90.0, 90.0)
        ha['dmin'] = np.amin(img)
        ha['dmax'] = np.amax(img)
        ha['mean'] = np.mean(img)
        ha['nversion'] = 20140  # mrc 2014, version 0
        ha['map'] = bytes('MAP ', 'utf-8') # always 4 bytes, space after MAP
        ha['rms'] = np.std(img)
        # ha['rms'] = np.sqrt(np.mean(np.square(img)))

        print(np.square(img))
        print(np.mean(np.square(img)))
        print(np.sqrt(np.mean(img)))
        print(img.std)
        print(np.std(img))

        # can pack the statements in a function to have less code
        # if I do it multiples of 4, i can simplify it into:
        # for i in HEADER_INFO:
        #     print(i)
        #     struct.pack_into(i[1], frame, i[2], ha[i[0]])
        
        struct.pack_into('@i', frame, 0, ha['nx'])
        struct.pack_into('@i', frame, 4, ha['ny'])
        struct.pack_into('@i', frame, 8, ha['nz'])
        struct.pack_into('@i', frame, 12, ha['mode'])
        struct.pack_into('@i', frame, 16, ha['nxstart'])
        struct.pack_into('@i', frame, 20, ha['nystart'])
        struct.pack_into('@i', frame, 24, ha['nzstart'])
        struct.pack_into('@i', frame, 28, ha['mx'])
        struct.pack_into('@i', frame, 32, ha['my'])
        struct.pack_into('@i', frame, 36, ha['mz'])
        struct.pack_into('@3f', frame, 40, ha['cella'][0], ha['cella'][1], ha['cella'][2])
        struct.pack_into('@3f', frame, 52, ha['cellb'][0], ha['cellb'][1], ha['cellb'][2])
        struct.pack_into('@i', frame, 64, ha['mapc'])
        struct.pack_into('@i', frame, 68, ha['mapr'])
        struct.pack_into('@i', frame, 72, ha['maps'])
        struct.pack_into('@f', frame, 76, ha['dmin'])
        struct.pack_into('@f', frame, 80, ha['dmax'])
        struct.pack_into('@f', frame, 84, ha['mean'])
        struct.pack_into('@i', frame, 88, ha['ispg'])
        struct.pack_into('@i', frame, 92, ha['nsymbt'])
        struct.pack_into('@4s', frame, 104, ha['exttyp'][0])
        struct.pack_into('@i', frame, 108, ha['nversion'])
        struct.pack_into('@3f', frame, 196, ha['origin'][0], ha['origin'][1], ha['origin'][2])
        struct.pack_into('@4s', frame, 208, ha['map'])
        struct.pack_into('@4s', frame, 212, ha['machst'][0])  # endiannes
        struct.pack_into('@f', frame, 216, ha['rms'])
        struct.pack_into('@i', frame, 220, ha['nlabl'])

        self.header_print()
        new_file = frame + img.tobytes()

        with open(path, "wb") as f:
            f.write(new_file)

    def make_slice(self, img, from_x, width):
        """
        returns slice of img from [x_coord, x_coord+width]
        """
        print('\nimg debug')
        print(img.shape)
        nx, ny = img.shape
        # im = img.reshape(nx, ny)
        # img = np.flip(img, 1)
        # nx, ny = img.squeeze().shape
        print(nx, ny)
        # print(img.squeeze()[:,0:3709].shape)
        
        to_x = from_x + width
        if to_x < self.header['nx']:
            # return img
            # return img[:, :3709, :]
            return img[:, from_x:to_x]
        else:
            ValueError('slice width is outside of image, nx: {}, from_x: {}, to_x: {}'.format(nx, from_x, to_x))

    def sliding_slices(self):
        pass
        # think over the steps for making slices
        # a set of slices, with some overlap
        # each new header
        # new filename with number of slice
        # call external gctf fitting
        # read the text file output results
        # when are results finished
        # read the results to array
        # fit a line to array defocus values

        # gctf fitting on the whole image for comparison
        # astigmatism values here and slices
        


def linear_transformation(src, a):
    # from https://mmas.github.io/linear-transformations-numpy
    M, N = src.shape
    points = np.mgrid[0:N, 0:M].reshape((2, M*N))
    new_points = np.linalg.inv(a).dot(points).round().astype(int)
    x, y = new_points.reshape((2, M, N), order='F')
    indices = x + N*y
    return np.take(src, indices, mode='wrap')


def test_image_read_write():
    fp = os.path.expanduser('~/wrk/test_mrc/0001.mrc')
    showplot = False

    test = mrc()
    test.read_mrc(fp)
    print(test.img.shape)
    # slice = test.make_slice(test.img, 0, 3837 )#test.img.shape[1]-1)
    slice = test.img[:,0:3000]

    if showplot:
        plt.imshow(slice)
        plt.gray()
        plt.show()

    # write 
    fp2 = os.path.expanduser('~/wrk/test_mrc/test_slice.mrc')
    test.write_mrc(slice, fp2)

    # read test
    test2 = mrc()
    test2.read_mrc(fp2)

    if showplot:
        plt.imshow(test2.img)
        plt.gray()
        plt.show()



def main():

    test_image_read_write()

    # # path = '/Users/martin/wrk/0001.mrc'
    # path = '/home/martin/wrk/test_mrc/0001.mrc'
    # # path = '/home/martin/wrk/run87_class001.mrc'
    # test = mrc()
    # test.read_mrc(path)
    # sl = test.make_slice(test.img, 3000, 700)
    # test.write_mrc(sl)    
    # # print(test.headerprint)


    # # pf = '/Users/martin/wrk/test/test_slice.mrc'
    # pf = '/home/martin/wrk/test/test_slice.mrc'
    # # # image display
    # tt = mrc()
    # tt.read_mrc(pf)
    # plt.imshow(tt.img)
    # plt.gray()
    # plt.show()

    # add stripes, make band
    # write out mrc

    # # transformations
    # # lin trans: parallel stay parallel
    # # scaling: A[x y]
    # # A [ 2 0 ]
    # #   [ 0 1 ]
    # aux = np.ones((100, 100), dtype=int)
    # src = np.vstack([np.c_[aux, 2*aux], np.c_[3*aux, 4*aux]])
    # angle = np.sin(np.pi*60./180.)
    # # angle = 1
    # A = np.array([[angle, 0],
    #               [0, 1]])
    # # src = test.img[80]
    # dst = linear_transformation(src, A)
    # # dst = np.dot(test.img[80], A)
    # plt.imshow(dst)
    # plt.show()

    my_dpi = 96

    # print(tt.img.shape)
    # plt.figure(figsize=(tt.img.shape[0]/my_dpi/3, tt.img.shape[1]/my_dpi/3), dpi=my_dpi)
    # plt.imshow(tt.img.squeeze())
    # plt.gray()
    # plt.show()

    # print(test.header)

    # for i in HEADER_INFO:
    #     print(i)
    #     # print(_unpack(i[3], test.mrc_file, i[1], i[2]))
    # print(test.header)

if __name__ == "__main__":
    main()

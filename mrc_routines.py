# routines for reading, writing mrc files


import os
import struct
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import io

class mrc:
    """
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
        # self.path2 = path2
        # self.header = None
        self.headerext = None
        self.image = None
        # print(self.__dict__)

    def header(self, path):
        self.path = path
        with open(self.path, "rb") as f:
            header = 220 # 220 bytes
            data = f.read(header)
            self.nx = struct.unpack('@i', data[0:4])
            self.ny = struct.unpack('@i', data[4:8])
            self.nz = struct.unpack('@i', data[8:12])
            self.mode = struct.unpack('@i', data[12:16]) # ...
            self.mx = struct.unpack('@i', data[28:32])
            self.my = struct.unpack('@i', data[32:36])
            self.mz = struct.unpack('@i', data[36:40])
            self.cella = struct.unpack('@3f', data[40:52])
            self.cellangles = struct.unpack('@3f', data[52:64]) # ...
            self.cellzzz = struct.unpack('@3f', data[64:76]) # ...
            self.dmin = struct.unpack('@f', data[76:80])
            self.dmax = struct.unpack('@f', data[80:84])
            self.mean = struct.unpack('@f', data[84:88])
            self.ispg = struct.unpack('@i', data[88:92])
            self.nsymbt = struct.unpack('@i', data[92:96])
            # self.nsymbtzzz = struct.unpack('@2i', data[96:104]) # ...
            self.char = struct.unpack('@4c', data[104:108])
            self.nversion = struct.unpack('@i', data[108:112])
            # self.gapzzz = struct.unpack('@21f', data[112:196]) # ...
            self.origin = struct.unpack('@3f', data[196:208])
            self.map = struct.unpack('@4c', data[208:212])
            self.machst = struct.unpack('@4c', data[212:216])
            self.rms = struct.unpack('@f', data[216:220])

            self.pxsize = self.cella[0] / self.nx[0]
            """
            Word 24 contains the item NSYMBT in the MRC/CCP4 format (see Table 1), which gives the number\ 
             of bytes used for symmetry data inserted between the main header (1024 bytes in length) and the data block
            from: https://doi.org/10.1016/j.jsb.2015.04.002
            """

            self.start = 1024 + self.nsymbt[0]

    def read_mrc(self, path):
        """
        store image part into numpy array
        """
        self.path = path
        with open(self.path, "rb") as f:
            image_data = f.read()


        # header




        b  = np.frombuffer(image_data, dtype=np.int32, count=-1, offset=1024)
        self.img = np.reshape(b, (160,160,160))            

        return self.img

        # image = Image.open(io.BytesIO(image_data[self.start : self.get_size]))
        #image.show()
        # print(a.byteorder)

        # a = np.frombuffer(image_data, dtype=int, count=-1, offset=0)
        # return a



    def check_size(self):
        """
        check whether or not the mrc file has size expected from header
        """
        expected_size = 1024 + self.nsymbt[0] * 80 * 4 + self.nx[0] * self.ny[0] * self.nz[0] * 4 # for test, mode is ignored
        file_size = self.get_size

        print('check file size: expected size = {} actual size = {}'.format(expected_size, file_size))
        if expected_size != file_size:
            print("file size not matching header info!!!")


    @property
    def get_size(self):
        return os.path.getsize(self.path)

    @property
    def headerprint(self):
        # keys = []
        for key, value in self.__dict__.items():
            print(key, ':', value)            
        #     keys.append([key, value])
        # return self.__dict__

    def addsth(self, add):
        self.add = add
        return self.add + 'ssdsdsd'




def main():
    
    path = '/Users/martin/wrk/run1.mrc'
    test = mrc()
    test.header(path)
    # test.header('/home/martin/wrk/run87_class001.mrc')

    print( test.headerprint )
    print( test.get_size )
    test.check_size()

    print( test.addsth('martin'))

    a = test.read_mrc(path)
    e = 'main() end'
    print(e)

if __name__ == "__main__":
    main()

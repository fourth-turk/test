# routines for reading, writing mrc files


import os
import struct

class mrc:
    """
    main mrc class
    mrc 2014 standard

    11–4Int32NXNumber of columns
    24–8Int32NYNumber of rows
    39–12Int32NZNumber of sections
    413–16Int32MODEData type
…
    829–32Int32MXNumber of intervals along X of the “unit cell”
    933–36Int32MYNumber of intervals along Y of the “unit cell”
    1037–40Int32MZNumber of intervals along Z of the “unit cell”
    11–1341–52Float32CELLACell dimension in angstroms
…
    2077–80Float32DMINMinimum density value
    2181–84Float32DMAXMaximum density value
    2285–88Float32DMEANMean density value
    2389–92Int32ISPGSpace group number 0, 1, or 401
    2493–96Int32NSYMBTNumber of bytes in extended header
…
    27105–108CharEXTTYPEExtended header type
    28109–112Int32NVERSIONFormat version identification number
…
    50–52197–208Float32ORIGINOrigin in X, Y, Z used in transform
    53209–212CharMAPCharacter string ‘MAP’ to identify file type
    54213–216CharMACHSTMachine stamp
    55217–220Float32RMSRMS deviation of map from mean density

    """
    def __init__(self):
        # self.path2 = path2
        # self.header = None
        self.headerext = None
        self.image = None
        print(self.__dict__)

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

    def check_size(self):
        """
        check whether or not the mrc file has size expected from header
        """
        expected_size = 1024 + self.nsymbt[0] * 80 * 4 + self.nx[0] * self.ny[0] * self.nz[0] * 4 # for test, mode is ignored
        file_size = self.get_size

        if expected_size != file_size:
            print("file size not matching header info!!!")
            print('check file size: expected size = {} actual size = {}'.format(expected_size, file_size))

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



test = mrc()
test.header('/Users/martin/wrk/run1.mrc')

print( test.headerprint )
print( test.get_size )
test.check_size()

print( test.addsth('martin'))

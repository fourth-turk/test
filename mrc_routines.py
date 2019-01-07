# routines for reading, writing mrc files

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
    def __init__(self, path):
        self.path = path

    def header(self):
        with open(self.path, "rb") as f:
            header = 220 # 220 bytes
            data = f.read(header)
            self.num_columns = struct.unpack('@',data[0-4])


test = mrc("/home/martin/wrk/run87_class001.mrc")
test.header()

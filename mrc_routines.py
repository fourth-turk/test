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

   def read_mrc(self, path):
      """
      store image part into numpy array
      """
      self.path = path
      with open(self.path, "rb") as f:
         mrc_file = f.read()


      # header
      self._header(mrc_file)
      self._pxsize
      self._data_mode
      self._check_size
        
      image  = np.frombuffer(mrc_file, dtype=self.dtype, count=-1, offset=self.header_size)
      self.img = np.reshape(image, (self.nx[0], self.ny[0], self.nz[0]))            
  
      # return self.img


   def _header(self, mrc_file):
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
      self.exttyp = struct.unpack('@4c', mrc_file[104:108])
      self.nversion = struct.unpack('@i', mrc_file[108:112]) # ...
      self.origin = struct.unpack('@3f', mrc_file[196:208])
      self.map = struct.unpack('@4s', mrc_file[208:212])[0].decode()
      # self.machst = struct.unpack('@4x', mrc_file[212:216])[0] # endiannes
      self.machst = mrc_file[212:216] # endiannes
      self.rms = struct.unpack('@f', mrc_file[216:220])
      self.nlabl = struct.unpack('@i', mrc_file[220:224])
      # self.labels = struct.unpack('@800s', mrc_file[224:1024])[0].decode()

      # extended header
      """
      Word 24 contains the item NSYMBT in the MRC/CCP4 format (see Table 1), which gives the number\ 
      of bytes used for symmetry data inserted between the main header (1024 bytes in length) and the data block
      from: https://doi.org/10.1016/j.jsb.2015.04.002
      """
      self.header_size = 1024 + self.nsymbt[0]


   @property
   def _pxsize(self):
      self.pxsize = self.cella[0] / self.nx[0]
      return self.pxsize

   @property
   def _data_mode(self):
      """
      bytes 13-16 = mode:
      0 8-bit signed integer (range -128 to 127) 
      1 16-bit signed integer
      2 32-bit signed real
      3 transform : complex 16-bit integers
      4 transform : complex 32-bit reals
      6 16-bit unsigned integer
      """
      self.dtype = None

      if self.mode[0] == 0:
         self.dtype = np.int8

      elif self.mode[0] == 1:
         self.dtype = np.int16

      elif self.mode[0] == 2:
         self.dtype = np.float32

      elif self.mode[0] == 3:
         self.dtype = None # no native dtype, needs np.dtype([('re', np.int16), ('im', np.int16)])

      elif self.mode[0] == 4:
         self.dtype = np.complex32

      elif self.mode[0] == 6:
         self.dtype = np.uint16
        
      else:
         raise ValueError('np.dtype: {}, mode: {} (mode 3 complex int16 is not supported)'.format(self.dtype, self.mode[0]))

      return self.dtype

   @property
   def _check_size(self):
      """
      check whether or not the mrc file has size expected from header
      """
      image_size = self.nx[0] * self.ny[0] * self.nz[0] * np.dtype(self.dtype).itemsize
      expected_size =  self.header_size + image_size
      file_size = os.path.getsize(self.path)

      print('check file size:')
      print('header: {}'.format(self.header_size))
      print('image: {} ({},{},{},{} bytes)'.format(image_size, self.nx[0], self.ny[0], self.nz[0], np.dtype(self.dtype).itemsize))
      print('expected size = {}, actual size = {}'.format(expected_size, file_size))
      if expected_size != file_size:
         raise ValueError('file size not matching header info!!! file: {}'.format(self.path))

   @property
   def headerprint(self):
      # keys = []
      for key, value in self.__dict__.items():
         print(key, ':', value)            
      #     keys.append([key, value])
      # return self.__dict__

   def make_slice(self, img, from_x, width):
      """
      returns slice of img from [x_coord, x_coord+width]
      """
      if to_x < self.nx[0]:
         return img[:, from_x:(from_x+width)]
      else:
         ValueError('slice width is outside of image, nx: {}, from_x: {}, to_x: {}'.format(self.nx, from_x, to_x))

   def get_mode(self, data_type):
      pass

   def write_mrc(self, img, header):
      pass
      # construct header
      frame = bytearray(1024)
      # struct.pack_into(fmt, buffer, offset, v1)

      # dimension .shape
      nx, ny, nz = img.shape

      # mode
      mode = 2

      dmin = img.shape

      struct.pack_into('@i', frame, 0, nx)
      struct.pack_into('@i', frame, 4, ny)
      struct.pack_into('@i', frame, 8, nz)
      struct.pack_into('@i', frame, 12, mode)
      struct.pack_into('@i', frame, 16, nxstart)
      struct.pack_into('@i', frame, 20, nystart)
      struct.pack_into('@i', frame, 24, nzstart)
      struct.pack_into('@i', frame, 28, mx)
      struct.pack_into('@i', frame, 32, my)
      struct.pack_into('@i', frame, 36, mz)
      struct.pack_into('@3f', frame, 40, cella)
      struct.pack_into('@3f', frame, 52, cellb)
      struct.pack_into('@i', frame, 64, mapc)
      struct.pack_into('@i', frame, 68, mapr)
      struct.pack_into('@i', frame, 72, maps)
      struct.pack_into('@f', frame, 76, dmin)
      struct.pack_into('@f', frame, 80, dmax)
      struct.pack_into('@f', frame, 84, mean)
      struct.pack_into('@i', frame, 88, ispg)
      struct.pack_into('@i', frame, 92, nsymbt)
      struct.pack_into('@4c', frame, 104, exttyp)
      struct.pack_into('@i', frame, 108, nversion) # ...
      struct.pack_into('@3f', frame, 196, origin)
      struct.pack_into('@4s', frame, 208, mapstr)
      struct.pack_into('@4s', frame, 212, machst) # endiannes
      struct.pack_into('@f', frame, 216, rms)
      struct.pack_into('@i', frame, 220, nlabl)


def linear_transformation(src, a):
   # from https://mmas.github.io/linear-transformations-numpy
   M, N = src.shape
   points = np.mgrid[0:N, 0:M].reshape((2, M*N))
   new_points = np.linalg.inv(a).dot(points).round().astype(int)
   x, y = new_points.reshape((2, M, N), order='F')
   indices = x + N*y
   return np.take(src, indices, mode='wrap')


def main():
    
   path = '/Users/martin/wrk/run1.mrc'
   # path = '/home/martin/wrk/run87_class001.mrc'
   test = mrc()
   test.read_mrc(path)
   # print(test.headerprint)

   # # image display
   # plt.pyplot.imshow(test.img[80]) 
   # plt.pyplot.gray()
   # plt.pyplot.show()

   # add stripes, make band
   # write out mrc

   # transformations
   # lin trans: parallel stay parallel
   # scaling: A[x y]
   # A [ 2 0 ]
   #   [ 0 1 ]
   aux = np.ones((100, 100), dtype=int)
   src = np.vstack([np.c_[aux, 2*aux], np.c_[3*aux, 4*aux]])
   angle = np.sin(np.pi*60./180.)
   # angle = 1
   A = np.array([[angle, 0],
               [0, 1]])
   # src = test.img[80]
   dst = linear_transformation(src, A)
   # dst = np.dot(test.img[80], A)
   plt.imshow(dst)
   plt.show()


if __name__ == "__main__":
   main()

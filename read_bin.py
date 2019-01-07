import pickle
import cPickle
import struct
import sys, os
from PyQt4 import QtCore, QtGui


# This example demonstrates how to read a binary file, by reading the width and
# height information from a bitmap file. First, the bytes are read, and then
# they are converted to integers.
# When reading a binary file, always add a 'b' to the file open mode


def read_bytes( block, start, end ):
    position=start
    while ( position < end ):
        a=position
        b=position+4
        string = struct.unpack('<4s', block[a:b] )
        num = struct.unpack('<i', block[a:b] )
        # num1 = struct.unpack('>i', block[a:b] )
        # num1 = struct.unpack('<i', block[a+1:b+1] )
        print("%03d [%03d:%03d] \t %s \t %s" % (a/4+1, a, b, string, num) )
        position = position + 4

def write_bytes( data, name, istart, iend ):
    header=1024
    with open( name + '.mrc', 'wb') as f:
        f.write(data[0:header])
        f.write(data[istart:iend])

filename = '2014-06-24-man-sel.mrc'

with open( filename, 'rb') as f:
    data = bytearray(os.path.getsize( filename ))
    f.readinto(data)
    
read_bytes( buffer(data), 0, 1024)

# print struct.calcsize( buffer( data[0:4]) )
print struct.unpack('i', buffer( data[8:12]) )
data[8:12]= struct.pack('i', 1)
print struct.unpack('i', buffer( data[8:12]) )


sys.exit()
header=1024
image_size = 160*160*4
size=header+image_size
print 
print size









print len( data )
for i in xrange(0,512,1):
    print data[i]
    


#f = open('2014-06-24-man-sel.mrc', 'rb')
#read_bytes( f.read(1024), 0, 512)



pkl_file = open('2014-06-24-man-sel.mrc', 'rb')
data1 = pickle.loads(pkl_file.read() )
    


with open('2014-06-24-man-sel.mrc', 'r+b') as f:
    # byte = f.read(1)
    block = f.read(1024)

    x=1
    new= struct.pack('<i', x )
    print struct.unpack('<i', new)
    
    # read_bytes( block, 0, 512)
    # struct.pack_into('<i', block, 12, new )


with open('new_file.mrc', 'wb') as f:
    f.write( block )
    # f.seek(8)
    # f.write( bytearray( 1000 ))

with open('new_file.mrc', 'rb') as f:
    f.seek(0)
    new_block= f.read(1024)
    print len( new_block)
    read_bytes( new_block , 0, 512)
    

x=[ 1 ]
print bytearray( x )        
b = bytearray(  block )
print len(block)
# block.see[8]=bytes(0)
b[8]=1
b[9]=0
b[10]=0
b[11]=0

print b, struct.unpack( '<i', b[8:12])


print struct.unpack( '<iiii', b[0:16])


# struct.pack_into( '<i', block[0:4], 1, 1)

    # while( len(byte) < 10 ):
        # size = struct.unpack('<cc', byte )

    # image= f.read(


    # BMP files store their width and height statring at byte 18 (12h), so seek
    # to that position
    # f.seek(18)

    # The width and height are 4 bytes each, so read 8 bytes to get both of them
    # bytes = f.read(8)

    # Here, we decode the byte array from the last step. The width and height
    # are each unsigned, little endian, 4 byte integers, so they have the format
    # code '<II'. See http://docs.python.org/3/library/struct.html for more info


    # Print the width and height of the image



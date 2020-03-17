#Python read binary boxes
#Author: Willem Elbers
#Date: 3 February 2019

import numpy as np;
import struct; #for bytes to float
from matplotlib import pyplot as plt;
import os;
import sys;

fname = sys.argv[1];

#Function to read binary files chunk by chunk
#Source: @codeape, https://stackoverflow.com/a/1035456
def bytes_from_file(filename, chunksize=512*512*512*4):
	with open(filename, "rb") as f:
		while True:
			chunk = f.read(chunksize);
			if chunk:
				yield from chunk;
			else:
				break;

def to_bytes_file(filename, arr):
	with open(filename, "wb") as f:
		size = arr.size;
		for i in range(size):
			z = i % width;
			y = int(i/width) % width;
			x = int(i/(width*width)) % width;
			the_float = arr[x,y,z];
			four_bytes = struct.pack('f', the_float);
			f.write(four_bytes);
		f.close();

def read_binary_array(filename, width):
	#Create empty array
	arr = np.zeros(width*width*width).reshape(width, width, width);

	#Read the file byte by byte
	#Each 4 bytes correspond to a C-style 32-bit float at location (x,y,z)
	i = 0;
	bytes = [];
	for b in bytes_from_file(fname):
		bytes.append(b);

		i += 1;
		j = i % 4;
		z = int((i-1)/4) % width;
		y = int((i-1)/(4*width)) % width;
		x = int((i-1)/(4*width*width)) % width;

		if j == 0:
			four_bytes = struct.pack('4B', *bytes);
			the_float = struct.unpack('f', four_bytes)[0];
			#We only need to overwrite if the value is nonzero
			if the_float != 0:
				arr[x,y,z] = the_float;
			bytes = [];
	return(arr)

#The array format is row-major: INDEX = z + width * (y + width*x).

#Get the box width
float_size = 4; #bits
floats_num = os.path.getsize(fname)/float_size;
width = round(floats_num**(1./3));

print("Box width:", width);
print("Reading the box from disk.");

#Read the array
arr = read_binary_array(fname,width);

#Plot a slice of the array
plt.imshow(arr[16]);plt.colorbar();plt.xlabel("z");plt.ylabel("y");plt.show();

print("");
print("Sum:\t", arr.sum());
print("Sigma:\t", np.sqrt(arr.var()));

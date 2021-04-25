import numpy as np
import h5py
import sys;

fname = sys.argv[1];

print("Unpadding Field/Field in ", fname, "by 2 along the z-axis");

f = h5py.File(fname, mode="r+")
arr = f["Field/Field"][:]
brr = arr[:,:,:-2]

print("New dimensions:", brr.shape)

del f["Field/Field"]
f["Field/Field"] = brr
f.close()

print("Done.")

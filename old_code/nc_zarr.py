import h5py
import numpy as np
import os

from kerchunk.hdf import SingleHdf5ToZarr


# CMIP6 files
# cmip6_test_file = r"/home/valeriu/climate_data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/sos/gn/v20190710/sos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_198001-198412.nc"

V = "/home/valeriu/climate_data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1/Omon/sos/gn/v20190710/"
BRYAN = "./"

filename = 'sos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_198001-198412.nc'
filename = 'tas_Amon_BCC-ESM1_historical_r1i1p1f1_gn_185001-201412.nc'
cmip6_test_file = BRYAN + filename


rfs = SingleHdf5ToZarr(cmip6_test_file)
ds = rfs.translate()
print("\nKerchunk-IT stuffs")
print("======================")
fileread = ds["templates"]["u"]
print(f"File read is {fileread}")
print("NetCDF file attrs: ", ds["refs"]['.zattrs'])
time_chunk_keys = []
for k, _ in ds["refs"].items():
    if "sos" in k:
        if k == "sos/.zarray":
            print("\nVariable array specs:", ds["refs"][k])
        elif k == "sos/.zattrs":
            print("\nVariable array attrs:", ds["refs"][k])
        else:
            print("Time chunk offset and size:", ds["refs"][k][1], ds["refs"][k][2])
chunk_sizes = []
for val in ds["refs"].values():
    if isinstance(val[2], int) or isinstance(val[2], float):
        chunk_sizes.append(val[2])
print("\nMin chunk size", np.min(chunk_sizes))
print("Max chunk size", np.max(chunk_sizes))
print("Total size (sum of chunks), UNCOMPRESSED:", np.sum(chunk_sizes))
print("\n")

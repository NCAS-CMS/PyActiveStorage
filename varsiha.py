import concurrent.futures
import numpy as np
import fsspec
import pyfive

def load_from_https(uri):
  """
  opening https file from uri using fsspec and pyfive
  """
   
  client_kwargs = {'auth': None}
  fs = fsspec.filesystem('http', **client_kwargs)
  http_file = fs.open(uri, 'rb')

  ds = pyfive.File(http_file)
  print(f"Dataset loaded from https with Pyfive: {uri}")
  return ds

def _iterate_range(i):
  """
  to iterate over various slices of the dataset
  """
  return ds[0:i]  # np.min(ds[0:i])


# ------------------------------------------
# --- Configuration which server testing ---
# ------------------------------------------

current_test = 'CEDA' 

# Define your environments
servers = {
  'CEDA': {
    'uri': "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/AerChemMIP/MOHC/UKESM1-0-LL/ssp370SST-lowNTCF/r1i1p1f2/Amon/cl/gn/latest/cl_Amon_UKESM1-0-LL_ssp370SST-lowNTCF_r1i1p1f2_gn_205001-209912.nc",
    'var': "cl"
  },
  'DKRZ': {
    'uri': "http://esgf3.dkrz.de/thredds/fileServer/cmip6/RFMIP/MPI-M/MPI-ESM1-2-LR/piClim-spAer-anthro/r1i1p1f2/Amon/clw/gn/v20190710/clw_Amon_MPI-ESM1-2-LR_piClim-spAer-anthro_r1i1p1f2_gn_184901-187912.nc",
    'var': "clw"
  },
  'JASMIN': {
    'uri': "https://gws-access.jasmin.ac.uk/public/canari/varsiha/clw_Amon_MPI-ESM1-2-LR_piClim-spAer-anthro_r1i1p1f2_gn_184901-187912.nc",
    'var': "clw"
  }
}

if current_test not in servers:
  raise ValueError(f"Unknown server: {current_test}. Choose from {list(servers.keys())}")

# Extract settings
config = servers[current_test]
print(f"--- Running Test on {current_test} ---")

file_obj = load_from_https(config['uri'])
ds = file_obj[config['var']]

with concurrent.futures.ThreadPoolExecutor(
  max_workers=100) as executor:
  futures = []
  for i in range(50):
    print("Thread ", i, " submitted")
    future = executor.submit(_iterate_range, i)
    futures.append(future)
  print("-- Checking completed threads-- ")
  for future in concurrent.futures.as_completed(futures):
    try:
      s = future.result()
      print("Thread ", np.shape(s)[0]," completed with Success")
    except Exception as exc:
      print(exc)
      #raise

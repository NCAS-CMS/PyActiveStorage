p5rem
=====

*Remote HDF5/NetCDF chunk access and reduction over SSH.*

`p5rem` lets desktop tools read and compute over HDF5/NetCDF or pp/fields files on a remote
HPC system — with no custom server infrastructure, open ports, or host
administrator involvement.  A small server is bootstrapped over
a standard SSH connection and communicates with the client entirely through
stdin/stdout.  No TCP port forwarding is needed. The server exploits
`pyfive <https://github.com/ncas-cms/pyfive>`_ or `ppfive <https://github.com/ncas-cms/ppfive>`_ running in the remote Python environment to access the files, and the client
fetches metadata and chunks on demand, optionally caching them locally for speed. The
server can also carry out specified reductions (e.g. mean, min, max) on the remote host, 
returning just the result.   

`p5rem` was designed for use in the `xconv2 <https://github.com/ncas-cms/xconv2>`_ package, but can be used anywhere you want
to re-use an SSH system login to access remote HDF5/NetCDF4 or pp/fields files from 
a Python client.

.. code-block:: python

   from p5rem import bootstrap_session

   with bootstrap_session(
       host="hpc",
       remote_setup="module load jaspy",
       remote_python="python",
       login_shell=True,
   ) as session:
       with session.open("data/temperature.nc") as f:
           tas = f["tas"]
           first_timestep = tas[0, :, :]   # NumPy array, fetched chunk-by-chunk

.. toctree::
   :maxdepth: 2
   :caption: User documentation

   getting-started
   user-guide

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Background

   design

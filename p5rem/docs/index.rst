p5rem
=====

*Remote HDF5/NetCDF chunk access and reduction over SSH.*

p5rem lets desktop tools read and compute over HDF5/NetCDF files on a remote
HPC system — with no custom server infrastructure, open ports, or host
administrator involvement.  A small pyfive-based server is bootstrapped over
a standard SSH connection and communicates with the client entirely through
stdin/stdout.  No TCP port forwarding is needed.

.. code-block:: python

   from p5rem import bootstrap_session

   with bootstrap_session(host="hpc", remote_python="conda run -n myenv python") as session:
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

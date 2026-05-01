Introduction
************

About PyActiveStorage
=====================

PyActiveStorage provides a Python client to local or remote “active” storage (aka “computational” storage), that is,
storage that instead of just receiving reads and returning blocks of data, can receive requests for, and return the
results of, computations on blocks of data, which are carried out by a server running or alongside the
storage. 

The computations supported by the active storage are currently limited to reductions (e.g. mean, min, max) over specified
axes, but the initial design, inspired by MPI, allows for more complex computations to be added in future, provided
they are are dimensiontal reductions and do not require the transfer of data from the client to the server and its storage.

This version supports reductions carried out by a remote `“Reductionist” <https://github.com/stackhpc/reductionist-rs>`_ 
server situationed alongside S3 or HTTPS
based data services (either in a separate system, or within the same webserver stacks), as well as reductions
on remote systems accessible via SSH which can run a small python server injected by this library (running
as the user with SSH access). The latter also supports fetching of data chunks on demand with
a similar API to that used for S3 and HTTPS (which is based on the fsspec library).

Earlier version supported a prototype interface to DDN Infinia(TM) active storage. This is no longer
supported as the functionality of this version is production ready for S3/HTTPS and SSH, we will add
support for other active storage servers in future releases.

Early results accessing global fields of 10km resolution data on the CEDA-JASMIN S3 object store show that time-series
of hemispheric means can be returned to remote users on low-bandwidth networks (and even in
different hemispheres) in times which are competitive with local users accessing POSIX data across a LAN.

We anticipate this technology will be game-changing for remote access to data, particularly where there is
insufficient local storage to make copies of data, or those with low-bandwidth networks – although anyone can benefit from
minimising network contention and reducing the carbon cost of moving data.

How it works
============

.. _fig_how_it_works:
.. figure:: /figures/how-it-works.png
   :align: center
   :width: 80%

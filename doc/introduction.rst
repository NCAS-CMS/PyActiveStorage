Introduction
************

About PyActiveStorage
=====================

PyActiveStorage provides a Python client to local or remote “active” storage (aka “computational” storage), that is,
storage that instead of just receiving reads and returning blocks of data, can receive requests for, and return the
results of, computations on blocks of data. We have implemented active storage which supports a limited set of reduction
computations. We have two implementations of active storage with which PyActiveStorage can interact:

* a prototype in DDN InfiniaTM
* a production ready `“Reductionist” <https://github.com/stackhpc/reductionist-rs>`_  middleware software which can be
  deployed to turn any S3 object store into active storage

Using PyActiveStorage can speed up workflows, especially if the data is remote, or the local area network is congested.
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

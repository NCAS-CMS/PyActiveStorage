We here explain how this package would be deployed across some simple architectures.

#### Example 1

A parallel file systems which has, for example, data striped across three object storage targets (`ost1` ... `ost3`), served by a server (`OSS`) and delivered by a client (kernel or fuse) on the client node. Applications read POSIX data via that kernel.

In this instance, we would expect the OSS to implement something with the semantics and interface of `_decode_chunk` and for us to implement a module in `Active.py` which is called when the file URI is recognised as this sort of POSIX. Our module would call a shimmy in the client kernel which would pass on a call to that OSS implementation.


```plantuml
package "hardware view 1" {
  left to right direction
    component storage {
        database ost1
        database ost2  
        database ost3 
        component OSS
        OSS -- ost1
        OSS -- ost2
        OSS -- ost3
    }
    
    component node {
        component application 
        component kernel
        application --> kernel
    }
    
    kernel --> OSS: LAN traffic.
    
}
```

#### Example 2

A storage system which implements parallel transfers direct to a client on the compute node. In this case the kernel (or fuse client) on the compute node receives data from the storage nodes directly.
There are two interesting variants of this: one where we control parallellism at the chunk level, and one where the storage system controls those transfers.  The first case we term "application parallelism" and this would gain no benefit unless there is also "storage parallelism". 

An example of application parallelism would be the use of asynchronous requests via S3. This would be delivered by some parallelism (which is yet to be) implemented in the routine `_from_storage`.

THe more interesting problem for now arises in where we have "storage parallelism". 

```plantuml
package "hardware view 2" {
    left to right direction
    component storage {
        database ost1
        database ost2  
        database ost3 
    }
    
    component node {
        component application 
        component kernel
        application --> kernel
    }
    kernel --> ost1: LAN
    kernel --> ost2: LAN
    kernel --> ost3: LAN
}
```

Where would we implement what? There is no benefit to be gained from implementing `_decode_chunks` in the kernel, as the entire chunk has already been served to the compute node, and no data movement has been avoided. Depending on the layout of chunks across OSTs, there may be no benefit in attempting active storage. It depends on how contiguous the data is in storage. 

If however, it is possble to break down the logic of `_from_storage` so that individual "part_chunks" which are contiguous are processed on the storage side of the LAN (in the ost or nearby) then meaningful performance is possible. The Python client in the end would simply see a list of partial products of the computational method which have come direct from the osts. It will not care whether those parts came from a different breakdown of storage than it anticpated in the chunking (though of course the storage will need to do the necessary mapping to generate the partial sums).

In the longer term, where we expect that we will have to pass a decompression method down through the `_decode_chunk` interface, it _will_ be necessary for the computational storage to respect the `_decode_chunk` interface server-side. This is of course what is required with S3, where we effectively need an S3 proxy to do the work. It might be that this is also required in some implementations of POSIX storage systems if they wish to implement computational storage OR they will need to somehow respect block contiguous placement in some way.


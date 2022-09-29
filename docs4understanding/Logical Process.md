
 1. We have an N dimensional data array in M `file-chunks` in an  HDF file, compressed with method `comp-method`.
 2. Kerchunk gives us the offset and size of each `file-chunk`.
 3. A reference file system (RFS) uses that information to mapp `zarr-files` one to one to the `file-chunks`.  In practice when using these files, when they are actually needed, zarr will read an entire file (ie file-chunk), decompress it, and load it into memory.

The zarr algorithm for slicing and dicing effectively does the following:

Consider $X(10,10,10)$ as 32-bit floats distributed in 5 HDF chunks, and user accessing  $X[a:b]$ :
1. Convert to slice notation using knowledge of the dimensionality of $X$, so we have, e.g.  $X[(i_1,i_2),(j_1,j_2),(k_1,k_2)]$  (e.g `x[1:3,2:4,5:6]` )
2. Identify which zarr files are needed to access the data. This data is unlikely to be contiguous in the file, so it is likely to represent a set of `sub-chunks` distributed across the `zarr-files` which in practice means, across the HDF chunks. 
 
 We can depict this as follows:

![[../meta/assets/0C5B5C0E-3562-49B0-9749-7648C41EBE1A.jpeg]]

  - the red blocks represent the necessary sub-chunks. These are distributed across the actual chunks in an un-even way, and there can be 0,1 or many sub-chunks requested from actual chunks.

What happens in zarr is that all the data for chunks 1,3 and 5 will be read from the file and loaded into memory, then the locations of the sub-chunks is used to populate a return array with the necessary information (details needed). The location of the subchunks in a buffer corresponding to the uncompressed chunks is known in the zarr machinery, so effectively there will be a set of $n_s$ slices $\mathrm{slice}_{c,i=1,n_s}$ which encode that information - but in practice what they really define is a set of offsets and sizes ($s_o,s_s$) within the buffer, so set($\mathrm{slices}_{c,i}$) $\equiv$ set($(s_{o,i},s_{s,i})$. In this example $n_s = 4$ and we would be doing three calls to (one per `file-chunk`) to get the information. 
 
What we need to happen is to implement something like the following:
 - `call read(o1,s1,comp-method,"float32",[(s_{o,1},s_{s,1)])`
 - `call read(o3,s3,comp-method,"float32",[(s_{o,2},s_{s,2),(s_{o,3},s_{s,3))])`
 - `call read(o5,s5,comp-method,"float32",[(s_{o,4},s_{s,4)])`
(obviously we need to implement a method to go from  $\mathrm{slices}_{c,i}$ to $s_{o,i},s_{s,i}$ ).

and have the active storage do  the following:
- load the bytes from `o1` to `o1+s1` and decompress them into buffer `Y` of 200 `float32` numbers using `comp-method`.  (We would first implement `comp-method = None`!)
- then extract and return a list of arrays `Y[s_o,s_i]`

which we then put into the right place in our return buffer.

When we have all that working, we can add methods to the mix!

---

See [[How Zarr to HDF via kerchunk currently works]]



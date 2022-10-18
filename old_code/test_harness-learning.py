
import unittest
import os
from activestorage.active_tools import make_an_array_instance_active
from activestorage.dummy_data import make_vanilla_ncdata
from netCDF4 import Dataset
import numpy as np
from numcodecs.compat import ensure_ndarray
from zarr.indexing import (
    OrthogonalIndexer,
    check_fields,
)

from activestorage import netcdf_to_zarr as nz

class Active:
    """ 
    Instantiates an interface to active storage which contains either zarr files
    or HDF5 (NetCDF4) files.

    This is Verson 1 which simply provides support for standard read operations, but done via
    explicit reads within this class rather than within the underlying format libraries.
    
    Version 2 will add methods for actual active storage.

    """
    def __init__(self,uri,*args,**kwargs):
        """ Instantiate in the same way as normal """
        # Assume NetCDF4 for now
        self.file = Dataset(uri)
        self._version = 1 
        self.uri = uri
        self.method = None
        self.zds = None

    def __getitem__(self, *args):
        """ 
        Provides support for a standard get item.
        """ 
        # In version one this is done by explicitly looping over each chunk in the file
        # and returning the requested slice ourselves. In version 2, we can pass this
        # through to the default method.
        if self.method is not None and self._version != 2:
            raise ValueError(f'Cannot apply active storage with this version ({self._version}) of the getitem method')
        if self._version == 0:
            return self.file.__getitem__(*args)
        elif self._version == 1:
            return self._via_kerchunk(*args)
        elif self._version  == 2:
            return self._via_kerchunk(*args)
        else:
            raise ValueError(f'Version {self._version} not supported')

    def method(self, method):
        """ Set the method for any future get items"""
        self.method = method

    def _get_active(self, method, *args):
        """ 
        *args defines a slice of data. This method loops over each of the chunks
        necessary to extract the parts of the slice, and asks the active storage 
        to apply the method to each part. It then applies the method to 
        the partial results and returns a value is if  method had been applied to
        an array returned via getitem.
        """
        raise NotImplementedError

    def _via_kerchunk(self, *args):
        """ 
        The objective is to use kerchunk to read the slices ourselves. 
        """
        print('#FIXME: Order of calls is hardcoded')
        if args == ('data',):
            if self.zds is None:
                ds = nz.load_netcdf_zarr_generic(self.uri, args[0])
                # The following is a hangove from exploration
                # and is needed if using the original doing it ourselves
                # self.zds = make_an_array_instance_active(ds)
                self.zds = ds
            return self
        else:
            return self.__doing_it_ourselves2(*args)

    def __vanilla_zarr(self, *args):
        return self.zds.__getitem__(*args)

    def __doing_it_ourselves(self,*args):
        """ This is the first version unpacking stuff ourselves directly.
        Doesn't quite get things in the right order, whether we read the 
        data or whether zarr does iet."""

        data_selection, chunk_info, chunk_coords = self.zds.get_orthogonal_selection(*args,
                                                 out=None, fields=None)

        chunks, chunk_sel, PCI= chunk_info[0]

        # get offsets and sizes from PCI
        offsets = []
        sizes = []
        for offset, size, _ in list(PCI):
            offsets.append(offset)
            sizes.append(size)

        # get chunks info from chunk store
        chunk_store = self.zds.chunk_store
        chunk_coords_formatted = []
        for ch_coord in chunk_coords:
            new_key = "data/" + ".".join([str(ch_coord[0]),
                                        str(ch_coord[1]),
                                        str(ch_coord[2])])
            chunk_coords_formatted.append(new_key)

        # decode bytes from chunks
        # this is vanilla zarr
        chunks_with_data = [self.zds._decode_chunk(chunk_store[k]) for k in chunk_coords_formatted]

        # this is going to the file ourselves directly
        chunks_our_way = [self._decode_chunk(k) for k in chunk_coords_formatted]

        print('Zarr',chunks_with_data)
        print('Us',chunks_our_way)        

        chunks_dict = {}
        for (i, k), f in zip(enumerate(chunk_coords_formatted), chunk_coords):
            flat_decoded = np.ndarray.flatten(self.zds._decode_chunk(chunk_store[k]))
            selection_in_chunk = []
            # NB: very important to remember that each start in "offsets" is to be
            # used for each chunk; it's not one start from "offsets" is per chunk
            for j, k in zip(offsets, sizes):
                partial_data = flat_decoded[j:j+k]
                selection_in_chunk.extend(partial_data)
            chunks_dict[f] = selection_in_chunk

        selection = []
        for _, v in chunks_dict.items():
            selection.extend(v)

        return np.array(selection)

    def __doing_it_ourselves2(self, *args):

        # conflating _get_orthogonal_selection ...

        # we probably want to get the results of needing ot use indexer and pass that down
        # as I suspect this might not be something to push down into a real active layer.
        indexer = OrthogonalIndexer(*args, self.zds)

        # ... with the start of as_get_selection
        out_dtype = self.zds._dtype
        out_shape = indexer.shape
        out = np.empty(out_shape, dtype=out_dtype, order=self.zds._order)

        for chunk_coords, chunk_selection, out_selection in indexer:
            self._process_chunk(chunk_coords,chunk_selection, out, out_selection,
                                    drop_axes=indexer.drop_axes) 
        
        if self.method is not None:
            return self.method(out)
        else:
            return out

    def _process_chunk(self, chunk_coords, chunk_selection, out, out_selection,
                       drop_axes=None):
        """Obtain part or whole of a chunk by taking binary data from storage and filling output array"""
        if self.zds._compressor:
            raise ValueError("No active support for compression as yet")
        if self.zds._filters:
            raise ValueError("No active support for filters as yet")
        
        key = "data/" + ".".join([str(c) for c in chunk_coords])
        chunk = self._decode_chunk(key)
        tmp = chunk[chunk_selection]
        if drop_axes:
            tmp = np.squeeze(tmp, axis=drop_axes)

        # store selected data in output
        out[out_selection] = tmp

    def _decode_chunk(self, key):
        """ We do our own read of chunks and decoding etc """
        
        # yes this next line is bordering on voodoo ... 
        myfsref = self.zds.chunk_store._mutable_mapping.fs.references
        rfile, offset, size = tuple(myfsref[key])
        #fIXME: for the moment, open the file every time ... we might want to do that, or not
        with open(rfile,'rb') as open_file:
            # get the data
            chunk = self._read_block(open_file, offset, size)
            # make it a numpy array of bytes
            chunk = ensure_ndarray(chunk)
            # convert to the appropriate data type
            chunk = chunk.view(self.zds._dtype)
            # sort out ordering and convert to the parent hyperslab dimensions
            chunk = chunk.reshape(-1, order='A')
            chunk = chunk.reshape(self.zds._chunks, order=self.zds._order)
        return chunk

    def _read_block(self, open_file, offset, size):
        """ Read <size> bytes from <open_file> at <offset>"""
        place = open_file.tell()
        open_file.seek(offset)
        data = open_file.read(size)
        open_file.seek(place)
        return data

    def close(self):
        self.file.close()

class TestActive(unittest.TestCase):
    """ 
    Test basic functionality
    """

    def setUp(self):
        """ 
        Ensure there is test data
        """
        self.testfile = 'test_bizarre.nc'
        if not os.path.exists(self.testfile):
            make_vanilla_ncdata(filename=self.testfile)
        
    def testRead0(self):
        """ 
        Test a normal read slicing the data an interesting way, using version 0 (native interface)
        """
        active = Active(self.testfile)
        active._version = 0
        var = active['data']
        d = var[0:2,4:6,7:9]
        nda = np.ndarray.flatten(d.data)
        assert np.array_equal(nda,np.array([740.,840.,750.,850.,741.,841.,751.,851.]))
        active.close()

    def testRead1(self):
        """ 
        Test a normal read slicing the data an interesting way, using version 1 (replicating native interface in our code)
        """
        active = Active(self.testfile)
        active._version = 1
        var = active['data']
        d = var[0:2,4:6,7:9]
        nda = np.ndarray.flatten(d)
        assert np.array_equal(nda,np.array([740.,840.,750.,850.,741.,841.,751.,851.]))
        active.close()

    def testActive(self):
        """ 
        Shows what we expect an active example test to achieve and provides "the right answer" 
        """
        active = Active(self.testfile)
        active._version = 0
        var = active['data']
        d = var[0:2,4:6,7:9]
        mean_result = np.mean(d)
        active.close()

        active = Active(self.testfile)
        active._version = 2
        active.method=np.mean
        var = active['data']
        result2 = var[0:2,4:6,7:9]
        self.assertEqual(mean_result, result2)
       

    def test_zarr_hijack(self):
        """ 
        Test the hijacking of Zarr. 
        """
        data_file = self.testfile
        varname = "test_bizarre"
        selection = (slice(0, 2, 1), slice(4, 6, 1), slice(7, 9, 1))

        # get slicing info
        (ds, master_chunks, chunk_selection,
         offsets, sizes, selected_chunks, chunks_data_dict) = \
            nz.slice_offset_size(data_file, varname, selection)

        # sanity checks
        assert master_chunks == (3, 3, 1)
        assert len(offsets) == 2
        assert len(sizes) == len(offsets)
        assert offsets == [1, 4]  # these are numbers of elements
        # and are found in each chunk ie 1st and 4th element start data in each chunk
        assert sizes == [2, 2]  # these are numbers of elements
        # and are found in each chunk ie 1st+2 and 4th+2 span selected data in each chunk
        assert selected_chunks == [(0, 1, 7), (0, 1, 8)]  # chunks coords of the chunks containing selected data
        assert selected_chunks == list(chunks_data_dict.keys())
        expected_selection = np.sort([740.,840.,750.,850.,741.,841.,751.,851.])
        selection = []
        for _, v in chunks_data_dict.items():
            selection.extend(v)
        print(f"UNITTEST: expected data selection: {expected_selection}")
        print(f"UNITTEST: obtained data selection: {np.array(selection)}")
        assert np.array_equal(np.array(expected_selection), np.sort(selection))


if __name__=="__main__":
    unittest.main()

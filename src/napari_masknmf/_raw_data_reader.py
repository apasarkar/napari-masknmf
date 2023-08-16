import numpy as np
import tifffile
from numpy.typing import DTypeLike



from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)


if TYPE_CHECKING:
    from enum import Enum

    from numpy.typing import DTypeLike

    # https://github.com/python/typing/issues/684#issuecomment-548203158
    class ellipsis(Enum):
        Ellipsis = "..."

    Ellipsis = ellipsis.Ellipsis  # noqa: A001
else:
    ellipsis = type(Ellipsis)


Index = Union[int, slice, ellipsis]


from napari.layers._data_protocols import LayerDataProtocol
from napari.layers.image._image_utils import guess_multiscale


class MultipageTiffVideo(LayerDataProtocol):
        
    def __init__(self, filepath):
        self.filepath = filepath
        self._shape = self._compute_shape(filepath)
        self._dtype = np.float32

    def _compute_shape(self, filename):
        with tifffile.TiffFile(self.filepath) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
                x, y = page.shape
        return (num_frames, x, y)
    
    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""
        return self._dtype
        
    @property
    def ndim(self) -> int:
        """
        Returns number of dimensions of data
        """
        return 2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        
        return self._shape
    
    def __getitem__(
        self, key #: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]. NOTE: Slicing anything aside from pages will be inefficient here. Tifffiles just don't seem to have the support for that as of now."""
        if key[1] == slice(None, None, None) and key[2] == slice(None, None, None):
            return tifffile.imread(self.filepath, key=[key[0]]).astype(self.dtype)
        else:
            page_slice, row_slice, col_slice = key
            pages = tifffile.TiffFile(self.filepath).pages
            if isinstance(page_slice, slice):
                start_page = page_slice.start or 0
                stop_page = min(len(pages), page_slice.stop or len(pages))
            else:
                start_page = stop_page = page_slice

            # Read data for the specified slice
            data = []
            for i in range(start_page, stop_page):
                page = pages[i]
                page_data = page.asarray()
                data.append(page_data[row_slice, col_slice])

            return np.array(data, dtype=self.dtype)
    

def tiff_frame_generate(path):
    """
    Generates a frame from a tiff file
    """
    tiff_object = MultipageTiffVideo(path)

    add_kwargs = {}
    layer_type = "image"  # optional, default is "image"
    my_output = [(tiff_object,add_kwargs,layer_type)]
    return my_output



def napari_get_tiff_reader(path):
    """
    A reader contribution for PMD objects
    
    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
      
    Returns
    -------
    function or None
        If the path is a recognized format, returns a PMD_frame_generator function. This
        function will generate subsequence frames of the PMD object
    """
    
    if isinstance(path, list):
        path = path[0]
        
    if not path.endswith(".tiff") and not path.endswith(".tif"):
        return None
    
    ##Here is some code to verify that the .npz file contains the things we want it to contain 
    
    return tiff_frame_generate
   

"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
from numpy.typing import DTypeLike
import scipy
import scipy.sparse

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

def Factorized_PMD_video(LayerDataProtocol):
    
    def __init__(self, filepath):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.order = data['fov_order']
        self.d1, self.d2 = data['fov_shape']
        print("the shape of d1 and d2 is {} and {}".format(self.d1, self.d2))
        self.U_sparse = scipy.sparse.csr_matrix(
        (data['U_data'], data['U_indices'], data['U_indptr']),
        shape=data['U_shape'])
        R = data['R']
        s = data['s']
        V = data['Vt']
        self.V = (R * s[None, :]).dot(V) #Fewer computations
        self.T = self.V.shape[1]
    
    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""
        return self.V.dtype
        

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        
        return (self.d1, self.d2, self.T)
    
    def __getitem__(
        self, key #: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        print("key is {}".format(key))
        output = (self.U_sparse.dot(self.V[:, key])).reshape((self.d1, self.d2), order=self.order)
        print("the type of output is {}".format(type(output)))
        print("the shape of output is {}".format(output.shape))
        return output

def napari_get_PMD_reader(path):
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
        
    if not path.endswith(".npz"):
        return None
    
    ##Here is some code to verify that the .npz file contains the things we want it to contain 
    
    datafile = np.load(path, allow_pickle=True)
    
    key_list = ['U_data', 'U_indices', 'U_indptr', 'U_shape', 'fov_order', 'fov_shape', 'R', 's', 'Vt']
    for k in key_list:
        if k not in datafile.keys():
            return None
    
    return PMD_frame_generate
    
def PMD_frame_generate(path):
    """
    Generates a frame from a PMD object
    """
    pmd_object = Factorized_PMD_video(path)
    print("the object was created successfully")
    return [(pmd_object,)]



def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".npy"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    # load all files into array
    arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    data = np.squeeze(np.stack(arrays))

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

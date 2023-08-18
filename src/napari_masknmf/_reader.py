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
import random

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

class ColorfulACMovie(LayerDataProtocol):  
    def __init__(self, filepath, seed=999):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.order = data['fov_order'].item()
        self.d1, self.d2 = data['fov_shape']
        self.a_dense = data['a'].reshape((self.d1, self.d2, -1), order=self.order)
        self.a = scipy.sparse.csr_matrix(data['a']) #For now, data['a'] is a dense array. If API changes, adjust this
        
        self.c = data['c'].T
        self.T = self.c.shape[1]
        self.var_img = data['noise_var_img']
        self.row_indices = np.arange(self.d1*self.d2).reshape((self.d1, self.d2), order=self.order)
    
        rgbrange = [80, 255]
        channels = 3
        np.random.seed(seed) #seed for random color generation (to keep it consistent if desired)
        random_values = np.array([random.randint(rgbrange[0],rgbrange[1]) for i in range(channels*self.a_dense.shape[2])]).reshape((self.a_dense.shape[2], channels))
        sum_random_values = np.sum(random_values, axis = 1, keepdims = True)
        random_color_norm = random_values / sum_random_values
        
        c_color = np.zeros((self.c.shape[0], 3, self.c.shape[1]))
        for i in range(3):
            c_color[:, i, :] = random_color_norm[:, [i]] * self.c

        self.c_color = c_color
    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""
        return np.float32 ##Maybe change this...


    @property
    def ndim(self) -> int:
        """
        Returns number of dimensions of data
        """
        return 2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        return (self.T, self.d1, self.d2, 3)

    def __getitem__(
        self, key #: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]. Compute Proj_{UR} W(UV - b) + b"""
        if key[1] == slice(None, None, None) and key[2] == slice(None, None, None): #Only slicing time points in this case -- the common use case
            c_color_crop = self.c_color[:, :, key[0]]
            if type(key[0]) is not slice:
                output = (self.a.dot(c_color_crop)).reshape((self.d1, self.d2, 3), order=self.order)
            else:
                output = np.tensordot(self.a_dense, c_color_crop, axes = (len(c_color_crop.shape)-1, 0))
                output = output.swapaxes(-1, -2)
        else: #In this case we are taking slices across time
            c_color_crop = self.c_color[:, :, key[0]]
            slice_a = self.a_dense[(key[1], key[2], slice(None, None, None))]
            output = np.tensordot(self.a_dense[(key[1], key[2], slice(None, None, None))], c_color_crop, axes=(len(slice_a.shape)-1, 0))
            output = output.swapaxes(-1, -2)
            '''The output here will have as its last two dimensions (3, T) assuming we did a slice within the first two dimensions. To make this play
            nicely with 
            '''
        return output.squeeze().astype(self.dtype)


class BackgroundPMDMovie(LayerDataProtocol):  
    def __init__(self, filepath):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.order = data['fov_order'].item()
        self.d1, self.d2 = data['fov_shape']
        self.U_sparse = scipy.sparse.csr_matrix(
        (data['U_data'], data['U_indices'], data['U_indptr']),
        shape=data['U_shape'])
        self.R = data['R']
        s = data['s']
        V = data['Vt']
        
        self.V = (self.R * s[None, :]).dot(V) #Fewer computations
        self.b = data['b']
        self.W = data['W'].item().tocsr()
        self.T = self.V.shape[1]
        self.row_indices = np.arange(self.d1*self.d2).reshape((self.d1, self.d2), order=self.order)
        self.var_img = data['noise_var_img']

    
        
    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""
        return np.float32


    @property
    def ndim(self) -> int:
        """
        Returns number of dimensions of data
        """
        return 2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        return (self.T, self.d1, self.d2)

    def __getitem__(
        self, key #: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]. Compute Proj_{UR} W(UV - b) + b"""
        if key[1] == slice(None, None, None) and key[2] == slice(None, None, None): #Only slicing rows
            W_used = self.W
            left_term = self.U_sparse.dot(self.V[:, key[0]])
            if len(left_term.shape) == 1:
                left_term = left_term[:, None] - self.b
            else:
                left_term -= self.b
            implied_fov_shape = (self.d1, self.d2)
            output = (W_used.dot(left_term) + self.b)#self.var_img
            output = self.U_sparse.T.dot(output)
            output = self.R.T.dot(output)
            output = self.R.dot(output)
            output = self.U_sparse.dot(output)
            output = output.reshape(implied_fov_shape + (-1,), order=self.order) * self.var_img[:, :, None]
        else: #In this case we are taking slices across time
            used_rows = self.row_indices[key[1:3]]
            implied_fov_shape = used_rows.shape
            U_used = self.U_sparse[used_rows.reshape((-1,), order=self.order)]
            UR = U_used.dot(self.R)
            URRt = UR.dot(self.R.T)
            URRtUt = (self.U_sparse.dot(URRt.T)).T
            URRtUtW = (self.W.T.dot(URRtUt.T)).T
            URRtUtWU = (self.U_sparse.T.dot(URRtUtW.T)).T
            URRtUtWUV = URRtUtWU.dot(self.V)
            URRtUtWb = URRtUtW.dot(self.b)
            output = URRtUtWUV - URRtUtWb + self.b[used_rows.reshape((-1,), order=self.order), :]
            output = output.reshape(implied_fov_shape + (-1,), order=self.order)
            output *= self.var_img[(key[1], key[2], None)]

        return output.squeeze().astype(self.dtype)


class ResidualPMDMovie(LayerDataProtocol):  
    def __init__(self, PMDMovie, BackgroundMovie, ACMovie):
        self.PMDMovie = PMDMovie
        self.BackgroundMovie = BackgroundMovie
        self.ACMovie = ACMovie
        self.mean_img = self.PMDMovie.mean_img
        self.T, self.d1, self.d2 = self.PMDMovie.shape
    
        
    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""
        return np.float32


    @property
    def ndim(self) -> int:
        """
        Returns number of dimensions of data
        """
        return 2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        return (self.T, self.d1, self.d2)

    def __getitem__(
        self, key #: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]. Compute Proj_{UR} W(UV - b) + b"""
        output = self.PMDMovie[key] - self.ACMovie[key] - self.BackgroundMovie[key]
        if key[1] == slice(None, None, None) and key[2] == slice(None, None, None): #Only slicing rows
            if type(key[0]) is slice:
                output -= self.mean_img[:, :, None]
            else:
                output -= self.mean_img
        else: #In this case we are taking slices across time
            output -= self.mean_img[(key[1], key[2], None)]
        return output.squeeze().astype(self.dtype)


class SignalPMDMovie(LayerDataProtocol):
    
    def __init__(self, filepath):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.order = data['fov_order'].item()
        self.d1, self.d2 = data['fov_shape']
        self.a = scipy.sparse.csr_matrix(data['a']) #For now, data['a'] is a dense array. If API changes, adjust this
        self.c = data['c'].T
        self.T = self.c.shape[1]
        self.mean_img = data['mean_img']
        self.var_img = data['noise_var_img']
        self.row_indices = np.arange(self.d1*self.d2).reshape((self.d1, self.d2), order=self.order)
        
    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""
        return np.float32


    @property
    def ndim(self) -> int:
        """
        Returns number of dimensions of data
        """
        return 2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        return (self.T, self.d1, self.d2)

    def __getitem__(
        self, key #: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if key[1] == slice(None, None, None) and key[2] == slice(None, None, None): #Only slicing rows
            a_used = self.a
            implied_fov_shape = (self.d1, self.d2)
        else:
            used_rows = self.row_indices[key[1:3]]
            implied_fov_shape = used_rows.shape
            a_used = self.a[used_rows.reshape((-1,), order=self.order)]

        output = a_used.dot(self.c[:, key[0]]).reshape(implied_fov_shape + (-1,), order=self.order)
        output = output * self.var_img[(key[1], key[2], None)]
        return output.squeeze().astype(self.dtype)



class Factorized_PMD_video(LayerDataProtocol):
    
    def __init__(self, filepath):
        self.filepath = filepath
        data = np.load(filepath, allow_pickle=True)
        self.order = data['fov_order'].item()
        self.d1, self.d2 = data['fov_shape']
        self.U_sparse = scipy.sparse.csr_matrix(
        (data['U_data'], data['U_indices'], data['U_indptr']),
        shape=data['U_shape'])
        R = data['R']
        s = data['s']
        V = data['Vt']
        self.V = (R * s[None, :]).dot(V) #Fewer computations
        self.T = self.V.shape[1]
        self.mean_img = data['mean_img']
        self.var_img = data['noise_var_img']
        self.row_indices = np.arange(self.d1*self.d2).reshape((self.d1, self.d2), order=self.order)
    
    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array elements."""
        return np.float32
        
    @property
    def ndim(self) -> int:
        """
        Returns number of dimensions of data
        """
        return 2

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array dimensions."""
        return (self.T, self.d1, self.d2)
    
    def __getitem__(
        self, key #: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if key[1] == slice(None, None, None) and key[2] == slice(None, None, None): #Only slicing rows
            U_used = self.U_sparse
            implied_fov_shape = (self.d1, self.d2)
        else:
            used_rows = self.row_indices[key[1:3]]
            implied_fov_shape = used_rows.shape
            U_used = self.U_sparse[used_rows.reshape((-1,), order=self.order)]

        output = U_used.dot(self.V[:, key[0]]).reshape(implied_fov_shape + (-1,), order=self.order)
        output = output * self.var_img[(key[1], key[2], None)] + self.mean_img[(key[1], key[2], None)]
        return output.squeeze().astype(self.dtype)

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
    keyset = np.load(path, allow_pickle=True).keys()
    
    layer_list = []
    if 'U_data' in keyset: #Rough check to verify that there is PMD data in the file:
        pmd_object = Factorized_PMD_video(path)
        add_kwargs = {'name':'PMD'}
        layer_type="image"
        layer_list.append((pmd_object, add_kwargs, layer_type))
        print("successfully added PMD")
        
    if 'a' in keyset:
        grayscale_signal_object = SignalPMDMovie(path)
        add_kwargs = {'name':'Signals'}
        layer_type="image"
        layer_list.append((grayscale_signal_object, add_kwargs, layer_type))
        print("successsfully added AC")
    
    if 'a' in keyset:
        color_movie = ColorfulACMovie(path)
        add_kwargs = {'name':'Colored Signals','rgb':True}
        layer_type="image"
        layer_list.append((color_movie, add_kwargs, layer_type))
        print("Successfully added colorful movie")
    
    if 'W' in keyset:
        background_movie =  BackgroundPMDMovie(path)
        add_kwargs = {'name':'Background'} 
        layer_type = "image"
        layer_list.append((background_movie, add_kwargs, layer_type))
        print("Successfully added background movie")
    
    if 'W' in keyset and 'a' in keyset:
        residual_movie = ResidualPMDMovie(pmd_object, grayscale_signal_object, background_movie)
        add_kwargs = {'name':'Residual'}
        layer_type = "image"
        layer_list.append((residual_movie, add_kwargs, layer_type))
        print("Successfully added residual movie")
    
    return layer_list

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

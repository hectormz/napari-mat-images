"""
This module is a plugin to read images from .mat files in napari
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import h5py
import numpy as np
import scipy.io as sio
from pluggy import HookimplMarker

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]

napari_hook_implementation = HookimplMarker("napari")
MAT_EXTENSIONS = '.mat'


@napari_hook_implementation
def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """A basic implementation of the napari_get_reader hook specification."""
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]
    if not path.endswith(MAT_EXTENSIONS):
        # if we know we cannot read the file, we immediately return None.
        return None
    # otherwise we return the *function* that can read ``path``.
    return reader_function


def load_mat_vars(file_path: str) -> Dict:
    """Load image variables from .mat file into dictionary

    Args:
        file_path (str): path to .mat file

    Returns:
        Dict: dictionary of image variables. Empty if file contains no images
    """
    try:
        # Load variable details before loading
        mat_vars = sio.whosmat(file_path)
        # Check if each variable is image based on shape
        is_image_list = [shape_is_image(var[1]) for var in mat_vars]
        var_list = [var[0] for var in mat_vars]
        # Filter list of variables whether they are images
        var_list = [i for (i, v) in zip(var_list, is_image_list) if v]
        if len(var_list) > 0:
            mat_dict = sio.loadmat(
                file_path, variable_names=var_list, squeeze_me=True
            )
            for var in list(mat_dict.keys()):
                if not hasattr(mat_dict[var], 'shape'):
                    del mat_dict[var]
        else:
            mat_dict = {}
    except NotImplementedError:
        mat_file = h5py.File(file_path, mode='r')
        var_list = list(mat_file.keys())
        # discard #refs# entry
        try:
            var_list.remove("#refs#")
        except ValueError:
            pass
        is_image_list = [
            shape_is_image(mat_file[var].shape) for var in var_list
        ]
        # Filter list of variables whether they are images
        var_list = [i for (i, v) in zip(var_list, is_image_list) if v]
        mat_dict = {}
        for var in var_list:
            array_size = mat_file[var].size
            chunk_size = mat_file[var].chunks
            chunk_size = update_chunk_size(array_size, chunk_size)
            array = da.from_array(mat_file[var], chunks=chunk_size).squeeze()
            # .mat are saved in reverse order
            array = rearrange_da_dims(array)
            mat_dict[var] = array
    return mat_dict


def reader_function(path: PathLike) -> List[LayerData]:
    """Take a path or list of paths and return a list of LayerData tuples."""
    paths = [path] if isinstance(path, str) else path

    # Generate list to hold potential images from each path provided
    data_list = [None for __ in range(len(paths))]
    for i, _path in enumerate(paths):
        mat_dict = load_mat_vars(_path)
        if not mat_dict:
            continue
        var_list = list(mat_dict.keys())
        data = [None for __ in var_list]
        for j, var in enumerate(var_list):
            array = mat_dict[var]
            # optional kwargs for the corresponding viewer.add_* method
            meta = {"name": var}
            if len(array.shape) == 3 or len(array.shape) == 2:
                meta["contrast_limits"] = array_contrast_limits(array)
            elif len(array.shape) == 4:
                meta["channel_axis"] = 3
                # Set contrast min/max for each channel
                num_channels = array.shape[3]
                contrast_limits = [None for __ in range(num_channels)]
                for chann_index in range(num_channels):
                    contrast_limits[chann_index] = array_contrast_limits(
                        array[:, :, :, chann_index]
                    )
                meta["contrast_limits"] = contrast_limits
            if isinstance(array, da.Array):
                meta["is_pyramid"] = False
            data[j] = (prep_array(array), meta)
        data_list[i] = data

    # Return None if no .mat files could be read
    if all(value is None for value in data_list):
        return None

    # Flatten potential list of lists
    data_list = [item for sublist in data_list for item in sublist]

    return data_list


def shape_is_image(shape: Sequence, min_size: int = 20) -> bool:
    """Checks if shape of array provided is at least 2D

    Args:
        shape (Sequence): shape of array to check

    Returns:
        bool : Whether shape belongs to at least 2D image
    """
    dims = np.sum(np.array(shape) > min_size)
    return dims >= 2


def prep_array(array: np.ndarray) -> np.ndarray:
    """Correct images after loading to match Python

    Args:
        array (np.ndarray): array of at least two dimensions

    Returns:
        np.ndarray: Corrected array
    """
    # Boolean/logical arrays from Matlab are read as uint8
    if array.dtype == "uint8":
        if array.max() == 1:
            array = array.astype("bool")

    # Rearrange dimensions if 3D or higher
    array = rearrange_dims(array)
    return array


def rearrange_dims(array: np.ndarray) -> np.ndarray:
    """If image is more than 2D, move third dimension to first

    Args:
        array (np.ndarray): Multidimensional array

    Returns:
        np.ndarray: Array with rearranged axes
    """
    if len(array.shape) > 2:
        # If third dimension is longer than first two, move to first position
        if np.all(array.shape[2] > np.array(array.shape[0:2])):
            array = np.moveaxis(array, 2, 0)
    return array


def rearrange_da_dims(array: da.Array) -> da.Array:
    """Flip dask array dims from HDF5 .mat files & move slices dim to 0th pos.

    Args:
        array (da.Array): 3-dimensional or more dask array

    Returns:
        da.Array: array with rearranged dimensions
    """
    array_shape = array.shape
    # Current dimension order
    dims = np.arange(len(array_shape))
    # Flip dims as if array dims were flipped to recover orig. saved dimensions
    dims_flipped = np.flip(dims)
    # breakpoint()
    if len(array_shape) > 2:
        # Flip array shape to recover original saved shape
        array_shape_flipped = np.flip(array_shape)
        # Find largest dimension (slices of stack) in flipped array shape
        slices_flipped_ind = np.argmax(array_shape_flipped)
        # Get slices dimension
        slices_dim = dims_flipped[slices_flipped_ind]
        # Remove slices dimensions from dims_flipped
        dims_flipped = np.delete(dims_flipped, slices_flipped_ind)
        # Insert slices dimensions to first dimension of dims_flipped
        dims_flipped = np.insert(dims_flipped, 0, slices_dim)
        # Determine which dimensions are no longer in agreement
        move_positions = dims != dims_flipped
        # If any dimensions need to be rearranged, move them
        if np.any(move_positions):
            array = da.moveaxis(
                array,
                source=dims_flipped[move_positions],
                destination=dims[move_positions],
            )
    elif len(array_shape) == 2:
        array = da.moveaxis(array, dims, dims_flipped)
    return array


def array_contrast_limits(array, axis=0, num_samples=100) -> List[float]:
    """Determine min/max of numpy/dask arrays along axis if n-dimensional

    Args:
        dask_array (Union[np.ndarray, dask.array]): n-dimensional array
        axis (int): Axis along n-dimensional array to sample min/max
        num_samples (int): Number of slices to sample from if large array.

    Returns:
        List[float]: min/max of array
    """
    if not isinstance(array, da.Array) and not isinstance(array, np.ndarray):
        raise TypeError("dask/numpy array expected")
    if len(array.shape) > 2:
        if num_samples is None:
            num_samples = array.shape[axis]
        num_samples = min(num_samples, array.shape[axis])
        random_samples = np.random.choice(
            array.shape[axis], num_samples, replace=False
        )
        # Sort random samples for dask slicing efficiency
        random_samples = np.sort(random_samples)
        # If unsigned int, use 0 as lower bound
        if np.issubdtype(array.dtype, np.unsignedinteger):
            contrast_min = 0
        else:
            contrast_min = array[random_samples].min()
            if isinstance(array, da.Array):
                contrast_min = contrast_min.compute()
        contrast_max = array[random_samples].max()
        if isinstance(array, da.Array):
            contrast_max = contrast_max.compute()
    elif len(array.shape) == 2:
        if num_samples is None:
            num_samples = array.size
        num_samples = min(num_samples, array.size)
        row_ind = np.random.randint(0, array.shape[0], num_samples)
        col_ind = np.random.randint(0, array.shape[1], num_samples)
        if isinstance(array, da.Array):
            contrast_min = array.vindex[row_ind, col_ind].min().compute()
            contrast_max = array.vindex[row_ind, col_ind].max().compute()
        else:
            contrast_min = array[row_ind, col_ind].min()
            contrast_max = array[row_ind, col_ind].max()
    else:
        raise ValueError("Array of dimensions >= 2 required.")

    return [contrast_min, contrast_max]


def update_chunk_size(array_size: Sequence, chunk_size: Sequence) -> List:
    """Determines new chunk size when loading dask array.
        Potentially increases slice axis chunk size to 10 if 1.
        This makes loading array faster for user.

    Args:
        array_size (Sequence): array size of dask array
        chunk_size (Sequence): original chunk size of dask array

    Returns:
        List: Updated (or original) chunk size
    """
    chunk_size = list(chunk_size)
    slice_index = np.argmax(array_size)
    if chunk_size[slice_index] == 1:
        chunk_size[slice_index] = 10
    return chunk_size

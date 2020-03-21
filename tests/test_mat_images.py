# -*- coding: utf-8 -*-

from tempfile import NamedTemporaryFile

import dask.array as da
import hdf5storage
import numpy as np
import pytest
import scipy.io as sio

from napari_mat_images import (
    array_contrast_limits,
    napari_get_reader,
    prep_array,
    rearrange_dims,
    shape_is_image,
)


def test_reader():
    with NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        out_data = np.random.rand(25, 25)
        sio.savemat(tmp.name, {"array": out_data})
        reader = napari_get_reader(tmp.name)
        in_data = reader(tmp.name)
        assert np.allclose(out_data, in_data[0][0])


def test_reader_channel_axis():
    with NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        out_data = np.stack(
            (
                np.random.randint(0, 8, (27, 25, 30), dtype='uint8'),
                np.random.randint(25, 29, (27, 25, 30), dtype='uint8'),
                np.random.randint(240, 245, (27, 25, 30), dtype='uint8'),
            ),
            axis=3,
        )
        sio.savemat(tmp.name, {"array": out_data})
        reader = napari_get_reader(tmp.name)
        in_data = reader(tmp.name)
        assert in_data[0][0].shape == (30, 27, 25, 3)
        assert in_data[0][1]["channel_axis"] == 3
    assert in_data[0][1]["contrast_limits"] == [[0, 7], [0, 28], [0, 244]]


def test_reader_int16():
    with NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        out_data = (np.random.randint(-84, 8, (27, 25, 30, 3), dtype='int16'),)

        sio.savemat(tmp.name, {"array": out_data})
        reader = napari_get_reader(tmp.name)
        in_data = reader(tmp.name)
        assert in_data[0][0].shape == (30, 27, 25, 3)
        assert in_data[0][1]["channel_axis"] == 3
    assert in_data[0][1]["contrast_limits"][0] == [-84, 7]


def test_reader_hdf5(tmp_path):
    out_data = np.stack(
        (
            np.random.randint(0, 10, (27, 25, 30), dtype='uint8'),
            np.random.randint(25, 29, (27, 25, 30), dtype='uint8'),
            np.random.randint(240, 255, (27, 25, 30), dtype='uint8'),
            np.random.randint(30, 71, (27, 25, 30), dtype='uint8'),
        ),
        axis=3,
    )
    mdict = {}
    mdict[u'array'] = out_data
    tmp = str(tmp_path / "temp.mat")
    hdf5storage.savemat(tmp, mdict, format='7.3')
    reader = napari_get_reader(tmp)
    in_data = reader(tmp)
    assert in_data[0][0].shape == (30, 27, 25, 4)
    assert in_data[0][1]["channel_axis"] == 3
    assert in_data[0][1]["contrast_limits"] == [
        [0, 9],
        [0, 28],
        [0, 254],
        [0, 70],
    ]


def test_reader_hdf5_3d(tmp_path):
    out_data = np.random.randint(0, 10, (27, 25, 31), dtype='uint8')

    mdict = {}
    mdict[u'array'] = out_data
    tmp = str(tmp_path / "temp.mat")
    hdf5storage.savemat(tmp, mdict, format='7.3')
    reader = napari_get_reader(tmp)
    in_data = reader(tmp)
    assert in_data[0][0].shape == (31, 27, 25)
    assert in_data[0][1]["contrast_limits"] == [0, 9]


def test_reader_hdf5_2d(tmp_path):
    out_data = np.random.randint(0, 10, (270, 350), dtype='uint8')

    mdict = {}
    mdict[u'array'] = out_data
    tmp = str(tmp_path / "temp.mat")
    hdf5storage.savemat(tmp, mdict, format='7.3')
    reader = napari_get_reader(tmp)
    in_data = reader(tmp)
    assert in_data[0][0].shape == (270, 350)
    assert in_data[0][1]["contrast_limits"] == [0, 9]


def test_reader_hdf5_1d(tmp_path):
    out_data = np.random.randint(0, 10, 150, dtype='uint8')

    mdict = {}
    mdict[u'array'] = out_data
    tmp = str(tmp_path / "temp.mat")
    hdf5storage.savemat(tmp, mdict, format='7.3')
    reader = napari_get_reader(tmp)
    assert reader(tmp) is None


def test_reader_no_images():
    with NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        out_data = "test string"
        sio.savemat(tmp.name, {"string_value": out_data})
        reader = napari_get_reader(tmp.name)
        in_data = reader(tmp.name)
        assert in_data is None


def test_get_reader_hit():
    reader = napari_get_reader('fake.mat')
    assert reader is not None
    assert callable(reader)


def test_get_reader_with_list():
    # a better test here would use real data
    reader = napari_get_reader(['fake.mat'])
    assert reader is not None
    assert callable(reader)


def test_get_reader_pass():
    reader = napari_get_reader('fake.file')
    assert reader is None


def test_is_image():
    shape = [300, 200, 8000]
    assert shape_is_image(shape)

    shape = [200, 200, 1]
    assert shape_is_image(shape)

    shape = [200, 1, 1]
    assert ~shape_is_image(shape)


def test_rearrange_dims():
    array = np.zeros((10, 10, 20))
    array_updated = rearrange_dims(array)
    assert array_updated.shape == (20, 10, 10)

    array = np.zeros((20, 10, 10))
    array_updated = rearrange_dims(array)
    assert array_updated.shape == (20, 10, 10)

    array = np.zeros((10, 10, 20, 4))
    array_updated = rearrange_dims(array)
    assert array_updated.shape == (20, 10, 10, 4)


def test_prep_array_uint8():
    array = np.random.randint(0, 255, (20, 20), dtype='uint8')
    array_prepped = prep_array(array)
    assert array_prepped.dtype == "uint8"


def test_prep_array_bool():
    array = np.zeros((20, 20), dtype='uint8')
    array[0, 0] = 1
    array_prepped = prep_array(array)
    assert array_prepped.dtype == "bool"


def test_dask_contrast_limits_all():
    array = da.random.randint(0, 2, (1000, 10, 15))
    contrast_limits = array_contrast_limits(array, axis=0, num_samples=None)
    assert contrast_limits == [0, 1]


def test_dask_contrast_limits():
    array = da.random.randint(0, 2, (1000, 10, 15))
    contrast_limits = array_contrast_limits(array, axis=0)
    assert contrast_limits == [0, 1]


def test_dask_contrast_limits_2d():
    array = da.random.randint(6, 8, (10, 15))
    contrast_limits = array_contrast_limits(array)
    assert contrast_limits == [6, 7]


def test_dask_contrast_limits_2d_all():
    array = da.random.randint(6, 8, (10, 15))
    contrast_limits = array_contrast_limits(array, axis=0, num_samples=None)
    assert contrast_limits == [6, 7]


def test_dask_contrast_limits_1d():
    array = da.random.randint(0, 2, 1000)
    with pytest.raises(ValueError):
        array_contrast_limits(array)


def test_dask_contrast_limits_int():
    with pytest.raises(TypeError):
        array_contrast_limits(1)

# -*- coding: utf-8 -*-

from tempfile import NamedTemporaryFile

import numpy as np
import scipy.io as sio

from napari_mat_images import (
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
        out_data = np.random.rand(25, 25, 30, 3)
        sio.savemat(tmp.name, {"array": out_data})
        reader = napari_get_reader(tmp.name)
        in_data = reader(tmp.name)
        assert in_data[0][0].shape == (30, 25, 25, 3)
        assert in_data[0][1]["channel_axis"] == 3


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

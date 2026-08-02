"""Test deprecation and future warnings.

"""
import pytest

import numpy as np
from numpy.ma.core import MaskedArrayFutureWarning
from numpy.ma.testutils import assert_equal


class TestArgsort:
    """ gh-8701: default axis is -1, matching ndarray.argsort """
    def _test_base(self, argsort, cls):
        arr_0d = np.array(1).view(cls)
        argsort(arr_0d)

        arr_1d = np.array([1, 2, 3]).view(cls)
        argsort(arr_1d)

        # Default axis is -1 (same as np.argsort); flattening requires axis=None
        arr_2d = np.array([[1, 2], [3, 4]]).view(cls)
        result = argsort(arr_2d)
        assert_equal(result, argsort(arr_2d, axis=-1))
        assert_equal(result.shape, arr_2d.shape)

        # Explicit None still flattens
        flat = argsort(arr_2d, axis=None)
        assert_equal(flat.shape, (arr_2d.size,))

    def test_function_ndarray(self):
        return self._test_base(np.ma.argsort, np.ndarray)

    def test_function_maskedarray(self):
        return self._test_base(np.ma.argsort, np.ma.MaskedArray)

    def test_method(self):
        return self._test_base(np.ma.MaskedArray.argsort, np.ma.MaskedArray)

    def test_matches_ndarray_default(self):
        a = np.ma.array([[3, 1], [4, 2]], mask=[[0, 0], [0, 1]])
        # Default axis matches explicit -1 and preserves shape (unlike old None)
        assert_equal(a.argsort(), a.argsort(axis=-1))
        assert_equal(np.ma.argsort(a), np.ma.argsort(a, axis=-1))
        assert_equal(a.argsort().shape, a.shape)
        # Unmasked data: ma.argsort agrees with np.argsort on the underlying array
        b = np.ma.array([[3, 1], [4, 2]])
        assert_equal(b.argsort(), np.argsort(np.asarray(b), axis=-1))


class TestMinimumMaximum:

    def test_axis_default(self):
        # NumPy 1.13, 2017-05-06

        data1d = np.ma.arange(6)
        data2d = data1d.reshape(2, 3)

        ma_min = np.ma.minimum.reduce
        ma_max = np.ma.maximum.reduce

        # check that the default axis is still None, but warns on 2d arrays
        result = pytest.warns(MaskedArrayFutureWarning, ma_max, data2d)
        assert_equal(result, ma_max(data2d, axis=None))

        result = pytest.warns(MaskedArrayFutureWarning, ma_min, data2d)
        assert_equal(result, ma_min(data2d, axis=None))

        # no warnings on 1d, as both new and old defaults are equivalent
        result = ma_min(data1d)
        assert_equal(result, ma_min(data1d, axis=None))
        assert_equal(result, ma_min(data1d, axis=0))

        result = ma_max(data1d)
        assert_equal(result, ma_max(data1d, axis=None))
        assert_equal(result, ma_max(data1d, axis=0))


class TestDtypeSet:
    def test_deprecated_dtype_set(self):
        # gh-31192: setting dtype on a MaskedArray should emit DeprecationWarning
        x = np.ma.array([1, 2, 3], mask=[0, 1, 0], dtype=np.float64)
        with pytest.warns(DeprecationWarning, match="Setting the dtype"):
            x.dtype = np.int64

"""Tests for numpy.ma.testutils helpers."""

import numpy as np
import pytest
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_raises


class TestAssertEqual:
    def test_nan_scalars(self):
        # gh-6661: match numpy.testing.assert_equal NaN handling
        assert_equal(np.nan, np.nan)
        assert_equal(float("nan"), float("nan"))
        assert_equal(np.float64("nan"), np.float64("nan"))
        assert_raises(AssertionError, assert_equal, np.nan, 1.0)
        assert_raises(AssertionError, assert_equal, np.nan, 0)

    def test_nan_in_sequences(self):
        assert_equal([np.nan, 1.0], [np.nan, 1.0])
        assert_raises(AssertionError, assert_equal, [np.nan], [1.0])

    def test_masked_arrays_still_compared(self):
        a = np.ma.array([1.0, np.nan], mask=[False, True])
        b = np.ma.array([1.0, 2.0], mask=[False, True])
        assert_equal(a, b)
        assert_equal(np.ma.array([np.nan]), np.ma.array([np.nan]))

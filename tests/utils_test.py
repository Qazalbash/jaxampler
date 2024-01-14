#  Copyright 2023 The Jaxampler Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys

import pytest
from jax import numpy as jnp


sys.path.append("../jaxampler")
from jaxampler.utils import jx_cast, nCr, nPr


class TestUtils:
    def test_jx_cast_success(self):
        a = 1
        b = [1, 2, 3]
        shape, casted_a, casted_b = jx_cast(a, b)
        assert shape == (3,)
        assert isinstance(casted_a, jnp.ndarray)
        assert isinstance(casted_b, jnp.ndarray)
        assert casted_a.shape == ()
        assert casted_b.shape == (3,)
        assert jnp.all(casted_a == 1)
        assert jnp.all(casted_b == jnp.array([1, 2, 3]))

    def test_jx_cast_fail(self):
        g = jnp.array([[1, 2, 3], [4, 5, 6]])
        h = jnp.array([[1, 2]])
        with pytest.raises(ValueError):
            jx_cast(g, h)

    def test_nPr_exist(self):
        assert nPr(5, 3) == 60
        assert nPr(10, 3) == 720
        assert nPr(10, 5) == 30_240
        assert nPr(10, 10) == 3_628_800

    def test_nPr_not_exist(self):
        with pytest.raises(AssertionError):
            nPr(5, 6)
        with pytest.raises(AssertionError):
            nPr(10, 11)
        with pytest.raises(AssertionError):
            nPr(10, -1)
        with pytest.raises(AssertionError):
            nPr(-1, 10)

    def test_nCr_exist(self):
        assert nCr(5, 3) == 10
        assert nCr(10, 3) == 120
        assert nCr(10, 5) == 252
        assert nCr(10, 10) == 1

    def test_nCr_not_exist(self):
        with pytest.raises(AssertionError):
            nCr(5, 6)
        with pytest.raises(AssertionError):
            nCr(10, 11)
        with pytest.raises(AssertionError):
            nCr(10, -1)
        with pytest.raises(AssertionError):
            nCr(-1, 10)

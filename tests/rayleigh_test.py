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

import jax
import jax.numpy as jnp
import pytest

sys.path.append("../jaxampler")
from jaxampler.rvs import Rayleigh


class TestRayleigh:
    def test_pdf_x(self):
        assert Rayleigh(sigma=0.5).pdf_x(1) == 0.5413411

    def test_shapes(self):
        assert jnp.allclose(
            Rayleigh(sigma=[0.5, 0.3]).pdf_x(1),
            jnp.array([0.5413411, 0.04295468]),
        )
        assert jnp.allclose(
            Rayleigh(sigma=[0.5, 0.5, 0.5]).pdf_x(1),
            jnp.array([0.5413411, 0.5413411, 0.5413411]),
        )
        assert jnp.allclose(
            Rayleigh(sigma=[[0.3, 0.5], [0.5, 0.4]]).pdf_x(1),
            jnp.array([[0.04295468, 0.5413411], [0.5413411, 0.2746058]]),
        )

    def test_incompatible_shapes(self):
        with pytest.raises(ValueError):
            Rayleigh(sigma=[[0.3, 0.5], [0.5]])

    def test_out_of_bound(self):
        with pytest.raises(AssertionError):
            assert Rayleigh(sigma=-1)

    def test_cdf_x(self):
        # when x is less than 0
        assert Rayleigh(sigma=0.5).cdf_x(-0.01) == 0
        # when x is equal to 0
        assert Rayleigh(sigma=5.5).cdf_x(0) == 0
        # when x is greater than 0
        assert Rayleigh(sigma=500).cdf_x(30) == 0.0017983912

    def test_rvs(self):
        tpl_rvs = Rayleigh(sigma=0.1)
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = tpl_rvs.rvs(shape, key)
        assert result.shape, shape + tpl_rvs._shape

        # without key
        result = tpl_rvs.rvs(shape)
        assert result.shape, shape + tpl_rvs._shape

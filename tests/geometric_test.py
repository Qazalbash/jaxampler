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
from jax.scipy.stats import geom as jax_geom


sys.path.append("../jaxampler")
from jaxampler.rvs import Geometric


class TestGeometric:
    def test_positive_p(self):
        assert jnp.allclose(
            Geometric(p=0.5, name="test_logpmf_p0.5").logpmf(5),
            jax_geom.logpmf(5, 0.5),
        )

    def test_p_greater_than_1(self):
        with pytest.raises(AssertionError):
            Geometric(p=1.5, name="test_logpmf_p1.5")

    def test_small_p(self):
        assert jnp.allclose(
            Geometric(p=0.0001, name="test_pmf_p0.0001").pmf(5),
            jax_geom.pmf(5, 0.0001),
        )
        assert jnp.allclose(
            Geometric(p=0.0001, name="test_logpmf_p0.0001").logpmf(5),
            jax_geom.logpmf(5, 0.0001),
        )

    def test_big_n(self):
        assert jnp.allclose(
            Geometric(p=0.1, name="test_pmf_p0.1").pmf(50),
            jax_geom.pmf(50, 0.1),
        )

    def test_shapes(self):
        assert Geometric(p=[0.5, 0.1], name="test_logpmf_p2").logpmf(5).shape == (2,)
        assert Geometric(p=[0.5, 0.1, 0.3], name="test_logpmf_p3").logpmf(5).shape == (3,)
        assert jnp.allclose(Geometric(p=0.1).logpmf(50), jax_geom.logpmf(50, 0.1))
        assert jnp.allclose(Geometric(p=0.5, name="test_pmf_p0.5").pmf(5), jax_geom.pmf(5, 0.5))
        assert jnp.allclose(
            Geometric(p=[0.5, 0.1], name="test_pmf_p2").pmf(5), jax_geom.pmf(5, jnp.asarray([0.5, 0.1]))
        )
        assert Geometric(p=[[0.5, 0.1], [0.4, 0.1]], name="test_pmf_p2x3").pmf(5).shape == (2, 2)

    def test_cdf(self):
        geom_cdf = Geometric(p=0.2, name="test_cdf")
        assert geom_cdf.cdf(-1) == 0.0
        assert geom_cdf.cdf(9) >= 0.0
        assert geom_cdf.cdf(9) <= 1.0

    def test_rvs(self):
        geom_rvs = Geometric(p=0.6, name="tets_rvs")
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = geom_rvs.rvs(shape, key)
        assert result.shape, shape + geom_rvs._shape

        # without key
        result = geom_rvs.rvs(shape)
        assert result.shape, shape + geom_rvs._shape

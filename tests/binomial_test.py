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
from jax.scipy.stats import binom as jax_binom

sys.path.append("../jaxampler")
from jaxampler.rvs import Binomial


class TestBinomial:
    def test_shapes(self):
        assert jnp.allclose(Binomial(p=0.5, n=10, name="test_logpmf_p0.5").logpmf_x(5), jax_binom.logpmf(5, 10, 0.5))

        # for different shapes
        assert Binomial(p=0.5, n=[10, 20], name="test_logpmf_n2").logpmf_x(5).shape == (2,)
        assert Binomial(p=[0.5, 0.1], n=[10, 20], name="test_logpmf_p2n2").logpmf_x(5).shape == (2,)
        assert Binomial(p=[0.5, 0.1, 0.3], n=[10, 20, 30], name="test_logpmf_p3n3").logpmf_x(5).shape == (3,)

        # when probability is very small
        assert jnp.allclose(
            Binomial(p=0.0001, n=10, name="test_logpmf_p0.0001").logpmf_x(5), jax_binom.logpmf(5, 10, 0.0001)
        )

        # when n is very large
        assert jnp.allclose(Binomial(p=0.1, n=100).logpmf_x(50), jax_binom.logpmf(50, 100, 0.1))
        assert jnp.allclose(Binomial(p=0.5, n=10, name="test_pmf_p0.5").pmf_x(5), jax_binom.pmf(5, 10, 0.5))
        assert jnp.allclose(
            Binomial(p=0.5, n=[10, 20], name="test_pmf_n2").pmf_x(5), jax_binom.pmf(5, jnp.asarray([10, 20]), 0.5)
        )
        assert jnp.allclose(
            Binomial(p=[0.5, 0.1], n=[10, 20], name="test_pmf_p2n2").pmf_x(5),
            jax_binom.pmf(5, jnp.asarray([10, 20]), jnp.asarray([0.5, 0.1])),
        )
        assert Binomial(p=[[0.5, 0.1], [0.4, 0.1]], n=[[10], [20]], name="test_pmf_p2x3n2").pmf_x(5).shape == (2, 2)

        # when probability is very small
        assert jnp.allclose(Binomial(p=0.0001, n=10, name="test_pmf_p0.0001").pmf_x(5), jax_binom.pmf(5, 10, 0.0001))

        # when n is very large
        assert jnp.allclose(
            Binomial(p=0.1, n=100000, name="test_pmf_p0.1n100000").pmf_x(50), jax_binom.pmf(50, 100000, 0.1)
        )

    def test_cdf_x(self):
        bin_cdf = Binomial(p=0.2, n=12, name="test_cdf")
        assert bin_cdf.cdf_x(13) == 1.0
        assert bin_cdf.cdf_x(-1) == 0.0
        assert bin_cdf.cdf_x(9) >= 0.0
        assert bin_cdf.cdf_x(9) <= 1.0

    def test_rvs(self):
        bin_rvs = Binomial(p=0.6, n=5, name="tets_rvs")
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = bin_rvs.rvs(shape, key)
        assert result.shape, shape + bin_rvs._shape

        # without key
        result = bin_rvs.rvs(shape)
        assert result.shape, shape + bin_rvs._shape

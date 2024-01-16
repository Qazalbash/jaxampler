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
from jax.scipy.stats import uniform as jax_uniform

sys.path.append("../jaxampler")
from jaxampler.rvs import Uniform


class TestUniform:
    def test_shape(self):
        assert jnp.allclose(Uniform(low=0.0, high=10.0, name="uniform_0_to_10").pdf_x(5), jax_uniform.pdf(5, 0, 10))

        # when low is negative
        assert jnp.allclose(
            Uniform(low=-10.0, high=10.0, name="uniform_n10_to_10").pdf_x(5), jax_uniform.pdf(5, -10, 20)
        )

        # when both low and high are negative
        assert jnp.allclose(
            Uniform(low=-10.0, high=-1.0, name="uniform_n10_to_n1").pdf_x(5), jax_uniform.pdf(5, -10, 9)
        )

        # when low is equal to high
        with pytest.raises(AssertionError):
            Uniform(low=10.0, high=10.0, name="uniform_10_to_10")

        # when high is greater than low
        with pytest.raises(AssertionError):
            Uniform(low=10.0, high=0.0, name="uniform_10_to_0")

    def test_cdf_x(self):
        uniform_cdf = Uniform(low=0.0, high=10.0, name="cdf_0_to_10")
        assert uniform_cdf.cdf_x(5) <= 1
        assert uniform_cdf.cdf_x(5) >= 0
        assert uniform_cdf.cdf_x(15) == 1
        assert uniform_cdf.cdf_x(-1) == 0

        # when low is negative
        uniform_cdf = Uniform(low=-10.0, high=10.0, name="cdf_n10_to_10")
        assert uniform_cdf.cdf_x(0) <= 1
        assert uniform_cdf.cdf_x(0) >= 0
        assert uniform_cdf.cdf_x(15) == 1
        assert uniform_cdf.cdf_x(-11) == 0

        # when low and high are negative
        uniform_cdf = Uniform(low=-10.0, high=-1.0, name="cdf_n10_to_n1")
        assert uniform_cdf.cdf_x(-5) <= 1
        assert uniform_cdf.cdf_x(-5) >= 0
        assert uniform_cdf.cdf_x(1) == 1
        assert uniform_cdf.cdf_x(-20) == 0

    def test_rvs(self):
        uniform_rvs = Uniform(low=0.0, high=10.0, name="tets_rvs")
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = uniform_rvs.rvs(shape, key)
        assert result.shape, shape + uniform_rvs._shape

        # without key
        result = uniform_rvs.rvs(shape)
        assert result.shape, shape + uniform_rvs._shape

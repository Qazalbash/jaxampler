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
from jaxampler.rvs import Triangular


class TestTriangular:
    def test_x_is_less_than_low(self):
        assert jnp.allclose(
            Triangular(
                low=0,
                mode=5,
                high=10,
                name="triangular_0_to_10",
            ).pdf(-1),
            0,
        )

    def test_x_is_less_than_mid(self):
        assert jnp.allclose(
            Triangular(
                low=0,
                mode=5,
                high=10,
                name="triangular_0_to_10",
            ).pdf(2),
            jnp.exp(jnp.log(2) + jnp.log(2) - jnp.log(10) - jnp.log(5)),
        )

    def test_x_is_equal_to_mid(self):
        assert jnp.allclose(
            Triangular(
                low=0,
                mode=5,
                high=10,
                name="triangular_0_to_10",
            ).pdf(5),
            0.2,
        )

    def test_x_is_greater_than_mid(self):
        assert jnp.allclose(
            Triangular(
                low=0.0,
                mode=5.0,
                high=10.0,
                name="triangular_0_to_10",
            ).pdf(7),
            0.12,
        )

    def test_x_is_greater_than_high(self):
        assert jnp.allclose(
            Triangular(
                low=0,
                mode=5,
                high=10,
                name="triangular_0_to_10",
            ).pdf(11),
            0,
        )

    def test_low_is_negative(self):
        assert jnp.allclose(
            Triangular(
                low=-10,
                mode=5,
                high=10,
                name="triangular_n10_to_10",
            ).pdf(2),
            0.08,
        )

    def test_low_and_high_are_negative(self):
        assert jnp.allclose(
            Triangular(
                low=-10,
                mode=-5,
                high=-1,
                name="triangular_n10_to_n1",
            ).pdf(-9),
            2 / 45,
        )

    def test_low_is_equal_to_high(self):
        with pytest.raises(AssertionError):
            Triangular(
                low=10,
                mode=5,
                high=0,
                name="triangular_10_to_0",
            )

    def test_low_is_greater_than_mid(self):
        with pytest.raises(AssertionError):
            Triangular(
                low=1,
                mode=0,
                high=10,
                name="triangular_0_to_10",
            )

    def test_mid_is_greater_than_high(self):
        with pytest.raises(AssertionError):
            Triangular(
                low=10,
                mode=30,
                high=20,
                name="triangular_10_to_20",
            )

    def test_cdf(self):
        triangular_cdf = Triangular(
            low=0,
            mode=5,
            high=10,
            name="cdf_0_to_10",
        )
        # when x is equal to mode
        assert triangular_cdf.cdf(5) == 0.5
        # when x is greater than mid
        assert triangular_cdf.cdf(7) == 0.82
        # when x is less than mid
        assert triangular_cdf.cdf(2) == 0.08
        # when x is less than low
        assert triangular_cdf.cdf(-1) == 0

        ## when low is negative
        triangular_cdf = Triangular(
            low=-5,
            mode=5,
            high=10,
            name="cdf_n5_to_10",
        )
        # when x is equal to mode
        assert triangular_cdf.cdf(5) == 0.6666666
        # when x is greater than mid
        assert triangular_cdf.cdf(7) == 0.88
        # when x is less than mid
        assert triangular_cdf.cdf(2) == round(49 / 150, 8)
        # # when x is less than low
        assert triangular_cdf.cdf(-6) == 0

        # when both high and low are negative
        triangular_cdf = Triangular(
            low=-5,
            mode=-2.5,
            high=-1,
            name="cdf_n5_to_n1",
        )
        # when x is equal to mid
        assert triangular_cdf.cdf(-2.5) == 0.625
        # when x is greater than mid
        assert triangular_cdf.cdf(-2) == 0.8333334
        # when x is less than mid
        assert triangular_cdf.cdf(-3) == 0.39999998
        # when x is less than low
        assert triangular_cdf.cdf(-6) == 0

    def test_rvs(self):
        triangular_rvs = Triangular(low=0, mode=5, high=10, name="tets_rvs")
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = triangular_rvs.rvs(shape, key)
        assert result.shape, shape + triangular_rvs._shape

        # without key
        result = triangular_rvs.rvs(shape)
        assert result.shape, shape + triangular_rvs._shape

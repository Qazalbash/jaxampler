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
from jaxampler.rvs import Exponential


class TestExponential:
    def test_invalid_params(self):
        with pytest.raises(AssertionError):
            Exponential(scale=-1.0)
        with pytest.raises(AssertionError):
            Exponential(scale=0.0)

    def test_valid_params(self):
        Exponential(scale=1.0)
        Exponential(scale=1.0, name="test")
        Exponential(scale=jnp.ones((3, 2)))

    def test_cdf(self):
        scale = jnp.array([1.0, 0.5, 1.0 / 3.0])
        exp = Exponential(scale=scale)
        assert jnp.allclose(
            exp.cdf(0.0),
            jnp.zeros((3,)),
        )
        assert jnp.allclose(
            exp.cdf(1.0),
            1.0 - jnp.exp(-1.0 / scale),
        )
        assert jnp.allclose(
            exp.cdf(2.0),
            1.0 - jnp.exp(-2.0 / scale),
        )

    def test_ppf(self):
        scale = jnp.array([1.0, 2.0, 3.0])
        exp = Exponential(scale=scale)
        assert jnp.allclose(
            exp.ppf(0.0),
            jnp.zeros((3,)),
        )
        assert jnp.allclose(
            exp.ppf(1.0 - jnp.exp(-1.0 / scale)),
            1.0,
        )
        assert jnp.allclose(
            exp.ppf(1.0 - jnp.exp(-2.0 / scale)),
            2.0,
        )

    def test_rvs(self):
        scale = jnp.linspace(0.1, 10.0, 10)
        exp = Exponential(scale=scale)
        key = jax.random.PRNGKey(0)
        shape = (10000,)
        rvs = exp.rvs(shape, key)
        print(jnp.mean(rvs, axis=(0,)))
        assert rvs.shape == shape + scale.shape
        assert jnp.all(rvs >= 0.0)
        assert jnp.allclose(
            jnp.mean(rvs, axis=(0,)),
            scale,
            atol=0.15,
        )

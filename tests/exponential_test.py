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


class TestBinomial:
    def test_invalid_params(self):
        with pytest.raises(AssertionError):
            Exponential(-1.0)
        with pytest.raises(AssertionError):
            Exponential(0.0)

    def test_valid_params(self):
        Exponential(1.0)
        Exponential(1.0, name="test")
        Exponential(jnp.ones((3, 2)))

    def test_cdf_x(self):
        lambdas = jnp.array([1.0, 2.0, 3.0])
        exp = Exponential(lambdas)
        assert jnp.allclose(
            exp.cdf_x(0.0),
            jnp.zeros((3,)),
        )
        assert jnp.allclose(
            exp.cdf_x(1.0),
            1.0 - jnp.exp(-1.0 * lambdas),
        )
        assert jnp.allclose(
            exp.cdf_x(2.0),
            1.0 - jnp.exp(-2.0 * lambdas),
        )

    def test_ppf_x(self):
        lambdas = jnp.array([1.0, 2.0, 3.0])
        exp = Exponential(lambdas)
        assert jnp.allclose(
            exp.ppf_x(0.0),
            jnp.zeros((3,)),
        )
        assert jnp.allclose(
            exp.ppf_x(1.0 - jnp.exp(-lambdas)),
            1.0,
        )
        assert jnp.allclose(
            exp.ppf_x(1.0 - jnp.exp(-2.0 * lambdas)),
            2.0,
        )

    def test_rvs(self):
        lambdas = jnp.linspace(0.1, 10.0, 10)
        exp = Exponential(lambdas)
        key = jax.random.PRNGKey(0)
        shape = (1000,)
        rvs = exp.rvs(shape, key)
        print(
            jnp.var(rvs, axis=(0,)),
            jnp.power(lambdas, -2.0),
        )
        assert rvs.shape == shape + lambdas.shape
        assert jnp.all(rvs >= 0.0)
        assert jnp.allclose(
            jnp.mean(rvs, axis=(0,)),
            1.0 / lambdas,
            atol=0.1,
        )

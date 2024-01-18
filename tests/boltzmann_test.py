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

import jax.numpy as jnp
import pytest
from jax.scipy.special import erf

sys.path.append("../jaxampler")
from jaxampler.rvs import Boltzmann


class TestBoltzmann:
    def test_invalid_params(self):
        with pytest.raises(AssertionError):
            Boltzmann(a=-1.0)
        with pytest.raises(AssertionError):
            Boltzmann(a=0.0)

    def test_logpdf(self):
        a = jnp.linspace(0.1, 10.0, 100)
        boltzmann = Boltzmann(a=a)
        x = jnp.linspace(0.1, 10.0, 100)
        logpdf_val = boltzmann.logpdf_x(x)
        logpmf = (
            lambda x_, a_: 2 * jnp.log(x_)
            - 3 * jnp.log(a_)
            - 0.5 * jnp.log(jnp.pi / 2.0)
            - 0.5 * jnp.power(x_, 2) / jnp.power(a_, 2)
        )
        expected = jnp.array(list(map(logpmf, x, a)))
        assert jnp.allclose(logpdf_val, expected)

    def test_cdf(self):
        a = jnp.linspace(0.1, 10.0, 100)
        boltzmann = Boltzmann(a=a)
        x = jnp.linspace(0.1, 10.0, 100)
        cdf_val = boltzmann.cdf_x(x)
        cdf = lambda x_, a_: erf(x_ / (jnp.sqrt(2) * a_)) - jnp.exp(
            jnp.log(x_) - 0.5 * jnp.power(x_ / a_, 2) - 0.5 * jnp.log(jnp.pi / 2) - jnp.log(a_)
        )
        expected = jnp.array(list(map(cdf, x, a)))
        assert jnp.allclose(cdf_val, expected)

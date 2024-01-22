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
from jaxampler.rvs import Weibull


class TestWeibull:
    def test_pdf(self):
        W = Weibull(scale=1, k=1)
        assert jnp.allclose(W.pdf_x(1.0), 1 / jnp.e)
        assert jnp.allclose(W.pdf_x(0.0), 0)

    def test_negative_x(self):
        assert jnp.allclose(Weibull(scale=1, k=1).pdf_x(-1.0), 0)

    def test_negative_lambda(self):
        with pytest.raises(AssertionError):
            Weibull(scale=-1, k=1)

    def test_negative_k(self):
        with pytest.raises(AssertionError):
            Weibull(scale=1, k=-1)

    def test_negative_lambda_and_k(self):
        with pytest.raises(AssertionError):
            Weibull(scale=-1, k=-1)

    def test_shapes(self):
        assert jnp.allclose(Weibull(scale=[1, 1], k=[1, 1]).pdf_x(1.0), jnp.array([0.3678794412, 0.3678794412]))
        assert jnp.allclose(
            Weibull(scale=[1, 1, 1], k=[1, 1, 1]).pdf_x(1.0),
            jnp.array([0.3678794412, 0.3678794412, 0.3678794412]),
        )

    def test_cdf_x(self):
        W = Weibull(scale=[1, 1, 1], k=[1, 1, 1])
        assert jnp.allclose(W.cdf_x(1), 1 - 1 / jnp.e)
        assert jnp.allclose(W.cdf_x(0), 0)

    def test_ppf(self):
        W = Weibull(scale=1, k=1)
        assert jnp.allclose(W.ppf_x(0.0), 0.0)
        assert jnp.allclose(W.ppf_x(1 - 1 / jnp.e), 1)

    def test_rvs(self):
        W = Weibull(scale=1, k=1)
        rvs = W.rvs((1000, 1000), key=jax.random.PRNGKey(0))
        assert jnp.allclose(rvs.mean(), 1.0, atol=0.1)
        assert jnp.allclose(rvs.std(), 1.0, atol=0.1)

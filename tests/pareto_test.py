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
from jaxampler.rvs import Pareto


class TestPareto:

    def test_pdf(self):
        assert Pareto(alpha=0.5, k=0.1).pdf_x(1) == 0.047481652
        # when alpha is negative
        assert Pareto(alpha=-1, k=0.1).pdf_x(1) == 0.21714725

    def test_shapes(self):
        assert jnp.allclose(Pareto(alpha=[0.5, 0.1], k=[0.1, 0.2]).pdf_x(1), jnp.array([0.04748165, 0.08857405]))
        assert jnp.allclose(
            Pareto(alpha=[0.5, 0.1, 0.2], k=[0.1, 0.2, 0.2]).pdf_x(1), jnp.array([0.04748165, 0.08857405, 0.07641375]))

    def test_imcompatible_shapes(self):
        with pytest.raises(ValueError):
            Pareto(alpha=[0.5, 0.1, 0.9], k=[0.1, 0.2])
        with pytest.raises(ValueError):
            Pareto(alpha=[0.5, 0.1, 0.9], k=[0.1, 0.2, 0.9])
        with pytest.raises(ValueError):
            Pareto(alpha=[0.5, 0.1], k=[0.1, 0.2, 0.9])

    def test_out_of_bound(self):
        # when x is less than k
        assert jnp.allclose(Pareto(alpha=0.5, k=0.1).pdf_x(-1), 0)
        # when x is greater than high
        assert jnp.allclose(Pareto(alpha=0.5, k=0.1).pdf_x(11), 0)
        # when k is negative
        with pytest.raises(AssertionError):
            Pareto(alpha=0.5, k=-1)
        # when higj is less than k
        with pytest.raises(AssertionError):
            Pareto(alpha=0.5, k=2)

    def test_cdf_x(self):
        # when x is less than k
        assert Pareto(alpha=0.5, k=0.1).cdf_x(0.01) == 0
        # when x is greater than high
        assert Pareto(alpha=0.5, k=0.1).cdf_x(11) == 1
        # when beta is equal to 0
        assert Pareto(alpha=-1, k=0.1).cdf_x(1) == 0.5
        # when beta is not equal to 0
        assert Pareto(alpha=0.5, k=0.1).cdf_x(1) == 0.030653432

    def test_rvs(self):
        tpl_rvs = Pareto(alpha=0.1, k=0.1)
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = tpl_rvs.rvs(shape, key)
        assert result.shape, shape + tpl_rvs._shape

        # without key
        result = tpl_rvs.rvs(shape)
        assert result.shape, shape + tpl_rvs._shape

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
        assert Pareto(alpha=0.5, scale=0.1).pdf_x(1) == 0.15811388

    def test_shapes(self):
        assert jnp.allclose(Pareto(alpha=[0.5, 0.1], scale=[0.1, 0.2]).pdf_x(1), jnp.array([0.15811388, 0.08513397]))
        assert jnp.allclose(
            Pareto(alpha=[0.5, 0.1, 0.2], scale=[0.1, 0.2, 0.2]).pdf_x(1),
            jnp.array([0.15811388, 0.08513397, 0.14495593]))

    def test_imcompatible_shapes(self):
        with pytest.raises(ValueError):
            Pareto(alpha=[0.5, 0.1, 0.9], scale=[0.1, 0.2])

    def test_out_of_bound(self):
        # when x is less than zero
        assert jnp.allclose(Pareto(alpha=0.5, scale=0.1).pdf_x(-1), 0)
        # when x is greater than scale
        with pytest.raises(AssertionError):
            assert jnp.allclose(Pareto(alpha=0.5, scale=0.1).pdf_x(11), 0)
        # when scale is negative
        with pytest.raises(AssertionError):
            Pareto(alpha=0.5, scale=-1)
        # when alpha is negative
        with pytest.raises(AssertionError):
            Pareto(alpha=-1, scale=2)

    def test_cdf_x(self):
        # when x is less than scale
        assert Pareto(alpha=0.5, scale=0.1).cdf_x(0.01) == 0
        # when x is greater than scale
        assert Pareto(alpha=0.5, scale=0.1).cdf_x(1) == 0.6837722

    def test_rvs(self):
        tpl_rvs = Pareto(alpha=0.1, scale=0.1)
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = tpl_rvs.rvs(shape, key)
        assert result.shape, shape + tpl_rvs._shape

        # without key
        result = tpl_rvs.rvs(shape)
        assert result.shape, shape + tpl_rvs._shape

# Copyright 2023 The Jaxampler Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

sys.path.append("../jaxampler")
import sys
import unittest
import jax
import jax.numpy as jnp
from jaxampler._src.rvs.binomial import Binomial


eps = 1e-3


class TestBinomial(unittest.TestCase):
    def test_logpmf_x(self):
        binomial = Binomial(p=0.5, n=10)
        result = binomial.logpmf_x(5)
        expected = jax.scipy.stats.binom.logpmf(5, 10, 0.5)
        self.assertTrue(jnp.allclose(result, expected))

        # when probability is very small
        binomial = Binomial(p=0.0001, n=10)
        result = binomial.logpmf_x(5)
        expected = jax.scipy.stats.binom.logpmf(5, 10, 0.0001)
        self.assertTrue(jnp.allclose(result, expected))

        # when n is very large
        binomial = Binomial(p=0.1, n=100000)
        result = binomial.logpmf_x(50)
        expected = jax.scipy.stats.binom.logpmf(50, 100000, 0.1)
        self.assertTrue(jnp.allclose(result, expected))

    def test_pmf_x(self):
        binomial = Binomial(p=0.5, n=10)
        result = binomial.pmf_x(5)
        expected = jax.scipy.stats.binom.pmf(5, 10, 0.5)
        self.assertTrue(jnp.allclose(result, expected))

        # when probability is very small
        binomial = Binomial(p=0.0001, n=10)
        result = binomial.pmf_x(5)
        expected = jax.scipy.stats.binom.pmf(5, 10, 0.0001)
        self.assertTrue(jnp.allclose(result, expected))

        # when n is very large
        binomial = Binomial(p=0.1, n=100000)
        result = binomial.pmf_x(50)
        expected = jax.scipy.stats.binom.pmf(50, 100000, 0.1)
        self.assertTrue(jnp.allclose(result, expected))

    def test_cdf_x(self):
        binomial = Binomial(p=0.2, n=12)
        result = binomial.cdf_x(13)
        expected = 1
        assert result == expected
        result = binomial.cdf_x(-1)
        expected = 0
        assert result == expected
        result = binomial.cdf(9)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_rvs(self):
        binomial = Binomial(p=0.6, n=5)
        shape = (3, 4)

        # with key
        key = jax.random.PRNGKey(123)
        result = binomial.rvs(shape, key)
        self.assertEqual(result.shape, shape + binomial._shape)

        # without key
        result = binomial.rvs(shape)
        self.assertEqual(result.shape, shape + binomial._shape)


if __name__ == "__main__":
    unittest.main()

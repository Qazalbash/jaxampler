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

from jax import numpy as jnp

sys.path.append("../jaxampler")
from jaxampler.rvs import Normal, Beta


class TestRandomVariable:
    beta = Beta(alpha=2, beta=2)
    norms = [Normal(loc=i, scale=1) for i in jnp.linspace(-5, 5, 30)]
    xx = jnp.linspace(-5, 5, 1000)

    def test_adding_rvs(self):
        Z = sum(self.norms)
        assert jnp.allclose(Z.pdf(self.xx), sum(norm.pdf(self.xx) for norm in self.norms))

    def test_adding_number(self):
        Z = self.norms[0] + 2.6
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) + 2.6)

    def test_subtracting_rvs(self):
        Z = self.norms[0] - self.norms[1]
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) - self.norms[1].pdf(self.xx))

    def test_subtracting_number(self):
        Z = self.norms[0] - 2.6
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) - 2.6)

    def test_multiplying_rvs(self):
        Z = self.norms[0] * self.norms[1]
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) * self.norms[1].pdf(self.xx))

    def test_multiplying_number(self):
        Z = self.norms[0] * 2.6
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) * 2.6)

    def test_dividing_rvs(self):
        Z = self.norms[0] / self.norms[1]
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) / self.norms[1].pdf(self.xx))

    def test_dividing_number(self):
        Z = self.norms[0] / 2.6
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) / 2.6)

    def test_powering_rvs(self):
        Z = self.norms[0] ** self.norms[1]
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) ** self.norms[1].pdf(self.xx))

    def test_powering_number(self):
        Z = self.norms[0] ** 2.6
        assert jnp.allclose(Z.pdf(self.xx), self.norms[0].pdf(self.xx) ** 2.6)

    def test_negating_rvs(self):
        Z = -self.norms[0]
        assert jnp.allclose(Z.pdf(self.xx), -self.norms[0].pdf(self.xx))

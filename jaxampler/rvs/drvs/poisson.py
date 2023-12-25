# Copyright 2023 The JAXampler Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import poisson as jax_poisson
from jax.typing import ArrayLike

from .drvs import DiscreteRV


class Poisson(DiscreteRV):

    def __init__(self, lmbda: ArrayLike, name: str = None) -> None:
        self._lmbda = lmbda
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._lmbda > 0.0), "Lambda must be positive"

    @partial(jit, static_argnums=(0,))
    def logpmf(self, k: ArrayLike) -> ArrayLike:
        return jax_poisson.logpmf(k, self._lmbda)

    @partial(jit, static_argnums=(0,))
    def pmf(self, k: ArrayLike) -> ArrayLike:
        return jax_poisson.pmf(k, self._lmbda)

    @partial(jit, static_argnums=(0,))
    def cdf(self, k: ArrayLike) -> ArrayLike:
        return jax_poisson.cdf(k, self._lmbda)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.poisson(key, self._lmbda, shape=(N, 1))

    def __repr__(self) -> str:
        string = f"Poisson(lmbda={self._lmbda}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

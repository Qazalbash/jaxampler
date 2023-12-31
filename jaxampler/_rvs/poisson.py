# Copyright 2023 The Jaxampler Authors

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

from ..utils import jx_cast
from .drvs import DiscreteRV


class Poisson(DiscreteRV):
    def __init__(self, lmbda: ArrayLike, name: str = None) -> None:
        (
            shape,
            self._lmbda,
        ) = jx_cast(lmbda)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._lmbda > 0.0), "Lambda must be positive"

    @partial(jit, static_argnums=(0,))
    def logpmf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_poisson.logpmf(x, self._lmbda)

    @partial(jit, static_argnums=(0,))
    def pmf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_poisson.pmf(x, self._lmbda)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_poisson.cdf(x, self._lmbda)

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key()
        shape += self._shape
        return jax.random.poisson(key, self._lmbda, shape=shape)

    def __repr__(self) -> str:
        string = f"Poisson(lmbda={self._lmbda}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

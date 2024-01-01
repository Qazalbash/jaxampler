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
from jax import vmap
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class Rayleigh(ContinuousRV):

    def __init__(self, sigma: float, name: str = None) -> None:
        self._sigma, = jx_cast(sigma)
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jnp.where(
            xx >= 0,
            jnp.log(xx) - 0.5 * jnp.power(xx / self._sigma, 2) - 2 * jnp.log(self._sigma),
            -jnp.inf,
        ))(x)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jnp.where(
            xx >= 0,
            jnp.log1p(-jnp.exp(-0.5 * jnp.power(xx / self._sigma, 2))),
            -jnp.inf,
        ))(x)

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jnp.where(
            xx >= 0,
            jnp.log(self._sigma) + 0.5 * jnp.log(-2 * jnp.log1p(-xx)),
            -jnp.inf,
        ))(x)

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.rayleigh(key, scale=self._sigma, shape=shape)

    def __repr__(self) -> str:
        string = f"Rayleigh(sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

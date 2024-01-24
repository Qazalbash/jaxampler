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

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import jax
from jax import Array, jit, numpy as jnp

from ..typing import Numeric
from ..utils import jx_cast
from .rvs import RandomVariable


class Rayleigh(RandomVariable):
    def __init__(self, loc: Numeric | Any = 0.0, sigma: Numeric | Any = 1.0, name: Optional[str] = None) -> None:
        shape, self._loc, self._sigma = jx_cast(loc, sigma)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x >= 0,
            jnp.log(x - self._loc) - 0.5 * jnp.power((x - self._loc) / self._sigma, 2) - 2 * jnp.log(self._sigma),
            -jnp.inf,
        )

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x >= 0,
            jnp.log1p(-jnp.exp(-0.5 * jnp.power((x - self._loc) / self._sigma, 2))),
            -jnp.inf,
        )

    @partial(jit, static_argnums=(0,))
    def _ppf_x(self, x: Numeric) -> Numeric | tuple[Numeric, ...]:
        return jnp.where(
            x < 0,
            jnp.zeros_like(x),
            self._loc + self._sigma * jnp.sqrt(-2 * jnp.log(1 - x)),
        )

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + jax.random.rayleigh(key, scale=self._sigma, shape=shape)

    def __repr__(self) -> str:
        string = f"Rayleigh(sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

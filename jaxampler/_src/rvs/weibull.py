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
from .crvs import ContinuousRV


class Weibull(ContinuousRV):
    def __init__(
        self,
        k: Numeric | Any,
        loc: Numeric | Any = 0.0,
        scale: Numeric | Any = 1.0,
        name: Optional[str] = None,
    ) -> None:
        shape, self._k, self._loc, self._scale = jx_cast(k, loc, scale)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._scale > 0.0), "scale must be greater than 0"
        assert jnp.all(self._k > 0.0), "concentration must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, x: Numeric) -> Numeric | tuple[Numeric, ...]:
        return jnp.where(
            x <= 0,
            jnp.full_like(x, -jnp.inf),
            jnp.log(self._k)
            - (self._k * jnp.log(self._scale))
            + (self._k - 1.0) * jnp.log(x - self._loc)
            - jnp.power(x / self._scale, self._k),
        )

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x <= 0.0,
            0.0,
            1.0 - jnp.exp(-jnp.power((x - self._loc) / self._scale, self._k)),
        )

    @partial(jit, static_argnums=(0,))
    def _ppf_x(self, x: Numeric) -> Numeric:
        return self._loc + self._scale * jnp.power(-jnp.log(1.0 - x), 1.0 / self._k)

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        U = jax.random.uniform(key, shape=shape)
        return self._ppf_x(U)

    def __repr__(self) -> str:
        string = f"Weibull(k={self._k}, loc={self._loc}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

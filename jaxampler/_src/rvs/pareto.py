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
from jax.scipy.stats import pareto as jax_pareto

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class Pareto(ContinuousRV):
    def __init__(self, alpha: Numeric | Any, scale: Numeric | Any, name: Optional[str] = None) -> None:
        shape, self._alpha, self._scale = jx_cast(alpha, scale)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._alpha > 0.0), "alpha must be greater than 0"
        assert jnp.all(self._scale > 0.0), "scale must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_pareto.logpdf(x, self._alpha, scale=self._scale)

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_pareto.pdf(x, self._alpha, scale=self._scale)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            self._scale <= x,
            jnp.log1p(-jnp.power(self._scale / x, self._alpha)),
            -jnp.inf,
        )

    @partial(jit, static_argnums=(0,))
    def logppf_x(self, x: Numeric) -> Numeric:
        conditions = [
            x < 0.0,
            (0.0 <= x) & (x < 1.0),
            1.0 <= x,
        ]
        choices = [
            -jnp.inf,
            jnp.log(self._scale) - (1.0 / self._alpha) * jnp.log(1 - x),
            jnp.log(1.0),
        ]
        return jnp.select(conditions, choices)

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.pareto(key, self._alpha, shape=new_shape) * self._scale

    def __repr__(self) -> str:
        string = f"Pareto(alpha={self._alpha}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

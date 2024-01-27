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
from jax.scipy.stats import expon as jax_expon

from ..typing import Numeric
from ..utils import jxam_array_cast
from .rvs import RandomVariable


class Exponential(RandomVariable):
    def __init__(
        self,
        loc: Numeric | Any = 0.0,
        scale: Numeric | Any = 1.0,
        name: Optional[str] = None,
    ) -> None:
        shape, self._loc, self._scale = jxam_array_cast(loc, scale)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._scale > 0.0), "lmbda must be positive"

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, x: Numeric) -> Numeric:
        return jax_expon.logpdf(
            x=x,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _pdf_x(self, x: Numeric) -> Numeric:
        return jax_expon.pdf(
            x=x,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x >= 0,
            jnp.log1p(-jnp.exp((self._loc - x) / self._scale)),
            -jnp.inf,
        )

    @partial(jit, static_argnums=(0,))
    def _logppf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x >= 0,
            jnp.log(self._loc - self._scale * jnp.log(1.0 - x)),
            -jnp.inf,
        )

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        U = jax.random.uniform(key=key, shape=shape)
        rvs_val = self._loc - self._scale * jnp.log(U)
        return rvs_val

    def __repr__(self) -> str:
        string = f"Exponential(loc={self._loc}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

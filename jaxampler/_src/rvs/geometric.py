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
from jax.scipy.stats import geom as jax_geom

from ..typing import Numeric
from ..utils import jxam_array_cast
from .rvs import RandomVariable


class Geometric(RandomVariable):
    """Geometric Random Variable"""

    def __init__(self, p: Numeric | Any, loc: Numeric | Any = 0.0, name: Optional[str] = None) -> None:
        shape, self._p, self._loc = jxam_array_cast(p, loc)
        self.check_params()
        self._q = 1.0 - self._p
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(0.0 <= self._p) & jnp.all(
            self._p <= 1.0
        ), "All p must be greater than or equals to 0 and less than or equals to 1"

    @partial(jit, static_argnums=(0,))
    def _logpmf_x(self, x: Numeric) -> Numeric:
        return jax_geom.logpmf(
            k=x,
            p=self._p,
            loc=self._loc,
        )

    @partial(jit, static_argnums=(0,))
    def _pmf_x(self, x: Numeric) -> Numeric:
        return jax_geom.pmf(
            k=x,
            p=self._p,
            loc=self._loc,
        )

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, x: Numeric) -> Numeric:
        conditions = [x < self._loc, x >= self._loc]
        choices = [jnp.zeros_like(self._q), 1.0 - jnp.power(self._q, jnp.floor(x - self._loc))]
        return jnp.select(conditions, choices)

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.log(self._cdf_x(x))

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + jax.random.geometric(key=key, p=self._p, shape=shape)

    def __repr__(self) -> str:
        string = f"Geometric(p={self._p}"
        if self._name is not None:
            string += f", (name: {self._name}"
        string += ")"
        return string

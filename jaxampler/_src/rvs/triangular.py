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
from jax import Array, jit, lax, numpy as jnp

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class Triangular(ContinuousRV):
    def __init__(
        self,
        low: Numeric | Any = 0,
        mode: Numeric | Any = 0.5,
        high: Numeric | Any = 1,
        name: Optional[str] = None,
    ) -> None:
        shape, self._low, self._mode, self._high = jx_cast(low, mode, high)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._low < self._high), "low must be less than or equal to high"
        assert jnp.all(self._low <= self._mode), "low must be less than or equal to mid"
        assert jnp.all(self._mode <= self._high), "mid must be less than or equal to high"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        conditions = [
            x < self._low,
            (self._low <= x) & (x < self._mode),
            x == self._mode,
            (self._mode < x) & (x <= self._high),
            x > self._high,
        ]
        choices = [
            -jnp.inf,
            jnp.log(2) + jnp.log(x - self._low) - jnp.log(self._high - self._low) - jnp.log(self._mode - self._low),
            jnp.log(2) - jnp.log(self._high - self._low),
            jnp.log(2) + jnp.log(self._high - x) - jnp.log(self._high - self._low) - jnp.log(self._high - self._mode),
            -jnp.inf,
        ]
        return jnp.select(conditions, choices)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        conditions = [
            x < self._low,
            (self._low < x) & (x <= self._mode),
            (self._mode < x) & (x < self._high),
            x >= self._high,
        ]
        choices = [
            -jnp.inf,
            2 * jnp.log(x - self._low) - jnp.log(self._high - self._low) - jnp.log(self._mode - self._low),
            jnp.log(
                1
                - jnp.exp(
                    (2 * jnp.log(self._high - x) - jnp.log(self._high - self._low) - jnp.log(self._high - self._mode))
                )
            ),
            jnp.log(1.0),
        ]
        return jnp.select(conditions, choices)

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: Numeric) -> Numeric:
        _Fc = self.cdf_v(self._mode)
        ppf_val = jnp.where(
            x < _Fc,
            self._low + lax.sqrt(x * (self._mode - self._low) * (self._high - self._low)),
            self._high - lax.sqrt((1 - x) * (self._high - self._low) * (self._high - self._mode)),
        )
        return ppf_val

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.triangular(
            key,
            left=self._low,
            right=self._high,
            mode=self._mode,
            shape=new_shape,
        )

    def __repr__(self) -> str:
        string = f"Triangular(low={self._low}, mode={self._mode}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

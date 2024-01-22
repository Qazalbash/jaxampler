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
from jax import jit, numpy as jnp
from jax.scipy.stats import uniform as jax_uniform
from jaxtyping import Array

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class Uniform(ContinuousRV):
    def __init__(self, low: Numeric | Any, high: Numeric | Any, name: Optional[str] = None) -> None:
        shape, self._low, self._high = jx_cast(low, high)
        self.check_params()
        super().__init__(name, shape)

    def check_params(self) -> None:
        assert jnp.all(self._low < self._high), "All low must be less than high"

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, x: Numeric) -> Numeric:
        return jax_uniform.logpdf(
            x,
            loc=self._low,
            scale=self._high - self._low,
        )

    @partial(jit, static_argnums=(0,))
    def _pdf_x(self, x: Numeric) -> Numeric:
        return jax_uniform.pdf(
            x,
            loc=self._low,
            scale=self._high - self._low,
        )

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        conditions = [
            x < self._low,
            (self._low <= x) & (x <= self._high),
            self._high < x,
        ]
        choice = [
            -jnp.inf,
            jnp.log(x - self._low) - jnp.log(self._high - self._low),
            jnp.log(1.0),
        ]
        return jnp.select(conditions, choice)

    @partial(jit, static_argnums=(0,))
    def _logppf_x(self, x: Numeric) -> Numeric:
        return jnp.log(x * (self._high - self._low) + self._low)

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return jax.random.uniform(key, minval=self._low, maxval=self._high, shape=shape)

    def __repr__(self) -> str:
        string = f"Uniform(low={self._low}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

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
from jax.scipy.stats import truncnorm as jax_truncnorm

from ..typing import Numeric
from ..utils import jx_cast
from .rvs import RandomVariable


class TruncNormal(RandomVariable):
    def __init__(
        self,
        loc: Numeric | Any = 0.0,
        scale: Numeric | Any = 1.0,
        low: Numeric | Any = -1.0,
        high: Numeric | Any = 1.0,
        name: Optional[str] = None,
    ) -> None:
        shape, self._loc, self._scale, self._low, self._high = jx_cast(loc, scale, low, high)
        self.check_params()
        self._alpha = (self._low - self._loc) / self._scale
        self._beta = (self._high - self._loc) / self._scale
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._low < self._high), "low must be smaller than high"
        assert jnp.all(self._scale > 0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, x: Numeric) -> Numeric:
        return jax_truncnorm.logpdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _pdf_x(self, x: Numeric) -> Numeric:
        return jax_truncnorm.pdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        return jax_truncnorm.logcdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, x: Numeric) -> Numeric:
        return jax_truncnorm.cdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + self._scale * jax.random.truncated_normal(
            key=key,
            lower=self._alpha,
            upper=self._beta,
            shape=shape,
        )

    def __repr__(self) -> str:
        string = f"TruncNorm(mu={self._loc}, sigma={self._scale}, low={self._low}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

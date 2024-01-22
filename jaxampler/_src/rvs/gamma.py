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
from jax.scipy.stats import gamma as jax_gamma

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class Gamma(ContinuousRV):
    def __init__(
        self,
        a: Numeric | Any,
        loc: Numeric | Any = 0.0,
        scale: Numeric | Any = 1.0,
        name: Optional[str] = None,
    ) -> None:
        shape, self._a, self._loc, self._scale = jx_cast(a, loc, scale)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._a > 0), "All a must be greater than 0"
        assert jnp.all(self._scale > 0), "All scale must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.logpdf(
            x=x,
            a=self._a,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.pdf(
            x=x,
            a=self._a,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.logcdf(
            x=x,
            a=self._a,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.cdf(
            x=x,
            a=self._a,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def logppf_x(self, x: Numeric) -> Numeric:
        raise NotImplementedError("Not able to find sufficient information to implement")

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + self._scale * jax.random.gamma(key=key, a=self._a, shape=shape)

    def __repr__(self) -> str:
        string = f"Gamma(a={self._a}, loc={self._loc}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

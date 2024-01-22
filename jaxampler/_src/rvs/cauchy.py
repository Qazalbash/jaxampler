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
from jax.scipy.stats import cauchy as jax_cauchy

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class Cauchy(ContinuousRV):
    def __init__(self, loc: Numeric | Any = 0, scale: Numeric | Any = 1.0, name: Optional[str] = None) -> None:
        shape, self._loc, self._scale = jx_cast(loc, scale)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._scale > 0.0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.logpdf(
            x=x,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _pdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.pdf(
            x=x,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.logcdf(
            x=x,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.cdf(
            x,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _ppf_x(self, x: Numeric) -> Numeric:
        return self._loc + self._scale * jnp.tan(jnp.pi * (x - 0.5))

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + self._scale * jax.random.cauchy(key=key, shape=shape)

    def __repr__(self) -> str:
        string = f"Cauchy(loc={self._loc}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

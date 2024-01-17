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
from jax.scipy.special import log_ndtr, ndtr, ndtri

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class LogNormal(ContinuousRV):
    def __init__(self, mu: Numeric | Any = 0.0, sigma: Numeric | Any = 1.0, name: Optional[str] = None) -> None:
        shape, self._mu, self._sigma = jx_cast(mu, sigma)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "All sigma must be greater than 0.0"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        constants = -(jnp.log(self._sigma) + 0.5 * jnp.log(2 * jnp.pi))
        logpdf_val = jnp.where(
            x <= 0,
            -jnp.inf,
            constants - jnp.log(x) - (0.5 * jnp.power(self._sigma, -2)) * jnp.power((jnp.log(x) - self._mu), 2),
        )
        return logpdf_val

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return log_ndtr((jnp.log(x) - self._mu) / self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return ndtr((jnp.log(x) - self._mu) / self._sigma)

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: Numeric) -> Numeric:
        return jnp.exp(self._mu + self._sigma * ndtri(x))

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        U = jax.random.uniform(key, shape=new_shape)
        return self.ppf_x(U)

    def __repr__(self) -> str:
        string = f"LogNormal(mu={self._mu}, sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

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
from jax.scipy.stats import poisson as jax_poisson

from ..typing import Numeric
from ..utils import jx_cast
from .rvs import RandomVariable


class Poisson(RandomVariable):
    def __init__(self, mu: Numeric | Any, loc: Numeric | Any = 0.0, name: Optional[str] = None) -> None:
        shape, self._mu, self._loc = jx_cast(mu, loc)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._mu > 0.0), "Lambda must be positive"

    @partial(jit, static_argnums=(0,))
    def _logpmf_x(self, x: Numeric) -> Numeric:
        return jax_poisson.logpmf(
            k=x,
            mu=self._mu,
            loc=self._loc,
        )

    @partial(jit, static_argnums=(0,))
    def _pmf_x(self, x: Numeric) -> Numeric:
        return jax_poisson.pmf(
            k=x,
            mu=self._mu,
            loc=self._loc,
        )

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.log(self._cdf_x(x))

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, x: Numeric) -> Numeric:
        return jax_poisson.cdf(
            k=x,
            mu=self._mu,
            loc=self._loc,
        )

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + jax.random.poisson(key=key, lam=self._mu, shape=shape)

    def __repr__(self) -> str:
        string = f"Poisson(lmbda={self._mu}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

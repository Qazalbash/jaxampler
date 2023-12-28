# Copyright 2023 The JAXampler Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import cauchy as jax_cauchy
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class Cauchy(ContinuousRV):

    def __init__(self, sigma: ArrayLike, loc: ArrayLike = 0, name: str = None) -> None:
        self._sigma, self._loc = jx_cast(sigma, loc)
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.logpdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.pdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.logcdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_cauchy.cdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return self._loc + self._sigma * jnp.tan(jnp.pi * (x - 0.5))

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        shape = (N,) + (self._sigma.shape or (1,))
        return jax.random.cauchy(key, shape=shape) * self._sigma + self._loc

    def __repr__(self) -> str:
        string = f"Cauchy(sigma={self._sigma}, loc={self._loc}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

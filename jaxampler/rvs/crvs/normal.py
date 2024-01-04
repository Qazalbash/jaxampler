# Copyright 2023 The Jaxampler Authors

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
from jax.scipy.stats import norm as jax_norm
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class Normal(ContinuousRV):

    def __init__(self, mu: ArrayLike = 0.0, sigma: ArrayLike = 1.0, name: str = None) -> None:
        self._mu, self._sigma = jx_cast(mu, sigma)
        self.check_params()
        self._logZ = 0.0
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "All sigma must be greater than 0.0"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.logpdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.logcdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.pdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.cdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.ppf(x, self._mu, self._sigma)

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.normal(key, shape=shape) * self._sigma + self._mu

    def __repr__(self) -> str:
        string = f"Normal(mu={self._mu}, sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

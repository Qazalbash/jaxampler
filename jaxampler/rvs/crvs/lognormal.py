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
from jax import vmap
from jax.scipy.special import log_ndtr, ndtr, ndtri
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class LogNormal(ContinuousRV):

    def __init__(self, mu: ArrayLike = 0.0, sigma: ArrayLike = 1.0, name: str = None) -> None:
        self._mu, self._sigma = jx_cast(mu, sigma)
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "All sigma must be greater than 0.0"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:

        def logpdf_x(xx: ArrayLike) -> ArrayLike:
            constants = -(jnp.log(self._sigma) + 0.5 * jnp.log(2 * jnp.pi))
            logpdf_val = jnp.where(
                xx <= 0,
                -jnp.inf,
                constants - jnp.log(xx) - (0.5 * jnp.power(self._sigma, -2)) * jnp.power((jnp.log(xx) - self._mu), 2),
            )
            return logpdf_val

        return vmap(logpdf_x)(x)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: log_ndtr((jnp.log(xx) - self._mu) / self._sigma))(x)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: ndtr((jnp.log(xx) - self._mu) / self._sigma))(x)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jnp.exp(self._mu + self._sigma * ndtri(xx)))(x)

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        U = jax.random.uniform(key, shape=shape)
        return self.ppf(U)

    def __repr__(self) -> str:
        string = f"LogNormal(mu={self._mu}, sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

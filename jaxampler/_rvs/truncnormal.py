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
from jax.scipy.stats import truncnorm as jax_truncnorm
from jax.typing import ArrayLike

from ..utils import jx_cast
from .crvs import ContinuousRV


class TruncNormal(ContinuousRV):
    def __init__(
        self,
        mu: ArrayLike,
        sigma: ArrayLike,
        low: ArrayLike = 0.0,
        high: ArrayLike = 1.0,
        name: str = None,
    ) -> None:
        shape, self._mu, self._sigma, self._low, self._high = jx_cast(
            mu, sigma, low, high
        )
        self.check_params()
        self._alpha = (self._low - self._mu) / self._sigma
        self._beta = (self._high - self._mu) / self._sigma
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._low < self._high), "low must be smaller than high"
        assert jnp.all(self._sigma > 0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.logpdf(
            x,
            self._alpha,
            self._beta,
            loc=self._mu,
            scale=self._sigma,
        )

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.pdf(
            x,
            self._alpha,
            self._beta,
            loc=self._mu,
            scale=self._sigma,
        )

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.logcdf(
            x,
            self._alpha,
            self._beta,
            loc=self._mu,
            scale=self._sigma,
        )

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.cdf(
            x,
            self._alpha,
            self._beta,
            loc=self._mu,
            scale=self._sigma,
        )

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key()
        shape += self._shape
        return (
            jax.random.truncated_normal(
                key,
                self._alpha,
                self._beta,
                shape=shape,
            )
            * self._sigma
            + self._mu
        )

    def __repr__(self) -> str:
        string = f"TruncNorm(mu={self._mu}, sigma={self._sigma}, low={self._low}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

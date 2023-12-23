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
from jax.scipy.stats import beta as jax_beta
from jax.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp

from .crvs import ContinuousRV


class Beta(ContinuousRV):

    def __init__(self, alpha: ArrayLike, beta: ArrayLike, name: str = None) -> None:
        self._alpha = alpha
        self._beta = beta
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._alpha > 0.0), "alpha must be positive"
        assert jnp.all(self._beta > 0.0), "beta must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.logpdf(x, self._alpha, self._beta)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.pdf(x, self._alpha, self._beta)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.logcdf(x, self._alpha, self._beta)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.cdf(x, self._alpha, self._beta)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return tfp.math.betaincinv(self._alpha, self._beta, x)

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.beta(key, self._alpha, self._beta, shape=(N,))

    def __repr__(self) -> str:
        string = f"Beta(alpha={self._alpha}, beta={self._beta}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"

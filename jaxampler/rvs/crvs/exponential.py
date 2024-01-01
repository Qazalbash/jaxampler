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
from jax.scipy.stats import expon as jax_expon
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class Exponential(ContinuousRV):

    def __init__(self, lmbda: ArrayLike, name: str = None) -> None:
        self._lmbda, = jx_cast(lmbda)
        self.check_params()
        self._scale = 1.0 / lmbda
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._lmbda > 0.0), "lmbda must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda x_: jax_expon.logpdf(x_, scale=self._scale))(x)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jax_expon.pdf(xx, scale=self._scale))(x)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jnp.where(
            xx >= 0,
            jnp.log1p(-jnp.exp(-self._lmbda * xx)),
            -jnp.inf,
        ))(x)

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jnp.where(
            xx >= 0,
            jnp.log(-jnp.log1p(-xx)) - jnp.log(self._lmbda),
            -jnp.inf,
        ))(x)

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        U = jax.random.uniform(key, shape=shape)
        rvs_val = jnp.log(-jnp.log(U)) - jnp.log(self._lmbda)
        return jnp.exp(rvs_val)

    def __repr__(self) -> str:
        string = f"Exponential(lmbda={self._lmbda}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

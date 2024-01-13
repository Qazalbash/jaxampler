# Copyright 2023 The Jaxampler Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
from jax import Array, jit, numpy as jnp

from jaxampler._src.rvs.crvs import ContinuousRV
from jaxampler._src.typing import Numeric
from jaxampler._src.utils import jx_cast


class Weibull(ContinuousRV):
    def __init__(self, lmbda: Numeric = 1.0, k: Numeric = 1.0, name: Optional[str] = None) -> None:
        shape, self._lmbda = jx_cast(lmbda)
        self._k = k
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._lmbda > 0.0), "scale must be greater than 0"
        assert jnp.all(self._k > 0.0), "concentration must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x < 0,
            -jnp.inf,
            jnp.log(self._k)
            - (self._k * jnp.log(self._lmbda))
            + (self._k - 1.0) * jnp.log(x)
            - jnp.power(x / self._lmbda, self._k),
        )

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x < 0,
            0.0,
            1.0 - jnp.exp(-jnp.power(x / self._lmbda, self._k)),
        )

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: Numeric) -> Numeric:
        return self._lmbda * jnp.power(-jnp.log(1 - x), 1.0 / self._k)

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        U = jax.random.uniform(key, shape=new_shape)
        return self.ppf_x(U)

    def __repr__(self) -> str:
        string = f"Weibull(lambda={self._lmbda}, k={self._k}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

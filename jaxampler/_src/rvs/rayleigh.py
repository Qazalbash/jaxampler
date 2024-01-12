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

from functools import partial
from typing import Optional

import jax
from jax import Array, jit, numpy as jnp

from jaxampler._src.rvs.crvs import ContinuousRV
from jaxampler._src.typing import Numeric
from jaxampler._src.utils import jx_cast


class Rayleigh(ContinuousRV):
    def __init__(self, sigma: float, name: Optional[str] = None) -> None:
        shape, self._sigma = jx_cast(sigma)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x >= 0,
            jnp.log(x) - 0.5 * jnp.power(x / self._sigma, 2) - 2 * jnp.log(self._sigma),
            -jnp.inf,
        )

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x >= 0,
            jnp.log1p(-jnp.exp(-0.5 * jnp.power(x / self._sigma, 2))),
            -jnp.inf,
        )

    @partial(jit, static_argnums=(0,))
    def logppf_x(self, x: Numeric) -> Numeric:
        return jnp.where(
            x >= 0,
            jnp.log(self._sigma) + 0.5 * jnp.log(-2 * jnp.log1p(-x)),
            -jnp.inf,
        )

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.rayleigh(key, scale=self._sigma, shape=new_shape)

    def __repr__(self) -> str:
        string = f"Rayleigh(sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

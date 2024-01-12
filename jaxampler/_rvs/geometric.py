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
from typing import Optional

import jax
from jax import Array, jit, numpy as jnp
from jax.scipy.stats import geom as jax_geom

from .._typing import Numeric
from .._utils import jx_cast
from .drvs import DiscreteRV


class Geometric(DiscreteRV):
    """Geometric Random Variable"""

    def __init__(self, p: Numeric, name: Optional[str] = None) -> None:
        shape, self._p = jx_cast(p)
        self.check_params()
        self._q = 1.0 - self._p
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._p >= 0.0), "All p must be greater than or equals to 0"

    @partial(jit, static_argnums=(0,))
    def logpmf_x(self, x: Numeric) -> Numeric:
        return jax_geom.logpmf(x, self._p)

    @partial(jit, static_argnums=(0,))
    def pmf_x(self, x: Numeric) -> Numeric:
        return jax_geom.pmf(x, self._p)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        conditions = [x < 0, x >= 0]
        choices = [jnp.zeros_like(self._q), 1.0 - jnp.power(self._q, jnp.floor(x))]
        return jnp.select(conditions, choices)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.log(self.cdf_x(x))

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.geometric(key, self._p, shape=new_shape)

    def __repr__(self) -> str:
        string = f"Geometric(p={self._p}"
        if self._name is not None:
            string += f", (name: {self._name}"
        string += ")"
        return string

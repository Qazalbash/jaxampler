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
from jax.scipy.stats import cauchy as jax_cauchy

from jaxampler._src.rvs.crvs import ContinuousRV
from jaxampler._src.typing import Numeric
from jaxampler._src.utils import jx_cast


class Cauchy(ContinuousRV):
    def __init__(self, sigma: Numeric, loc: Numeric = 0, name: Optional[str] = None) -> None:
        shape, self._sigma, self._loc = jx_cast(sigma, loc)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.logpdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.pdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.logcdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return jax_cauchy.cdf(x, self._loc, self._sigma)

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: Numeric) -> Numeric:
        return self._loc + self._sigma * jnp.tan(jnp.pi * (x - 0.5))

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.cauchy(key, shape=new_shape) * self._sigma + self._loc

    def __repr__(self) -> str:
        string = f"Cauchy(sigma={self._sigma}, loc={self._loc}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

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
from jax.scipy.stats import gamma as jax_gamma

from jaxampler._src.rvs.crvs import ContinuousRV
from jaxampler._src.typing import Numeric
from jaxampler._src.utils import jx_cast


class Gamma(ContinuousRV):
    def __init__(self, alpha: Numeric, beta: Numeric, name: Optional[str] = None) -> None:
        shape, self._alpha, self._beta = jx_cast(alpha, beta)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._alpha > 0), "All alpha must be greater than 0"
        assert jnp.all(self._beta > 0), "All beta must be greater than 0"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.logpdf(x, self._alpha, scale=1 / self._beta)

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.pdf(x, self._alpha, scale=1 / self._beta)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.logcdf(x, self._alpha, scale=1 / self._beta)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return jax_gamma.cdf(x, self._alpha, scale=1 / self._beta)

    @partial(jit, static_argnums=(0,))
    def logppf_x(self, x: Numeric) -> Numeric:
        raise NotImplementedError("Not able to find sufficient information to implement")

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.gamma(key, self._alpha, shape=new_shape) / self._beta

    def __repr__(self) -> str:
        string = f"Gamma(alpha={self._alpha}, beta={self._beta}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

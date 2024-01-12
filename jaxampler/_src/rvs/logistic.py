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
from jax.scipy.special import logit
from jax.scipy.stats import logistic as jax_logistic

from jaxampler._src.rvs.crvs import ContinuousRV
from jaxampler._src.typing import Numeric
from jaxampler._src.utils import jx_cast


class Logistic(ContinuousRV):
    def __init__(self, mu: Numeric = 0.0, scale: Numeric = 1.0, name: Optional[str] = None) -> None:
        shape, self._scale = jx_cast(scale)
        self.check_params()
        self._mu = mu
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._scale > 0.0), "scale must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_logistic.logpdf(x, self._mu, self._scale)

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_logistic.pdf(x, self._mu, self._scale)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return jax_logistic.cdf(x, self._mu, self._scale)

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: Numeric) -> Numeric:
        return self._mu + self._scale * logit(x)

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.logistic(key, shape=new_shape) * self._scale + self._mu

    def __repr__(self) -> str:
        string = f"Logistic(mu={self._mu}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

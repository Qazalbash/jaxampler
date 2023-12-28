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
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class TruncPowerLaw(ContinuousRV):

    def __init__(self, alpha: ArrayLike, low: ArrayLike = 0, high: ArrayLike = 1, name: str = None) -> None:
        self._alpha, self._low, self._high = jx_cast(alpha, low, high)
        self.check_params()
        self._beta = 1.0 + self._alpha
        self._logZ = self.logZ()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._low > 0.0), "low must be greater than 0"
        assert jnp.all(self._high > self._low), "high must be greater than low"

    @partial(jit, static_argnums=(0,))
    def logZ(self) -> ArrayLike:
        logZ_val = jnp.where(
            self._beta == 0.0,
            jnp.log(jnp.log(self._high) - jnp.log(self._low)),
            jnp.log(jnp.abs(jnp.power(self._high, self._beta) - jnp.power(self._low, self._beta))) -
            jnp.log(jnp.abs(self._beta)),
        )
        return logZ_val

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        logpdf_val = jnp.log(x) * self._alpha - self._logZ
        logpdf_val = jnp.where((x >= self._low) * (x <= self._high), logpdf_val, -jnp.inf)
        return logpdf_val

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        logcdf_val = jnp.where(
            self._beta == 0.0,
            jnp.log(jnp.log(x) - jnp.log(self._low)),
            jnp.log(jnp.abs(jnp.power(x, self._beta) - jnp.power(self._low, self._beta))) -
            jnp.log(jnp.abs(self._beta)),
        )
        logcdf_val -= self._logZ
        logcdf_val = jnp.where(x >= self._low, logcdf_val, -jnp.inf)
        logcdf_val = jnp.where(x <= self._high, logcdf_val, jnp.log(1.0))
        return logcdf_val

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        logppf_val = jnp.where(
            self._beta == 0.0,
            x * jnp.log(self._high) + (1.0 - x) * jnp.log(self._low),
            jnp.power(self._beta, -1) * jnp.log(x * jnp.power(self._high, self._beta) +
                                                (1.0 - x) * jnp.power(self._low, self._beta)))
        logppf_val = jnp.where(x >= 0.0, logppf_val, -jnp.inf)
        logppf_val = jnp.where(x <= 1.0, logppf_val, jnp.log(1.0))
        return logppf_val

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        shape = (N,) + (self._alpha.shape or (1,))
        U = jax.random.uniform(key, shape=shape, dtype=jnp.float32)
        rvs_val = self.ppf(U)
        return rvs_val

    def __repr__(self) -> str:
        string = f"TruncPowerLaw(alpha={self._alpha}, low={self._low}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

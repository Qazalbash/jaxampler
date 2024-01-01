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
from jax.scipy.stats import chi2 as jax_chi2
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class Chi2(ContinuousRV):

    def __init__(self, nu: ArrayLike, name: str = None) -> None:
        self._nu, = jx_cast(nu)
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._nu.dtype == jnp.int32), "nu must be an integer"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jax_chi2.logpdf(xx, self._nu))(x)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jax_chi2.pdf(xx, self._nu))(x)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jax_chi2.logcdf(xx, self._nu))(x)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jax_chi2.cdf(xx, self._nu))(x)

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Not able to find sufficient information to implement")

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.chisquare(key, self._nu, shape=shape)

    def __repr__(self) -> str:
        string = f"Chi2(nu={self._nu}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

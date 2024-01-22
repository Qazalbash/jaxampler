#  Copyright 2023 The Jaxampler Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from functools import partial
from typing import Optional

from jax import jit, numpy as jnp, vmap
from jaxtyping import Array

from ..jobj import JObj
from ..typing import Numeric


class GenericRV(JObj):
    """Generic random variable class."""

    def __init__(self, name: Optional[str] = None, shape: tuple[int, ...] = ()) -> None:
        self._shape = shape
        super().__init__(name=name)

    def check_params(self) -> None:
        raise NotImplementedError

    # POINT VALUED

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self.logcdf_x(*x))

    @partial(jit, static_argnums=(0,))
    def logppf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self.logppf_x(*x))

    # VECTOR VALUED

    @partial(jit, static_argnums=(0,))
    def logcdf_v(self, *x: Numeric) -> Numeric:
        return vmap(self.logcdf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def cdf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self.logcdf_v(*x))

    @partial(jit, static_argnums=(0,))
    def logppf_v(self, *x: Numeric) -> Numeric:
        return vmap(self.logppf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def ppf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self.logppf_v(*x))

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return self._rvs(shape=new_shape, key=key)

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        raise NotImplementedError

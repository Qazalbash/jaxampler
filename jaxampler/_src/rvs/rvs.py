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
from typing import Callable, Optional

import numpy as np
from jax import jit, lax, numpy as jnp, vmap
from jax._src import core
from jaxtyping import Array

from ..jobj import JObj
from ..typing import Numeric


class RandomVariable(JObj):
    """Generic random variable class."""

    def __init__(self, name: Optional[str] = None, shape: tuple[int, ...] = ()) -> None:
        self._shape = shape
        super().__init__(name=name)

    def check_params(self) -> None:
        raise NotImplementedError

    # POINT VALUED

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logcdf_x(*x))

    @partial(jit, static_argnums=(0,))
    def _logppf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _ppf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logppf_x(*x))

    # VECTOR VALUED

    @partial(jit, static_argnums=(0,))
    def _logcdf_v(self, *x: Numeric) -> Numeric:
        return vmap(self._logcdf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def _cdf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logcdf_v(*x))

    @partial(jit, static_argnums=(0,))
    def _logppf_v(self, *x: Numeric) -> Numeric:
        return vmap(self._logppf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def _ppf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logppf_v(*x))

    # XXF FACTORY METHODS
    @staticmethod
    def _pv_factory(
        func_p: Callable[..., Numeric],
        func_v: Callable[..., Numeric],
        *x: Numeric,
    ) -> Numeric:
        # partially taken from the implementation of `jnp.broadcast_arrays`
        shapes = [np.shape(arg) for arg in x]
        if not shapes or all(core.definitely_equal_shape(shapes[0], s) for s in shapes):
            shape = shapes[0]
        else:
            shape: tuple[int, ...] = lax.broadcast_shapes(*shapes)

        if len(shape) < 2:
            return func_p(*x)
        return func_v(*x)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, *x: Numeric) -> Numeric:
        return self._pv_factory(self._logcdf_x, self._logcdf_v, *x)

    @partial(jit, static_argnums=(0,))
    def cdf(self, *x: Numeric) -> Numeric:
        return self._pv_factory(self._cdf_x, self._cdf_v, *x)

    @partial(jit, static_argnums=(0,))
    def logppf(self, *x: Numeric) -> Numeric:
        return self._pv_factory(self._logppf_x, self._logppf_v, *x)

    @partial(jit, static_argnums=(0,))
    def ppf(self, *x: Numeric) -> Numeric:
        return self._pv_factory(self._ppf_x, self._ppf_v, *x)

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return self._rvs(shape=new_shape, key=key)

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        raise NotImplementedError

    # POINT VALUED

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _pdf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logpdf_x(*x))

    # VECTOR VALUED

    @partial(jit, static_argnums=(0,))
    def _logpdf_v(self, *x: Numeric) -> Numeric:
        return vmap(self._logpdf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def _pdf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self._logpdf_v(*x))

    # FACTORY METHODS

    @partial(jit, static_argnums=(0,))
    def logpdf(self, *x: Numeric) -> Numeric:
        return self._pv_factory(self._logpdf_x, self._logpdf_v, *x)

    @partial(jit, static_argnums=(0,))
    def pdf(self, *x: Numeric) -> Numeric:
        return self._pv_factory(self._pdf_x, self._pdf_v, *x)

    # POINT VALUED

    @partial(jit, static_argnums=(0,))
    def _logpmf_x(self, *k: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def _pmf_x(self, *k: Numeric) -> Numeric:
        return jnp.exp(self._logpmf_x(*k))

    # VECTOR VALUED

    @partial(jit, static_argnums=(0,))
    def _logpmf_v(self, *k: Numeric) -> Numeric:
        return vmap(self._logpmf_x, in_axes=0)(*k)

    @partial(jit, static_argnums=(0,))
    def _pmf_v(self, *k: Numeric) -> Numeric:
        return jnp.exp(self._logpmf_v(*k))

    # FACTORY METHODS

    @partial(jit, static_argnums=(0,))
    def logpmf(self, *k: Numeric) -> Numeric:
        return self._pv_factory(self._logpmf_x, self._logpmf_v, *k)

    @partial(jit, static_argnums=(0,))
    def pmf(self, *k: Numeric) -> Numeric:
        return self._pv_factory(self._pmf_x, self._pmf_v, *k)

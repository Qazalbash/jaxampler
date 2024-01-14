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

import jax
from jax import Array, jit, numpy as jnp
from jax.scipy.stats import chi2 as jax_chi2

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class Chi2(ContinuousRV):
    def __init__(self, nu: Numeric, name: Optional[str] = None) -> None:
        shape, self._nu = jx_cast(nu)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._nu.dtype == jnp.int32), "nu must be an integer"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_chi2.logpdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_chi2.pdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jax_chi2.logcdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return jax_chi2.cdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def logppf_x(self, x: Numeric) -> Numeric:
        raise NotImplementedError("Not able to find sufficient information to implement")

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.chisquare(key, self._nu, shape=new_shape)

    def __repr__(self) -> str:
        string = f"Chi2(nu={self._nu}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string

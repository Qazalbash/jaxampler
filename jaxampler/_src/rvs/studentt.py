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
from typing import Any, Optional

import jax
from jax import Array, jit, numpy as jnp
from jax.scipy.special import betainc
from jax.scipy.stats import t as jax_t

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class StudentT(ContinuousRV):
    def __init__(
        self,
        df: Numeric | Any,
        loc: Numeric | Any = 0.0,
        scale: Numeric | Any = 1.0,
        name: Optional[str] = None,
    ) -> None:
        shape, self._df, self._loc, self._scale = jx_cast(df, loc, scale)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._df > 0.0), "nu must be positive"
        assert jnp.all(self._scale > 0.0), "scale must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_t.logpdf(
            x=x,
            df=self._df,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_t.pdf(
            x=x,
            df=self._df,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.log(self.cdf_x(x))

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return 1 - 0.5 * betainc(
            a=self._df * 0.5,
            b=0.5,
            x=1 / (1 + (jnp.power((x - self._loc) / self._scale, 2) / self._df)),
        )

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: Numeric) -> Numeric:
        """A method is addressed in this paper https://www.homepages.ucl.ac.uk/~ucahwts/lgsnotes/JCF_Student.pdf"""
        raise NotImplementedError

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + self._scale * jax.random.t(key=key, df=self._df, shape=shape)

    def __repr__(self) -> str:
        string = f"StudentT(nu={self._df}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
